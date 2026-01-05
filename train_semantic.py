#!/usr/bin/env python3
"""
语义增强的3D高斯训练脚本 - 全程0.5分辨率版本
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from semantic_3dgs.trainer.semantic_gaussian_trainer import SemanticGaussianTrainer
from semantic_3dgs.core.adaptive_densification import SemanticAdaptiveDensification, DensificationConfig
import time
import gc
import json
import pickle
import copy

# 尝试导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 场景特定的语义配置
SCENE_SEMANTIC_CONFIGS = {
    'truck': {
        'important_labels': ["truck", "vehicle", "cargo",
        "wheel", "tire",
        "cabin", "window",
        "wood", "crate", "box",  # 木质货箱很重要
        "text", "rust"],
    },

    'train': {
        'important_labels': ["train", "locomotive", "track", "railway", "signal", "wheel"],
    },

    'playground': {
        'important_labels': ["toy", "book", "text", "letter",
        "furniture", "shelf", "table",
        "mat", "rug", "cushion",
        "decoration", "wall art"],
    },

    'drjohnson': {
        'important_labels': [ "portrait", "painting", "face", "picture", "frame",
        "rug", "carpet", "pattern", "textile",
        "door", "chair", "furniture"],
    },

    # ===== Mip-NeRF 360场景配置 =====
    'kitchen': {
        'important_labels': [
            "toy", "lego", "car", "vehicle",
            "table", "wood", "furniture",
            "mat", "rug", "fabric"
        ],
    },

    'room': {
        'important_labels': [
            "furniture", "chair", "sofa", "table",
            "speaker", "electronics",
            "bottle", "bowl", "shoe",
            "curtain", "fabric"
        ],
    },

    'bicycle': {
        'important_labels': [
            "bicycle", "bike",
            "wheel", "tire", "spoke",
            "chain", "handlebar", "seat",
            "frame", "pedal"
        ],
    },

    'garden': {
        'important_labels': [
            "plant", "flower", "tree", "grass",
            "table", "furniture", "wood",
            "pot", "decoration",
            "building", "brick", "window"
        ],
    },

    'counter': {
        'important_labels': [
            "bowl", "container", "cutting board",
            "food", "fruit", "vegetable", "onion",
            "pot", "pan", "cookware",
            "counter", "kitchen equipment"
        ],
    },

    'default': {
        'important_labels': ["face", "text", "sign", "person", "car"],
    }
}


def get_scene_config(dataset_path, force_scene_type=None):
    """根据数据集路径自动检测场景类型并返回配置"""
    if force_scene_type and force_scene_type in SCENE_SEMANTIC_CONFIGS:
        scene_name = force_scene_type
    else:
        scene_name = 'default'
        path_lower = dataset_path.lower()
        for scene in ['truck', 'train', 'playground', 'drjohnson', 'kitchen', 'room', 'bicycle', 'garden', 'counter']:
            if scene in path_lower:
                scene_name = scene
                break

    print(f"\n=== Scene type: '{scene_name}' ===")
    config = SCENE_SEMANTIC_CONFIGS.get(scene_name, SCENE_SEMANTIC_CONFIGS['default'])
    return scene_name, config


def compute_scene_extent_from_cameras(cameras):
    """基于相机位置计算场景范围 - 原始3DGS方式"""
    cam_centers = []

    for cam in cameras:
        # 获取相机中心（世界坐标）
        if hasattr(cam, 'camera_center'):
            cam_center = cam.camera_center
            if not isinstance(cam_center, torch.Tensor):
                cam_center = torch.tensor(cam_center)
            cam_centers.append(cam_center)
        elif hasattr(cam, 'R') and hasattr(cam, 'T'):
            R = cam.R if isinstance(cam.R, torch.Tensor) else torch.tensor(cam.R)
            T = cam.T if isinstance(cam.T, torch.Tensor) else torch.tensor(cam.T)
            cam_center = -torch.matmul(R.T, T)
            cam_centers.append(cam_center)

    if len(cam_centers) == 0:
        print("[Warning] No camera centers found, using default extent")
        return 10.0

    # 计算相机的包围球半径
    cam_centers = torch.stack(cam_centers)
    center = cam_centers.mean(dim=0)
    radius = torch.norm(cam_centers - center, dim=1).max().item()

    # 原始3DGS使用1.1倍作为安全边界
    scene_extent = radius * 1.1

    print(f"[Scene Extent] Computed from {len(cameras)} cameras")
    print(f"  Camera center range: {cam_centers.min(dim=0)[0]} to {cam_centers.max(dim=0)[0]}")
    print(f"  Scene radius: {radius:.3f}")
    print(f"  Scene extent (radius * 1.1): {scene_extent:.3f}")

    return scene_extent

def load_cached_initialization(scene_type, dataset_path):
    """尝试加载缓存的语义初始化"""
    cache_dir = "cache/semantic_init"
    metadata_file = os.path.join(cache_dir, f"{scene_type}_metadata.json")

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            cache_file = metadata['cache_file']
            if os.path.exists(cache_file):
                print(f"\n=== Loading cached semantic initialization ===")
                print(f"Cache file: {cache_file}")

                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                # 转换numpy回tensor
                gaussian_params = {}
                for key, value in cache_data['gaussian_params'].items():
                    if isinstance(value, np.ndarray):
                        gaussian_params[key] = torch.from_numpy(value).cuda()
                    else:
                        gaussian_params[key] = value

                print(f"✓ Loaded {gaussian_params['positions'].shape[0]} cached gaussians")
                print(f"✓ Skip time-consuming SAM segmentation")

                spatial_lr_scale = cache_data.get('spatial_lr_scale', 1.0)
                return gaussian_params, spatial_lr_scale
            else:
                print(f"Cache file not found: {cache_file}")
        except Exception as e:
            print(f"Error loading cache: {e}")

    return None, None


def create_scaled_camera_view(original_cam, scale=1.0):
    """
    创建缩放版本的相机视图，用于渲染
    不修改原始相机对象，返回一个包装对象
    """
    if scale == 1.0:
        return original_cam

    # 创建一个动态的包装类
    class ScaledCameraWrapper:
        def __init__(self, cam, scale):
            self._original_cam = cam
            self._scale = scale

            # 计算缩放后的尺寸
            self.image_width = int(cam.image_width * scale)
            self.image_height = int(cam.image_height * scale)

            # 缩放GT图像
            if hasattr(cam, 'original_image'):
                original_image = cam.original_image.unsqueeze(0)  # [1, 3, H, W]
                scaled_image = F.interpolate(
                    original_image,
                    size=(self.image_height, self.image_width),
                    mode='bilinear',
                    align_corners=False
                )
                self.original_image = scaled_image.squeeze(0)  # [3, H, W]

        def __getattr__(self, name):
            # 对于没有特殊处理的属性，直接返回原始相机的属性
            return getattr(self._original_cam, name)

    return ScaledCameraWrapper(original_cam, scale)


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, scene_type=None):
    """主训练函数 - 全程0.5分辨率版本"""

    # 获取场景配置
    scene_name, scene_config = get_scene_config(dataset.source_path, scene_type)

    # 检查预处理掩码
    preprocessed_masks_dir = os.path.join(dataset.source_path, "sam_preprocessed", "masks")
    use_preprocessed_masks = os.path.exists(preprocessed_masks_dir)

    if use_preprocessed_masks:
        print(f"\n✓ Found preprocessed SAM masks at: {preprocessed_masks_dir}")
        print("  Will use preprocessed masks to save memory and time!")

    # 创建配置
    trainer_config = Namespace(
        # SAM配置
        sam_checkpoint="checkpoints/sam/sam_vit_h_4b8939.pth",
        clip_model="ViT-B/32",
        use_preprocessed_masks=use_preprocessed_masks,
        preprocessed_masks_dir=preprocessed_masks_dir if use_preprocessed_masks else None,
        use_lightweight_sam=False,

        # 语义配置
        #semantic_prompts=scene_config['semantic_prompts'],
        num_init_images=5,

        # SAM-RPS配置
        grad_threshold=0.0004,
        percent_dense=0.01,
        opacity_cull=0.005,
        semantic_weight=0.3,
        #protected_labels=scene_config['protected_labels'],
        max_gaussians=1100000,

        # 内存管理参数
        memory_aware=True,
        target_memory_usage=0.85,
        min_free_memory_gb=3.0,
        aggressive_pruning=False,
        pruning_min_opacity=0.005,
        pruning_max_scale=0.1,
        max_operations_per_iter=5000,

        # 其他参数
        opacity_reset_interval=3000,

        # SAM-ES配置
        geometric_threshold=0.15,
        semantic_iou_threshold=0.5,
        max_exploratory_points=200,
        max_holes_per_iter=50,
        #exploration_focus_areas=scene_config.get('exploration_focus', []),
        debug_mode=False
    )

    # 初始化语义训练器
    print("\n=== Initializing Semantic Gaussian Trainer ===")
    semantic_trainer = SemanticGaussianTrainer(trainer_config)
    if hasattr(semantic_trainer, 'initializer'):
        semantic_trainer.initializer.sh_degree = dataset.sh_degree  # 使用数据集的sh_degree

    print("Loading SAM-RPS module...")

    # 创建高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # 计算场景范围 - 只做一次！
    train_cameras = scene.getTrainCameras()
    scene_extent = compute_scene_extent_from_cameras(train_cameras)
    print(f"✓ Scene extent computed from cameras: {scene_extent:.3f}")

    # 设置场景范围到 densifier - 使用新的 set_scene_extent 方法
    if hasattr(semantic_trainer.densifier, 'set_scene_extent'):
        semantic_trainer.densifier.set_scene_extent(scene_extent)
        print("✓ Scene extent set for densifier (fixed, will not change)")

    # 对 explorer 也设置固定值
    if hasattr(semantic_trainer, 'explorer'):
        semantic_trainer.explorer.scene_extent = scene_extent  # 直接赋值，不调用方法
        print("✓ Scene extent set for explorer")

    # 初始化标记
    semantic_modules_released = False

    # 语义初始化
    if scene.loaded_iter:
        print(f"Loading checkpoint from iteration {scene.loaded_iter}")
    else:
        # ========== 暂时禁用语义初始化，使用原始3DGS方式 ==========
        use_semantic_init = True  # 设置为 True 恢复语义初始化

        if use_semantic_init:
            # 首先尝试加载缓存
            cached_params, cached_lr_scale = load_cached_initialization(scene_name, dataset.source_path)

            if cached_params is not None:
                # 使用缓存的参数（保持原逻辑）
                gaussian_params = cached_params
                spatial_lr_scale = cached_lr_scale or dataset.spatial_lr_scale
                gaussians.create_from_semantic_init(gaussian_params, spatial_lr_scale)
                print("✓ 从缓存加载高斯模型")
            else:
                # 新的语义增强初始化（使用完整点云）
                print("\n=== 使用语义增强初始化（完整点云 + SAM/CLIP）===")
                print(f"场景类型: {scene_name}")
                print("策略: 原始3DGS初始化 + 语义重要性标记")

                # 获取点云（与原始3DGS初始化相同的逻辑）
                if hasattr(scene, 'point_cloud'):
                    point_cloud = scene.point_cloud
                    print(f"使用场景点云: {len(point_cloud.points)}个点")
                else:
                    # 从文件加载
                    ply_path = os.path.join(dataset.source_path, "sparse/0/points3D.ply")
                    if os.path.exists(ply_path):
                        from plyfile import PlyData
                        plydata = PlyData.read(ply_path)
                        vertices = plydata['vertex']
                        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
                        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
                        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

                        from utils.graphics_utils import BasicPointCloud
                        point_cloud = BasicPointCloud(points=positions, colors=colors, normals=normals)
                        print(f"从文件加载点云: {len(positions)}个点")
                    else:
                        raise FileNotFoundError(f"找不到点云文件: {ply_path}")

                # 准备图像和相机参数（用于语义分析）
                train_images = []
                train_cameras = []
                train_image_names = []
                num_views_for_semantic = 10  # 增加到10个视图

                # 选择分布较好的视图（不要只选前N个，要选择不同角度的）
                all_train_cams = scene.getTrainCameras()
                selected_indices = []

                if len(all_train_cams) <= num_views_for_semantic:
                    selected_indices = list(range(len(all_train_cams)))
                else:
                    # 均匀采样，确保覆盖不同角度
                    step = len(all_train_cams) // num_views_for_semantic
                    selected_indices = [i * step for i in range(num_views_for_semantic)]

                for idx in selected_indices:
                    cam = all_train_cams[idx]

                    # 获取图像
                    gt_image = cam.original_image.permute(1, 2, 0).cpu().numpy()
                    gt_image = (gt_image * 255).astype(np.uint8)
                    train_images.append(gt_image)

                    # 获取图像名称
                    if hasattr(cam, 'image_name'):
                        train_image_names.append(cam.image_name)
                    else:
                        train_image_names.append(f"image_{idx:04d}")

                    # 构建相机参数字典
                    camera_params = {}

                    # 计算内参矩阵
                    import math
                    W = cam.image_width
                    H = cam.image_height

                    # 从视场角计算焦距
                    fx = W / (2 * math.tan(cam.FoVx / 2))
                    fy = H / (2 * math.tan(cam.FoVy / 2))
                    cx = W / 2.0
                    cy = H / 2.0

                    camera_params['K'] = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)

                    # 获取外参矩阵
                    # 注意：world_view_transform 是列主序的，需要转置
                    W2C = cam.world_view_transform.T.cpu().numpy()
                    camera_params['R'] = W2C[:3, :3]
                    camera_params['t'] = W2C[:3, 3]

                    # 添加调试信息
                    if idx == selected_indices[0]:
                        print(f"  Camera {idx} intrinsics:")
                        print(f"    Image size: {W}x{H}")
                        print(f"    FoV: {math.degrees(cam.FoVx):.1f}° x {math.degrees(cam.FoVy):.1f}°")
                        print(f"    Focal length: fx={fx:.1f}, fy={fy:.1f}")
                        print(f"    Principal point: cx={cx:.1f}, cy={cy:.1f}")

                    train_cameras.append(camera_params)

                print(f"  Selected {len(selected_indices)} views from {len(all_train_cams)} total views")
                print(f"  View indices: {selected_indices}")

                # 调用新的初始化方法
                # 调用初始化方法时
                gaussian_params = semantic_trainer.initializer.initialize_from_full_pointcloud_with_semantic(
                    train_images,
                    point_cloud,
                    train_cameras,
                    None,  # text_prompts不需要了
                    train_image_names,
                    important_labels=scene_config.get('important_labels', ["face", "text", "sign"])
                )

                # 获取相机范围
                spatial_lr_scale = scene.cameras_extent
                print(f"使用spatial_lr_scale (cameras_extent): {spatial_lr_scale}")

                # 创建高斯模型
                gaussians.create_from_semantic_init(gaussian_params, spatial_lr_scale)

                # 设置场景范围
                train_cameras_all = scene.getTrainCameras()
                scene_extent = compute_scene_extent_from_cameras(train_cameras_all)

                if hasattr(semantic_trainer.densifier, 'set_scene_extent'):
                    semantic_trainer.densifier.set_scene_extent(scene_extent)
                    print(f"✓ 为密度控制器设置场景范围: {scene_extent:.3f}")

                if hasattr(semantic_trainer, 'explorer'):
                    semantic_trainer.explorer.scene_extent = scene_extent
                    print(f"✓ 为探索器设置场景范围: {scene_extent:.3f}")

                # 初始化梯度累积器
                num_points = len(gaussians._xyz)
                gaussians.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
                gaussians.denom = torch.zeros((num_points, 1), device="cuda")
                gaussians.max_radii2D = torch.zeros(num_points, device="cuda")
                print("✓ 梯度累积器已初始化")

                # 设置优化器
                gaussians.training_setup(opt)
                print("✓ 优化器已初始化")

                # 释放语义模块
                print("\n=== 释放语义初始化模块 ===")
                if hasattr(semantic_trainer, 'initializer'):
                    del semantic_trainer.initializer
                semantic_trainer.cleanup_sam()
                gc.collect()
                torch.cuda.empty_cache()
                semantic_modules_released = True

                # 打印内存状态
                if torch.cuda.is_available():
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    print(f"✓ 清理后内存: {free_memory / 1e9:.1f}GB空闲 / {total_memory / 1e9:.1f}GB总计")

                if not use_preprocessed_masks:
                    print("\nTip: Run 'python scripts/preprocess_sam.py' to cache SAM masks for faster future runs!")
        else:
            # ========== 使用原始3DGS初始化 ==========
            print("\n=== Using Original 3DGS Initialization ===")
            print("Semantic initialization is disabled for comparison")

            # 原始3DGS不需要语义训练器的初始化模块
            if hasattr(semantic_trainer, 'initializer'):
                del semantic_trainer.initializer
            semantic_trainer.cleanup_sam()
            gc.collect()
            torch.cuda.empty_cache()
            semantic_modules_released = True

            # 检查点云信息
            if hasattr(scene, 'point_cloud'):
                point_cloud = scene.point_cloud
                print(f"Point cloud has {len(point_cloud.points)} points")
            else:
                # 从文件读取点云
                ply_path = os.path.join(dataset.source_path, "sparse/0/points3D.ply")
                if os.path.exists(ply_path):
                    from plyfile import PlyData
                    plydata = PlyData.read(ply_path)
                    vertices = plydata['vertex']
                    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
                    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
                    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

                    from utils.graphics_utils import BasicPointCloud
                    point_cloud = BasicPointCloud(points=positions, colors=colors, normals=normals)
                    print(f"Loaded point cloud with {len(positions)} points")
                else:
                    raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

            # 获取相机数量 - 这是 create_from_pcd 需要的第二个参数
            train_cameras = scene.getTrainCameras()
            cam_infos = len(train_cameras)
            print(f"Number of training cameras: {cam_infos}")

            # 使用场景的相机范围作为 spatial_lr_scale
            spatial_lr_scale = scene.cameras_extent
            print(f"Using spatial_lr_scale (cameras_extent): {spatial_lr_scale}")

            # 原始3DGS初始化 - 添加缺失的 cam_infos 参数
            gaussians.create_from_pcd(point_cloud, cam_infos, spatial_lr_scale)

            if scene_name == 'truck':
                print("\n[Semantic Enhancement] Setting semantic importance for truck scene...")

                positions = gaussians._xyz.detach()  # 使用detach()来避免梯度问题
                num_points = positions.shape[0]

                # 初始化为默认值
                semantic_importance = gaussians._semantic_importance.clone()  # 已经是0.5

                # 策略1：基于高度 - 卡车主体通常在中间高度
                heights = positions[:, 1]  # Y轴
                height_percentiles = torch.quantile(heights, torch.tensor([0.2, 0.8], device="cuda"))
                truck_body_mask = (heights > height_percentiles[0]) & (heights < height_percentiles[1])
                semantic_importance[truck_body_mask] = 0.7

                # 策略2：基于位置密度 - 密集区域可能是主体
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=20)
                nn.fit(positions.cpu().numpy())  # 现在可以安全地调用numpy()
                distances, _ = nn.kneighbors()
                density = 1.0 / (distances.mean(axis=1) + 1e-6)
                density_tensor = torch.from_numpy(density).cuda()
                density_percentile = torch.quantile(density_tensor, 0.7)
                high_density_mask = density_tensor > density_percentile
                semantic_importance[high_density_mask] = torch.maximum(
                    semantic_importance[high_density_mask],
                    torch.tensor(0.8, device="cuda")
                )

                # 策略3：基于初始颜色（如果有深色区域，可能是轮胎）
                colors = gaussians._features_dc.detach()[:, 0, :] * 0.28209479177387814  # SH to RGB, 也要detach
                dark_mask = colors.max(dim=1)[0] < 0.3
                semantic_importance[dark_mask] = 0.9  # 轮胎等深色部件给最高重要性

                # 更新语义重要性 - 直接赋值，不影响梯度
                gaussians._semantic_importance = semantic_importance

                # 统计
                high_importance = (semantic_importance > 0.7).sum().item()
                medium_importance = ((semantic_importance >= 0.5) & (semantic_importance <= 0.7)).sum().item()
                low_importance = (semantic_importance < 0.5).sum().item()

                print(f"  Total points: {num_points}")
                print(f"  High importance (>0.7): {high_importance} ({high_importance / num_points * 100:.1f}%)")
                print(
                    f"  Medium importance (0.5-0.7): {medium_importance} ({medium_importance / num_points * 100:.1f}%)")
                print(f"  Low importance (<0.5): {low_importance} ({low_importance / num_points * 100:.1f}%)")
                print(f"  Mean importance: {semantic_importance.mean().item():.3f}")
                print(f"  Max importance: {semantic_importance.max().item():.3f}")

            # 确保优化器被初始化
            gaussians.training_setup(opt)

            # 初始化完成后的检查
            print(f"\n[Original 3DGS Init Complete]")
            print(f"  Number of gaussians: {gaussians._xyz.shape[0]}")
            print(f"  Position range: {gaussians._xyz.min(dim=0)[0]} to {gaussians._xyz.max(dim=0)[0]}")
            pos_center = gaussians._xyz.mean(dim=0)
            pos_extent = torch.norm(gaussians._xyz - pos_center, dim=1).max().item()
            print(f"  Actual extent: {pos_extent:.3f}")
            print(f"  Scene extent: {scene.cameras_extent:.3f}")
            print(f"  Ratio: {pos_extent / scene.cameras_extent:.1f}x")

            # 打印内存状态
            if torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                print(f"✓ Memory after init: {free_memory / 1e9:.1f}GB free / {total_memory / 1e9:.1f}GB total")

    # 确保优化器被初始化（如果还没有）
    if gaussians.optimizer is None:
        gaussians.training_setup(opt)
        print("✓ Optimizer initialized")

    # 设置背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 初始化进度条
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    # 训练统计
    viewpoint_stack = None
    first_iter = 0

    # 调试信息间隔
    debug_interval = 500
    monitor_interval = 100
    memory_check_interval = 100

    # 统计变量
    last_num_gaussians = 0
    total_splits = 0
    total_clones = 0
    total_prunes = 0

    # 内存管理变量
    consecutive_high_memory = 0

    # ===== 渲染分辨率控制 - 全程0.5 =====
    render_resolution_scale = 0.5  # 固定使用0.5倍分辨率
    print(f"\n[Resolution] Using fixed render resolution: {render_resolution_scale}x for entire training")

    # 记录分辨率统计（简化版）
    resolution_stats = {
        'fixed_scale': render_resolution_scale,
        'memory_warnings': 0
    }

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # === 简单监控输出 ===
        if iteration % monitor_interval == 0 and iteration > 0:
            num_gaussians = gaussians._xyz.shape[0]
            growth = num_gaussians - last_num_gaussians

            # 计算质量指标
            avg_opacity = torch.sigmoid(gaussians._opacity).mean().item()
            avg_scale = torch.exp(gaussians._scaling).mean().item()
            max_scale = torch.exp(gaussians._scaling).max().item()

            print(f"\n[Monitor {iteration}] Gaussians: {num_gaussians} (+{growth}), "
                  f"Loss: {ema_loss_for_log:.4f}, "
                  f"Opacity: {avg_opacity:.3f}, "
                  f"Scale: avg={avg_scale:.4f}, max={max_scale:.4f}, "
                  f"Render: {render_resolution_scale}x")

            last_num_gaussians = num_gaussians

        # === 详细调试输出 ===
        if iteration % debug_interval == 0 and iteration > 0:
            print(f"\n{'=' * 60}")
            print(f"[Debug Stats at iteration {iteration}]")
            print(f"{'=' * 60}")

            # 基本统计
            num_gaussians = gaussians._xyz.shape[0]
            print(f"Total Gaussians: {num_gaussians}")
            print(f"Render Resolution: {render_resolution_scale}x (fixed)")

            # 梯度统计
            if hasattr(gaussians, 'xyz_gradient_accum') and gaussians.denom.sum() > 0:
                # 只计算有效的梯度（denom > 0）
                valid_mask = gaussians.denom.squeeze() > 0
                if valid_mask.sum() > 0:
                    avg_grads = gaussians.xyz_gradient_accum[valid_mask] / gaussians.denom[valid_mask]
                    avg_grad = avg_grads.mean().item()
                    max_grad = avg_grads.max().item()
                    print(f"Gradient stats: avg={avg_grad:.6f}, max={max_grad:.6f}")

            # 尺度分布
            scales = torch.exp(gaussians._scaling)
            scale_percentiles = torch.quantile(scales.max(dim=-1)[0],
                                               torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device="cuda"))
            print(f"Scale distribution: P10={scale_percentiles[0]:.4f}, "
                  f"P25={scale_percentiles[1]:.4f}, P50={scale_percentiles[2]:.4f}, "
                  f"P75={scale_percentiles[3]:.4f}, P90={scale_percentiles[4]:.4f}")

            # 计算会被分裂的高斯数量
            if semantic_trainer.densifier.scene_extent:
                scale_threshold = 0.01 * semantic_trainer.densifier.scene_extent
                large_gaussians = (scales.max(dim=-1)[0] > scale_threshold).sum().item()
                print(f"Large gaussians (>{scale_threshold:.4f}): {large_gaussians} "
                      f"({large_gaussians / num_gaussians * 100:.1f}%)")

            # 不透明度分布
            opacities = torch.sigmoid(gaussians._opacity)
            opacity_percentiles = torch.quantile(opacities.squeeze(),
                                                 torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device="cuda"))
            print(f"Opacity distribution: P10={opacity_percentiles[0]:.3f}, "
                  f"P25={opacity_percentiles[1]:.3f}, P50={opacity_percentiles[2]:.3f}, "
                  f"P75={opacity_percentiles[3]:.3f}, P90={opacity_percentiles[4]:.3f}")

            # 语义统计
            if hasattr(gaussians, '_is_exploratory'):
                num_exploratory = gaussians._is_exploratory.sum().item()
                print(f"Exploratory gaussians: {num_exploratory} ({num_exploratory / num_gaussians * 100:.1f}%)")

            # 密度控制统计
            stats = semantic_trainer.densifier.get_statistics()
            print(f"\nDensification stats:")
            print(f"  Total splits: {stats['total_splits']}")
            print(f"  Total clones: {stats['total_clones']}")
            print(f"  Total prunes: {stats['total_prunes']}")
            print(f"  Semantic boosts: {stats['semantic_boosts']}")

            # 分辨率统计
            print(f"\nResolution stats:")
            print(f"  Fixed render scale: {render_resolution_scale}x")
            print(f"  Memory warnings: {resolution_stats['memory_warnings']}")

            print(f"{'=' * 60}\n")

        # === 内存监控 ===
        if iteration % memory_check_interval == 0:
            if torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                used_memory = (total_memory - free_memory) / 1e9
                free_gb = free_memory / 1e9
                usage_percent = (used_memory / (total_memory / 1e9)) * 100

                # 计算渲染缓冲区大小
                if 'render_cam' in locals():
                    render_buffer_size = render_cam.image_width * render_cam.image_height * 4 * 8 / 1e9
                else:
                    # 估算
                    original_width = scene.getTrainCameras()[0].image_width
                    original_height = scene.getTrainCameras()[0].image_height
                    render_buffer_size = (original_width * render_resolution_scale) * \
                                         (original_height * render_resolution_scale) * 4 * 8 / 1e9

                print(f"\n[Memory Check] Iteration {iteration}")
                print(f"  GPU Memory: {used_memory:.1f}GB used ({usage_percent:.1f}%), "
                      f"{free_gb:.1f}GB free")
                print(f"  Total Gaussians: {gaussians._xyz.shape[0]}")
                print(f"  Render resolution: {render_resolution_scale}x (fixed)")
                print(f"  Estimated render buffer: ~{render_buffer_size:.3f}GB")

                # 内存压力警告
                if usage_percent > 90:
                    print(f"  ⚠️  High memory usage! Consider reducing max_gaussians")
                    consecutive_high_memory += 1
                    resolution_stats['memory_warnings'] += 1
                elif usage_percent > 80:
                    print(f"  ⚠️  Memory usage above 80%, monitoring closely")
                else:
                    consecutive_high_memory = 0

                # 极低内存时的紧急清理
                if free_gb < 1.5:
                    print(f"  🚨 Critical memory! Forcing cleanup...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    time.sleep(0.1)  # 给GPU一点时间

        # 定期内存清理
        if iteration % 50 == 0 and iteration > 0:
            torch.cuda.empty_cache()
            gc.collect()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 每1000次迭代做一次深度清理
        if iteration % 1000 == 0 and iteration > 0:
            print(f"\n[Deep Clean] Performing deep memory cleanup at iteration {iteration}")

            # 同步所有CUDA操作
            torch.cuda.synchronize()

            # 清理优化器中的死状态
            if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
                # 获取当前参数数量
                current_size = len(gaussians._xyz)

                # 检查并清理优化器状态
                dead_states = []
                for tensor, state in gaussians.optimizer.state.items():
                    if not any(tensor is p for group in gaussians.optimizer.param_groups for p in group['params']):
                        dead_states.append(tensor)

                for tensor in dead_states:
                    del gaussians.optimizer.state[tensor]

                if dead_states:
                    print(f"  Cleaned {len(dead_states)} dead optimizer states")

            # 多次强制垃圾回收
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()

            # 打印清理后的内存状态
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                free_memory, total_memory = torch.cuda.mem_get_info()
                print(
                    f"  After deep clean: {free_memory / 1e9:.1f}GB free, {(total_memory - free_memory) / 1e9:.1f}GB used")

        # 随机选择相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # ===== 创建缩放版本用于渲染 =====
        render_cam = create_scaled_camera_view(viewpoint_cam, render_resolution_scale)

        # 打印首次使用缩放相机的信息
        if iteration == 1:
            print(f"\n[Render] First iteration render info:")
            print(f"  Original resolution: {viewpoint_cam.image_width}x{viewpoint_cam.image_height}")
            print(
                f"  Render resolution: {render_cam.image_width}x{render_cam.image_height} ({render_resolution_scale}x)")
            print(f"  Resolution fixed at {render_resolution_scale}x for entire training")

        # 渲染（使用缩放相机）
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(render_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        # 获取缩放后的GT图像
        gt_image = render_cam.original_image.cuda()

        # 计算损失（在缩放分辨率上）
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        # 累积梯度统计（在密度控制区间内）
        if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
            # 注意：visibility_filter和radii是基于缩放分辨率的
            # 但梯度累积仍然有效，因为是基于可见的高斯点
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        iter_end.record()

        with torch.no_grad():
            # 更新损失统计
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # 进度条更新
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Gaussians": f"{gaussians._xyz.shape[0]}",
                    "Scene": scene_name,
                    "Render": f"{render_resolution_scale}x"
                })
                progress_bar.update(10)

            # === 密度控制（原始3DGS方式）===
            if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
                if iteration % 2000 == 0 and iteration > 0 and iteration < 10000:
                    # 简单的移动衰减，不需要重新投影
                    with torch.no_grad():
                        if not hasattr(gaussians, '_last_positions'):
                            gaussians._last_positions = gaussians._xyz.clone()
                        elif len(gaussians._last_positions) == len(gaussians._xyz):
                            movement = torch.norm(gaussians._xyz - gaussians._last_positions, dim=1)
                            moved_mask = movement > 0.05 * semantic_trainer.densifier.scene_extent

                            if moved_mask.sum() > 100:
                                print(f"[Semantic Decay] {moved_mask.sum()} gaussians moved significantly")
                                gaussians._semantic_importance[moved_mask] *= 0.95
                                gaussians._semantic_importance = torch.clamp(gaussians._semantic_importance, min=0.2)

                            gaussians._last_positions = gaussians._xyz.clone()
                        else:
                            # 点数变化了，重置
                            gaussians._last_positions = gaussians._xyz.clone()
                # 执行密度控制
                if iteration % opt.densification_interval == 0:
                    # 添加调试信息 - 密度控制前
                    print(f"\n[Debug] Before densification at iteration {iteration}:")
                    print(f"  Scene extent: {semantic_trainer.densifier.scene_extent:.3f}")

                    # 检查高斯位置范围
                    pos_min = gaussians._xyz.min(dim=0)[0]
                    pos_max = gaussians._xyz.max(dim=0)[0]
                    actual_extent = (pos_max - pos_min).max().item()
                    print(f"  Gaussian positions range: {actual_extent:.3f}")
                    print(f"  Position bounds - Min: [{pos_min[0]:.2f}, {pos_min[1]:.2f}, {pos_min[2]:.2f}], "
                          f"Max: [{pos_max[0]:.2f}, {pos_max[1]:.2f}, {pos_max[2]:.2f}]")

                    # 先管理探索性高斯的生命周期（如果有）
                    if hasattr(semantic_trainer, 'lifecycle_manager') and hasattr(gaussians, '_is_exploratory'):
                        lifecycle_stats = semantic_trainer.lifecycle_manager.manage_exploratory_lifecycle(
                            gaussians, iteration, opt.iterations
                        )
                        if iteration % 1000 == 0 and lifecycle_stats['promoted'] > 0:
                            print(
                                f"[Lifecycle] Promoted {lifecycle_stats['promoted']} exploratory gaussians to permanent")

                    # 设置当前迭代（用于剪枝保护）
                    semantic_trainer.densifier.current_iteration = iteration
                    semantic_trainer.densifier.max_iterations = opt.iterations

                    # 设置屏幕空间剪枝阈值（原始3DGS的逻辑）
                    if iteration > opt.opacity_reset_interval:  # 3000迭代后
                        size_threshold = 40  # 屏幕空间20像素
                    else:
                        size_threshold = None  # 前3000迭代不用

                    # 获取当前参数
                    gaussians_dict = gaussians.get_params_dict()

                    # 记录操作前的数量
                    num_before = len(gaussians_dict['positions'])

                    # 通过trainer执行密度控制
                    updated_dict = semantic_trainer.densify_and_prune(
                        gaussians_dict,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        iteration,
                        semantic_trainer.densifier.scene_extent,
                        max_screen_size=size_threshold
                    )

                    # 更新高斯模型
                    gaussians.update_from_params_dict(updated_dict)

                    # 添加调试信息 - 密度控制后
                    print(f"\n[Debug] After densification at iteration {iteration}:")
                    print(f"  Scene extent: {semantic_trainer.densifier.scene_extent:.3f}")

                    # 再次检查高斯位置范围
                    pos_min_after = gaussians._xyz.min(dim=0)[0]
                    pos_max_after = gaussians._xyz.max(dim=0)[0]
                    actual_extent_after = (pos_max_after - pos_min_after).max().item()
                    print(f"  Gaussian positions range: {actual_extent_after:.3f}")
                    print(
                        f"  Position bounds - Min: [{pos_min_after[0]:.2f}, {pos_min_after[1]:.2f}, {pos_min_after[2]:.2f}], "
                        f"Max: [{pos_max_after[0]:.2f}, {pos_max_after[1]:.2f}, {pos_max_after[2]:.2f}]")

                    # 检查场景范围是否被意外修改
                    if hasattr(semantic_trainer.densifier, '_original_scene_extent'):
                        if abs(semantic_trainer.densifier.scene_extent - semantic_trainer.densifier._original_scene_extent) > 0.01:
                            print(f"  ⚠️ WARNING: Scene extent changed! "
                                  f"Original: {semantic_trainer.densifier._original_scene_extent:.3f}, "
                                  f"Current: {semantic_trainer.densifier.scene_extent:.3f}")

                    # 输出统计
                    num_after = len(gaussians._xyz)
                    stats = semantic_trainer.densifier.get_statistics()
                    print(
                        f"\n[Densify {iteration}] {num_before} -> {num_after} gaussians (render scale: {render_resolution_scale}x)")
                    print(
                        f"  Operations: {stats['total_splits']} splits, {stats['total_clones']} clones, {stats['total_prunes']} prunes")
                    if stats['semantic_boosts'] > 0:
                        print(f"  Semantic boosts: {stats['semantic_boosts']}")

                # 每1000次迭代额外检查一次
                elif iteration % 1000 == 0:
                    print(f"\n[Debug Check at iteration {iteration}]")
                    print(f"  Scene extent: {semantic_trainer.densifier.scene_extent:.3f}")

                    pos_min = gaussians._xyz.min(dim=0)[0]
                    pos_max = gaussians._xyz.max(dim=0)[0]
                    actual_extent = (pos_max - pos_min).max().item()
                    print(f"  Gaussian positions range: {actual_extent:.3f}")
                    print(f"  Position bounds - Min: [{pos_min[0]:.2f}, {pos_min[1]:.2f}, {pos_min[2]:.2f}], "
                          f"Max: [{pos_max[0]:.2f}, {pos_max[1]:.2f}, {pos_max[2]:.2f}]")

                    # 如果位置范围远大于场景范围，发出警告
                    if actual_extent > semantic_trainer.densifier.scene_extent * 2:
                        print(f"  ⚠️ WARNING: Gaussian positions ({actual_extent:.3f}) "
                              f"far exceed scene extent ({semantic_trainer.densifier.scene_extent:.3f})!")

                # === 探索性分裂（使用全分辨率）===
                if iteration > 500 and iteration % 500 == 0:
                    current_num_gaussians = gaussians._xyz.shape[0]

                    # 当高斯数量超过10万时，完全禁用探索性分裂
                    if current_num_gaussians > 100000:
                        print(f"[ES] Disabled due to high gaussian count: {current_num_gaussians} > 100k")
                    elif torch.cuda.is_available():
                        free_memory, _ = torch.cuda.mem_get_info()
                        if free_memory > 3e9:  # 有足够内存
                            # 对于探索性分裂，我们使用原始分辨率以保证精度
                            print(f"\n[ES] Performing exploratory split at iteration {iteration}")

                            # 始终需要重新渲染全分辨率，因为我们全程用0.5x
                            print(
                                f"  Re-rendering at full resolution for hole detection (training uses {render_resolution_scale}x)")
                            full_render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                            full_image = full_render_pkg["render"]
                            full_gt_image = viewpoint_cam.original_image.cuda()

                            # 准备相机参数
                            camera_params = {}

                            # 处理K矩阵
                            if hasattr(viewpoint_cam, 'K'):
                                K = viewpoint_cam.K
                                camera_params['K'] = K if isinstance(K, np.ndarray) else K.cpu().numpy()
                            else:
                                camera_params['K'] = np.eye(3)

                            # 处理R矩阵
                            if hasattr(viewpoint_cam, 'R'):
                                R = viewpoint_cam.R
                                camera_params['R'] = R if isinstance(R, np.ndarray) else R.cpu().numpy()
                            else:
                                camera_params['R'] = np.eye(3)

                            # 处理t向量
                            if hasattr(viewpoint_cam, 'T'):
                                t = viewpoint_cam.T
                                camera_params['t'] = t if isinstance(t, np.ndarray) else t.cpu().numpy()
                            else:
                                camera_params['t'] = np.zeros(3)

                            image_name = viewpoint_cam.image_name if hasattr(viewpoint_cam, 'image_name') else None

                            # 执行探索性分裂
                            num_created = semantic_trainer.explorer.detect_and_create_exploratory_gaussians(
                                full_image.permute(1, 2, 0),
                                full_gt_image.permute(1, 2, 0),
                                gaussians,
                                camera_params,
                                iteration,
                                image_name
                            )

                            if num_created > 0:
                                print(f"[ES] Created {num_created} exploratory gaussians")
                        else:
                            print(f"[ES] Skipped due to low memory: {free_memory / 1e9:.1f}GB")

                # 重置不透明度
                if iteration % opt.opacity_reset_interval == 0 and iteration < opt.densify_until_iter:
                    print(f"\n[Reset] Resetting opacity at iteration {iteration} (during densification)")
                    gaussians.reset_opacity()
                    # 强制同步和清理
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

                    # 等待一下让GPU稳定
                    time.sleep(0.5)

                    # 再次检查内存
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    print(f"[Reset] After cleanup: {free_memory / 1e9:.1f}GB free")

            # 优化器步进
            if iteration < opt.iterations:
                # 梯度裁剪 - 防止梯度爆炸
                # 对位置参数进行梯度裁剪
                if gaussians._xyz.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._xyz], max_norm=1.0)

                # 对不透明度参数进行梯度裁剪（降低阈值）
                if gaussians._opacity.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._opacity], max_norm=1.0)  # 更严格

                # 对缩放参数进行梯度裁剪
                if gaussians._scaling.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._scaling], max_norm=1.0)

                # 对旋转参数进行梯度裁剪
                if gaussians._rotation.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._rotation], max_norm=1.0)

                # 对特征参数进行梯度裁剪
                if gaussians._features_dc.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._features_dc], max_norm=1.0)

                if gaussians._features_rest.grad is not None:
                    torch.nn.utils.clip_grad_norm_([gaussians._features_rest], max_norm=1.0)

                # 执行优化器步进
                gaussians.optimizer.step()

                # 清零梯度
                gaussians.optimizer.zero_grad(set_to_none=True)

                # 每100次迭代输出一次梯度信息用于调试
                if iteration % 100 == 0 and iteration > 0:
                    gaussians.compress_optimizer_state()
                    grad_norms = []
                    if gaussians._xyz.grad is not None:
                        grad_norms.append(('xyz', torch.norm(gaussians._xyz.grad).item()))
                    if gaussians._opacity.grad is not None:
                        grad_norms.append(('opacity', torch.norm(gaussians._opacity.grad).item()))
                    if gaussians._scaling.grad is not None:
                        grad_norms.append(('scaling', torch.norm(gaussians._scaling.grad).item()))

                    if grad_norms:
                        print(f"[Grad Norms {iteration}] " + ", ".join(
                            [f"{name}: {norm:.4f}" for name, norm in grad_norms]))

            # 保存检查点
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # 保存场景
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

                # 最终统计
                print(f"\nStatistics at save point (iteration {iteration}) for scene '{scene_name}':")
                print(f"  Total Gaussians: {gaussians._xyz.shape[0]}")
                print(f"  Render resolution: {render_resolution_scale}x (fixed)")
                final_stats = semantic_trainer.densifier.get_statistics()
                print(f"  Total operations: {final_stats['total_splits']} splits, "
                      f"{final_stats['total_clones']} clones, "
                      f"{final_stats['total_prunes']} prunes")
                print(f"  Current loss: {ema_loss_for_log:.4f}")
                print(f"  Memory warnings during training: {resolution_stats['memory_warnings']}")

    print("\n=== Training Complete ===")

    # 获取最终统计
    final_stats = semantic_trainer.get_training_stats(gaussians.get_params_dict(), iteration)
    print(f"\nFinal statistics for scene '{scene_name}':")
    print(f"  Total Gaussians: {final_stats['num_gaussians']}")
    print(f"  Exploratory Gaussians: {final_stats.get('num_exploratory', 0)}")

    # 打印最终内存信息
    if 'memory_info' in final_stats:
        mem = final_stats['memory_info']
        print(f"  Final GPU Memory: {mem['gpu_used_gb']:.1f}GB used ({mem['gpu_usage_percent']:.1f}%), "
              f"{mem['gpu_free_gb']:.1f}GB free")

    # 打印分辨率统计总结
    print(f"\nRender resolution summary:")
    print(f"  Fixed resolution throughout training: {render_resolution_scale}x")
    print(f"  Total memory warnings: {resolution_stats['memory_warnings']}")
    print(f"\nNote: Training used {render_resolution_scale}x resolution for memory efficiency.")
    print(f"      Test/evaluation should use full resolution for best quality.")


def prepare_output_and_logger(args):
    """准备输出目录和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer


if __name__ == "__main__":
    # 设置参数
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--scene_type", type=str, default=None,
                        choices=['truck', 'train', 'playground', 'drjohnson',
                                 'kitchen', 'room', 'bicycle', 'garden', 'counter', 'default'],
                        help="Manually specify scene type for semantic configuration")
    parser.add_argument("--use_preprocessed_masks", action="store_true", default=True,
                        help="Use preprocessed SAM masks if available")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # 安全随机种子
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 训练
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, scene_type=args.scene_type)

    print("\nTraining complete.")

