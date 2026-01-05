"""
语义增强的3D高斯训练器 - 支持预处理掩码版
整合SAM-SI, SAM-RPS, SAM-ES三大模块
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys
import gc

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from semantic_3dgs.core.semantic_initializer import OptimizedSemanticInitializer as SemanticSparseInitializer
from semantic_3dgs.core.adaptive_densification import SemanticAdaptiveDensification, DensificationConfig
from semantic_3dgs.core.exploratory_split import SemanticExplorarySplit, ExploratoryConfig
from segment_anything import sam_model_registry, SamPredictor
from semantic_3dgs.core.adaptive_configs import AdaptiveExploratoryConfig, RapidHoleFilling, ExploratoryGaussianOptimizer

class SemanticGaussianTrainer:
    """语义增强的3D高斯训练器 - 支持预处理掩码版"""

    def __init__(self, config):
        self.config = config

        print("Initializing Semantic Gaussian Trainer (Memory Optimized)...")

        # 检查是否使用预处理掩码
        use_preprocessed_masks = getattr(config, 'use_preprocessed_masks', True)
        preprocessed_masks_dir = getattr(config, 'preprocessed_masks_dir', None)

        if use_preprocessed_masks and preprocessed_masks_dir:
            print(f"✓ Using preprocessed masks from: {preprocessed_masks_dir}")
            # 不初始化SAM模型
            self.sam_model = None
            self.sam_predictor = None
        else:
            # 检查是否使用轻量级模型
            use_lightweight_sam = getattr(config, 'use_lightweight_sam', False)
            if use_lightweight_sam:
                print("Using lightweight SAM model for memory efficiency")

            # 初始化SAM模型（共享）- 可选择轻量级版本
            if use_lightweight_sam:
                try:
                    # 尝试使用vit_b模型
                    sam_checkpoint = config.sam_checkpoint.replace("vit_h", "vit_b").replace("4b8939", "0b3195")
                    self.sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                    print("✓ Loaded lightweight SAM model (vit_b)")
                except:
                    print("Lightweight SAM not found, falling back to vit_h")
                    self.sam_model = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint)
            else:
                self.sam_model = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint)

            self.sam_model.eval()
            if torch.cuda.is_available():
                self.sam_model = self.sam_model.cuda()
            self.sam_predictor = SamPredictor(self.sam_model)

        # 初始化三大模块
        print("Loading SAM-SI module...")
        self.initializer = SemanticSparseInitializer(
            sam_checkpoint=config.sam_checkpoint,
            clip_model_name=config.clip_model,
            use_lightweight_sam=getattr(config, 'use_lightweight_sam', False),
            use_preprocessed_masks=use_preprocessed_masks,
            preprocessed_masks_dir=preprocessed_masks_dir
        )

        print("Loading SAM-RPS module...")
        densification_config = DensificationConfig(
            grad_threshold=config.grad_threshold,
            percent_dense=config.percent_dense,
            opacity_cull=config.opacity_cull,
            semantic_weight=config.semantic_weight,
            #protected_labels=config.protected_labels,
            max_gaussians=config.max_gaussians
        )
        self.densifier = SemanticAdaptiveDensification(densification_config)
        # 延迟初始化CLIP以节省内存
        # self.densifier.initialize_clip()

        print("Loading SAM-ES module...")
        exploratory_config = ExploratoryConfig(
            geometric_threshold=config.geometric_threshold,
            semantic_iou_threshold=config.semantic_iou_threshold,
            max_exploratory_points=config.max_exploratory_points,
            debug_mode=config.debug_mode
        )
        self.explorer = SemanticExplorarySplit(
            exploratory_config,
            self.sam_predictor,
            use_preprocessed_masks=use_preprocessed_masks,
            preprocessed_masks_dir=preprocessed_masks_dir
        )
        # 新增：自适应配置和快速填补
        self.adaptive_es_config = AdaptiveExploratoryConfig()
        self.rapid_filler = RapidHoleFilling()
        self.lifecycle_manager = ExploratoryGaussianOptimizer()
        self._last_es_count = 0

        print("✓ Semantic Gaussian Trainer initialized successfully!")

        # 打印内存状态
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            print(f"Initial GPU memory: {(total_memory-free_memory)/1e9:.1f}GB used, {free_memory/1e9:.1f}GB free")

    def cleanup_sam(self):
        """释放SAM模型内存"""
        if self.sam_model is not None:
            print("Releasing SAM model from memory...")
            del self.sam_model
            self.sam_model = None
        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None
        gc.collect()
        torch.cuda.empty_cache()

    def initialize_gaussians_from_sparse(self, images, sparse_points, cameras, text_prompts=None, image_names=None):
        """
        使用SAM-SI从稀疏点云初始化高斯
        注意：初始化完成后会自动释放SAM和CLIP模型

        Args:
            images: 图像列表 (numpy arrays)
            sparse_points: 稀疏3D点 (Nx3 numpy array)
            cameras: 相机参数列表
            text_prompts: CLIP文本提示
            image_names: 图像名称列表（用于加载预处理掩码）

        Returns:
            初始化的高斯参数字典
        """
        print("\n=== SAM-SI: Semantic-Enhanced Initialization ===")

        # 使用SAM-SI初始化
        gaussian_params = self.initializer.initialize_gaussians(
            images=images,
            sfm_points=sparse_points,
            camera_params=cameras,
            text_prompts=text_prompts or ["face", "text", "detailed_texture", "window", "door"],
            image_names=image_names
        )

        # 注意：initializer.initialize_gaussians 内部会自动调用 cleanup_models()
        # 释放 SAM 和 CLIP 模型内存

        # 释放trainer的SAM相关资源
        self.cleanup_sam()

        gc.collect()
        torch.cuda.empty_cache()

        return gaussian_params

    def densify_and_prune(self, gaussians_dict, viewspace_gradients,
                          visibility_filter, radii, iteration, scene_extent,
                          max_screen_size=None):
        """使用SAM-RPS进行语义感知的密度控制"""
        # 计算密度控制统计
        stats = self.densifier.compute_densification_stats(
            gaussians_dict,
            viewspace_gradients,
            visibility_filter,
            radii
        )

        # 执行密度控制和剪枝
        updated_gaussians = self.densifier.densify_and_prune(
            gaussians_dict,
            stats,
            scene_extent,
            max_screen_size
        )

        # 获取统计信息用于调试
        densifier_stats = self.densifier.get_statistics()

        # 每500次迭代输出详细统计
        if iteration % 500 == 0:
            num_semantic = 0
            if 'semantic_importance' in updated_gaussians:
                num_semantic = (updated_gaussians['semantic_importance'] > 0.7).sum().item()

            print(f"\n[Densify Stats at {iteration}]")
            print(f"  Total gaussians: {len(updated_gaussians['positions'])}")
            print(f"  High semantic importance: {num_semantic}")
            print(
                f"  Semantic boost ratio: {densifier_stats['semantic_boosts'] / max(1, densifier_stats['total_splits'] + densifier_stats['total_clones']):.2%}")

        return updated_gaussians

    def detect_and_fill_holes(self, gaussians_dict, rendered_image, gt_image,
                              camera_params, iteration, max_iterations, image_name=None):
        """
        使用SAM-ES检测并填补空洞

        Args:
            gaussians_dict: 当前高斯参数字典
            rendered_image: 渲染图像 (HxWx3 tensor)
            gt_image: 真实图像 (HxWx3 tensor)
            camera_params: 相机参数
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数
            image_name: 图像名称（用于加载预处理掩码）

        Returns:
            包含新探索性高斯的参数字典（如果有）
        """
        # 在内存紧张时跳过探索性分裂
        if torch.cuda.is_available():
            free_memory, _ = torch.cuda.mem_get_info()
            if free_memory < 3e9:  # 少于3GB空闲内存
                return None

        # 设置迭代信息
        self.explorer.set_iteration_info(iteration, max_iterations)

        # 检测多模态空洞
        holes = self.explorer.detect_multimodal_holes(
            rendered_image,
            gt_image,
            camera_params=camera_params,
            image_name=image_name
        )

        # 生成探索性高斯
        new_gaussians = None
        if holes['count'] > 0:
            # 限制空洞数量
            max_holes = min(self.config.max_holes_per_iter, 5)  # 进一步限制
            new_gaussians = self.explorer.generate_exploratory_gaussians(
                holes['holes'][:max_holes],
                gaussians_dict
            )

            if new_gaussians is not None:
                print(f"[Iter {iteration}] Generated {len(new_gaussians['positions'])} exploratory gaussians")

        return new_gaussians

    def merge_gaussians(self, existing_dict, new_dict):
        """
        合并现有高斯和新高斯

        Args:
            existing_dict: 现有高斯参数字典
            new_dict: 新高斯参数字典

        Returns:
            合并后的高斯参数字典
        """
        if new_dict is None or len(new_dict['positions']) == 0:
            return existing_dict

        # 检查合并后是否会超过限制
        total_after_merge = len(existing_dict['positions']) + len(new_dict['positions'])
        if total_after_merge > self.config.max_gaussians:
            print(f"[Warning] Merging would exceed max gaussians ({total_after_merge} > {self.config.max_gaussians})")
            # 只添加部分新高斯
            max_new = self.config.max_gaussians - len(existing_dict['positions'])
            if max_new <= 0:
                return existing_dict

            # 截断新高斯
            for key in new_dict:
                if isinstance(new_dict[key], torch.Tensor):
                    new_dict[key] = new_dict[key][:max_new]

        merged = {}

        # 合并所有参数
        for key in existing_dict.keys():
            if key in new_dict:
                # 检查维度匹配
                existing_tensor = existing_dict[key]
                new_tensor = new_dict[key]

                # 特殊处理不同的参数类型
                if key == 'opacities':
                    # 确保 opacities 是 2D tensor [N, 1]
                    if existing_tensor.dim() == 1:
                        existing_tensor = existing_tensor.unsqueeze(-1)
                    if new_tensor.dim() == 1:
                        new_tensor = new_tensor.unsqueeze(-1)

                elif key == 'semantic_importance' or key == 'semantic_labels':
                    # 确保这些是 1D tensor [N]
                    if existing_tensor.dim() == 2 and existing_tensor.shape[1] == 1:
                        existing_tensor = existing_tensor.squeeze(-1)
                    if new_tensor.dim() == 2 and new_tensor.shape[1] == 1:
                        new_tensor = new_tensor.squeeze(-1)

                elif key == 'is_exploratory':
                    # 确保是 1D boolean tensor
                    if existing_tensor.dim() == 2:
                        existing_tensor = existing_tensor.squeeze(-1)
                    if new_tensor.dim() == 2:
                        new_tensor = new_tensor.squeeze(-1)

                # 合并
                try:
                    merged[key] = torch.cat([existing_tensor, new_tensor], dim=0)
                except RuntimeError as e:
                    print(f"[Warning] Failed to merge {key}: {e}")
                    print(f"  Existing shape: {existing_tensor.shape}, New shape: {new_tensor.shape}")
                    # 跳过这个键
                    merged[key] = existing_tensor

            else:
                # 为新高斯创建默认值
                num_new = len(new_dict['positions'])
                if key == 'semantic_importance':
                    # 对于语义重要性，使用较高的默认值（因为是探索性高斯）
                    default = torch.ones(num_new, device=existing_dict[key].device) * 0.8
                elif key == 'semantic_labels':
                    default = torch.zeros(num_new, dtype=torch.long, device=existing_dict[key].device)
                elif key == 'is_exploratory':
                    default = torch.ones(num_new, dtype=torch.bool, device=existing_dict[key].device)
                else:
                    # 跳过未知键
                    merged[key] = existing_dict[key]
                    continue

                # 确保维度匹配
                if key == 'opacities' and existing_dict[key].dim() == 2:
                    if default.dim() == 1:
                        default = default.unsqueeze(-1)
                elif key != 'opacities' and existing_dict[key].dim() == 1:
                    if default.dim() == 2:
                        default = default.squeeze(-1)

                merged[key] = torch.cat([existing_dict[key], default], dim=0)

        # 添加新字典中独有的键
        for key in new_dict.keys():
            if key not in merged:
                if key == 'metadata':
                    # 跳过元数据，因为它不是tensor
                    continue
                # 为旧高斯创建默认值
                num_old = len(existing_dict['positions'])
                if key == 'is_exploratory':
                    default = torch.zeros(num_old, dtype=torch.bool, device=new_dict[key].device)
                elif key == 'user_triggered':
                    default = torch.zeros(num_old, dtype=torch.bool, device=new_dict[key].device)
                else:
                    continue

                # 确保维度匹配
                new_tensor = new_dict[key]
                if new_tensor.dim() == 2 and new_tensor.shape[1] == 1:
                    new_tensor = new_tensor.squeeze(-1)

                merged[key] = torch.cat([default, new_tensor], dim=0)

        # 打印合并信息
        print(
            f"[Merge] Successfully merged gaussians: {len(existing_dict['positions'])} + {len(new_dict['positions'])} = {len(merged['positions'])}")

        return merged

    def get_training_stats(self, gaussians_dict, iteration):
        """获取训练统计信息"""
        num_gaussians = len(gaussians_dict['positions'])
        num_exploratory = gaussians_dict.get('is_exploratory', torch.zeros(num_gaussians)).sum().item()
        avg_opacity = torch.sigmoid(gaussians_dict['opacities']).mean().item()
        avg_scale = torch.exp(gaussians_dict['scales']).mean().item()

        # 内存信息
        memory_info = {}
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            memory_info = {
                'gpu_used_gb': (total_memory - free_memory) / 1e9,
                'gpu_free_gb': free_memory / 1e9,
                'gpu_usage_percent': ((total_memory - free_memory) / total_memory) * 100
            }

        stats = {
            'num_gaussians': num_gaussians,
            'num_exploratory': num_exploratory,
            'avg_opacity': avg_opacity,
            'avg_scale': avg_scale,
            'densifier_stats': self.densifier.get_statistics(),
            'explorer_stats': self.explorer.get_statistics() if hasattr(self, 'explorer') else {},
            'memory_info': memory_info
        }

        return stats