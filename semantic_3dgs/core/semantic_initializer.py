#!/usr/bin/env python3
"""
优化版的语义初始化模块 - 支持预处理掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import clip
from segment_anything import sam_model_registry, SamPredictor
from concurrent.futures import ThreadPoolExecutor
import time
import gc
import os
import pickle


class OptimizedSemanticInitializer:
    """优化的语义增强稀疏高斯初始化器"""

    def __init__(self, sam_checkpoint: str = "checkpoints/sam/sam_vit_h_4b8939.pth",
                 clip_model_name: str = "ViT-B/32", use_lightweight_sam: bool = True,
                 use_preprocessed_masks: bool = True,
                 preprocessed_masks_dir: str = None,
                 sh_degree: int = 3):
        """初始化语义模型"""
        self.sh_degree = sh_degree  # 保存SH阶数
        self.use_lightweight_sam = use_lightweight_sam
        self.use_preprocessed_masks = use_preprocessed_masks
        self.preprocessed_masks_dir = preprocessed_masks_dir

        # 如果使用预处理掩码，就不需要初始化SAM
        if not use_preprocessed_masks:
            # 初始化SAM - 可选择轻量级版本
            if use_lightweight_sam and "vit_h" in sam_checkpoint:
                # 尝试使用更小的模型
                try:
                    sam_checkpoint = sam_checkpoint.replace("vit_h", "vit_b").replace("4b8939", "0b3195")
                    print(f"Using lightweight SAM model: vit_b")
                    self.sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                except:
                    print(f"Lightweight SAM not found, using original model")
                    self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            else:
                self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)

            self.sam_model.eval()
            if torch.cuda.is_available():
                self.sam_model = self.sam_model.cuda()
            self.sam_predictor = SamPredictor(self.sam_model)
            print(f"✓ Initialized SAM model (optimized mode)")
        else:
            print(f"✓ Using preprocessed masks from: {preprocessed_masks_dir}")
            self.sam_model = None
            self.sam_predictor = None

        # 初始化CLIP
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device='cuda')
        self.clip_model.eval()
        print(f"✓ Initialized CLIP model: {clip_model_name}")

    def cleanup_models(self):
        """释放SAM和CLIP模型内存"""
        print("\nReleasing semantic models from memory...")

        # 释放SAM
        if hasattr(self, 'sam_model') and self.sam_model is not None:
            del self.sam_model
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
            del self.sam_predictor

        # 释放CLIP
        if hasattr(self, 'clip_model'):
            del self.clip_model
        if hasattr(self, 'clip_preprocess'):
            del self.clip_preprocess

        # 强制垃圾回收和清空CUDA缓存
        gc.collect()
        torch.cuda.empty_cache()

        # 打印内存状态
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            print(f"✓ Models released. Free GPU memory: {free_memory / 1e9:.1f}GB / {total_memory / 1e9:.1f}GB")

    def multi_view_sam_segmentation_fast(self, images: List[np.ndarray], image_names: List[str] = None) -> List[
        np.ndarray]:
        """
        优化的多视图SAM分割 - 支持预处理掩码
        """
        if self.use_preprocessed_masks and self.preprocessed_masks_dir:
            masks = []

            for img_idx, (img, img_name) in enumerate(zip(images, image_names or range(len(images)))):
                # 构建掩码文件路径
                if isinstance(img_name, str):
                    base_name = os.path.splitext(img_name)[0]
                else:
                    base_name = f"image_{img_idx:04d}"

                label_map_path = os.path.join(self.preprocessed_masks_dir, f"{base_name}_labels.npy")

                if os.path.exists(label_map_path):
                    # 加载预处理的label map
                    label_map = np.load(label_map_path)
                    masks.append(label_map)
                    print(f"  Loaded preprocessed mask for {base_name}: {np.unique(label_map).size - 1} masks")
                else:
                    print(f"  Warning: No preprocessed mask found for {base_name}, using empty mask")
                    masks.append(np.zeros((img.shape[0], img.shape[1]), dtype=np.int32))

            return masks
        else:
            # 原有的SAM处理代码
            return self._original_multi_view_sam_segmentation_fast(images)

    def _original_multi_view_sam_segmentation_fast(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        原始的多视图SAM分割实现
        """
        masks = []

        for img_idx, img in enumerate(images):
            start_time = time.time()

            # 设置SAM图像
            self.sam_predictor.set_image(img)

            # 使用稀疏网格采样点而不是自动生成所有mask
            height, width = img.shape[:2]

            # 自适应网格大小
            grid_size = 64  # 减少采样点
            step_y = height // grid_size
            step_x = width // grid_size

            # 生成网格点
            grid_points = []
            for y in range(0, height, step_y):
                for x in range(0, width, step_x):
                    grid_points.append([x, y])

            grid_points = np.array(grid_points)

            # 合并mask
            combined_mask = np.zeros((height, width), dtype=np.int32)
            mask_id = 1

            # 批处理预测 - 修复维度问题
            batch_size = 64
            for i in range(0, len(grid_points), batch_size):
                batch_points = grid_points[i:i + batch_size]
                batch_labels = np.ones(len(batch_points))

                # 修复：predict_torch 需要添加批处理维度
                # 将点坐标从 [N, 2] 转换为 [1, N, 2]
                point_coords_batch = torch.from_numpy(batch_points).cuda().float().unsqueeze(0)
                point_labels_batch = torch.from_numpy(batch_labels).cuda().float().unsqueeze(0)

                # 预测
                masks_batch, scores_batch, _ = self.sam_predictor.predict_torch(
                    point_coords=point_coords_batch,
                    point_labels=point_labels_batch,
                    multimask_output=False
                )

                # 处理输出 - masks_batch 形状为 [1, 1, H, W]
                mask_np = masks_batch[0, 0].cpu().numpy()  # 取第一个批次的第一个mask
                score = scores_batch[0, 0].item()  # 取第一个批次的第一个分数

                if score > 0.8:  # 只保留高质量mask
                    # 避免重叠
                    new_region = (combined_mask == 0) & mask_np
                    if new_region.sum() > 100:  # 忽略太小的区域
                        combined_mask[new_region] = mask_id
                        mask_id += 1

            masks.append(combined_mask)
            print(f"  Image {img_idx + 1}: {mask_id - 1} masks in {time.time() - start_time:.2f}s")

            # 定期清理CUDA缓存
            if img_idx % 5 == 0:
                torch.cuda.empty_cache()

        return masks

    def compute_clip_features_batch(self, image_regions: List[torch.Tensor],
                                    text_prompts: List[str]) -> np.ndarray:
        """
        批量计算CLIP特征相似度 - 修复版本
        处理不同大小的图像区域
        """
        if not image_regions:
            return np.array([])

        # 批量预处理图像
        batch_size = 32
        all_similarities = []

        # 预计算文本特征
        text_tokens = clip.tokenize(text_prompts).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i in range(0, len(image_regions), batch_size):
            batch_regions = image_regions[i:i + batch_size]

            # 处理每个区域，确保大小一致
            processed_regions = []
            for region in batch_regions:
                # 确保是4D张量 [1, C, H, W]
                if region.dim() == 3:
                    region = region.unsqueeze(0)
                elif region.dim() == 2:
                    # 如果是灰度图，添加通道维度
                    region = region.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

                # 确保在正确的设备上
                region = region.cuda().float()

                # 归一化到 [0, 1]
                if region.max() > 1.0:
                    region = region / 255.0

                # 调整大小到 224x224
                region_resized = F.interpolate(region, size=(224, 224), mode='bilinear', align_corners=False)
                processed_regions.append(region_resized)

            # 现在所有区域都是 [1, 3, 224, 224]，可以安全拼接
            batch_tensor = torch.cat(processed_regions, dim=0)  # [B, 3, 224, 224]

            # 获取图像特征
            with torch.no_grad():
                image_features = self.clip_model.encode_image(batch_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                similarities = (image_features @ text_features.T).cpu().numpy()
                all_similarities.append(similarities)

            # 清理中间结果
            del batch_tensor, processed_regions
            torch.cuda.empty_cache()

        return np.vstack(all_similarities) if all_similarities else np.array([])

    def generate_semantic_heatmap_fast(self,
                                       semantic_masks: List[np.ndarray],
                                       images: List[np.ndarray],
                                       text_prompts: List[str],
                                       image_names: List[str] = None) -> np.ndarray:
        """
        快速生成语义热力图 - 修复版本
        """
        height, width = images[0].shape[:2]

        # 使用分层的热力图构建策略
        # Layer 1: 语义掩码存在性（最低层）
        mask_layer = np.zeros((height, width), dtype=np.float32)

        # Layer 2: CLIP语义相似度（中间层）
        clip_layer = np.zeros((height, width), dtype=np.float32)

        # Layer 3: 边缘和梯度信息（细节层）
        detail_layer = np.zeros((height, width), dtype=np.float32)

        # === Layer 1: 基础掩码层 ===
        # 降低基础权重，只标记有掩码的区域
        for mask in semantic_masks:
            if mask.max() > 0:
                # 将所有非背景区域标记为0.1
                mask_layer[mask > 0] = 0.1

        print(f"  Mask layer: coverage={np.sum(mask_layer > 0) / (height * width) * 100:.1f}%")

        # === Layer 2: CLIP语义层 ===
        if self.use_preprocessed_masks and self.preprocessed_masks_dir and image_names:
            # 收集所有需要处理的区域
            all_regions = []
            region_info = []

            for img_idx, (img, mask, img_name) in enumerate(zip(images, semantic_masks, image_names)):
                # 尝试加载详细的掩码信息
                base_name = os.path.splitext(img_name)[0] if isinstance(img_name, str) else f"image_{img_idx:04d}"
                masks_data_path = os.path.join(self.preprocessed_masks_dir, f"{base_name}_masks.pkl")

                if os.path.exists(masks_data_path):
                    with open(masks_data_path, 'rb') as f:
                        masks_data = pickle.load(f)

                    # 只选择较大的、高置信度的掩码
                    for i, (mask_info, segmentation) in enumerate(
                            zip(masks_data['masks'], masks_data['segmentations'])):
                        if mask_info['area'] > 500 and mask_info['score'] > 0.8:  # 更严格的筛选
                            bbox = mask_info['bbox']
                            x_min, y_min, w, h = bbox
                            x_max, y_max = x_min + w, y_min + h

                            if w > 20 and h > 20:  # 只处理较大的区域
                                region_img = img[y_min:y_max, x_min:x_max]
                                if region_img.size > 0:
                                    region_tensor = torch.from_numpy(region_img.copy()).permute(2, 0, 1)
                                    all_regions.append(region_tensor)
                                    region_info.append((img_idx, segmentation, bbox, mask_info['score']))

        # 限制区域数量
        if len(all_regions) > 200:  # 减少到200个
            # 根据区域大小和置信度排序
            region_scores = [(info[3] * np.sum(info[1]), idx) for idx, info in enumerate(region_info)]
            region_scores.sort(reverse=True)
            selected_indices = [idx for _, idx in region_scores[:200]]
            all_regions = [all_regions[i] for i in selected_indices]
            region_info = [region_info[i] for i in selected_indices]

        # 批量计算CLIP相似度
        if all_regions and len(text_prompts) > 0:
            try:
                similarities = self.compute_clip_features_batch(all_regions, text_prompts)

                # 分析相似度分布
                all_sims = similarities.flatten()
                sim_mean = np.mean(all_sims)
                sim_std = np.std(all_sims)
                print(f"  CLIP similarities: mean={sim_mean:.3f}, std={sim_std:.3f}, "
                      f"min={np.min(all_sims):.3f}, max={np.max(all_sims):.3f}")

                # 更新CLIP层
                for (img_idx, region_mask, bbox, score), sim_scores in zip(region_info, similarities):
                    # 使用更严格的映射
                    max_sim = float(sim_scores.max())

                    # 只有真正相关的区域才获得高分
                    if max_sim > sim_mean + sim_std:  # 高于平均值一个标准差
                        # 映射到[0.3, 0.8]
                        importance = 0.3 + 0.5 * (max_sim - sim_mean) / (3 * sim_std)
                        importance = np.clip(importance, 0.3, 0.8)
                    elif max_sim > sim_mean:  # 高于平均值
                        # 映射到[0.1, 0.3]
                        importance = 0.1 + 0.2 * (max_sim - (sim_mean - sim_std)) / (2 * sim_std)
                        importance = np.clip(importance, 0.1, 0.3)
                    else:  # 低于平均值
                        importance = 0.0  # 不增加重要性

                    if importance > 0:
                        clip_layer[region_mask] = np.maximum(clip_layer[region_mask], importance)

                print(f"  CLIP layer: active pixels={np.sum(clip_layer > 0)}, "
                      f"max={clip_layer.max():.3f}, mean={clip_layer[clip_layer > 0].mean():.3f}")

            except Exception as e:
                print(f"  Warning: CLIP computation failed: {e}")

        # === Layer 3: 细节层（边缘和梯度）===
        # 合并多个视图的边缘信息
        for i, img in enumerate(images[:3]):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 边缘检测
            edges = cv2.Canny(gray, 100, 200) / 255.0  # 提高阈值，只检测强边缘

            # 梯度幅度
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # 只保留强梯度
            grad_percentile = np.percentile(grad_mag, 90)  # 前10%
            strong_grad = (grad_mag > grad_percentile).astype(np.float32)

            # 更新细节层
            detail_layer = np.maximum(detail_layer, edges * 0.2)
            detail_layer = np.maximum(detail_layer, strong_grad * 0.15)

        # === 组合所有层 ===
        # 使用加权组合，而不是maximum
        heatmap = (
                mask_layer * 0.2 +  # 基础层权重最低
                clip_layer * 0.6 +  # CLIP层权重最高
                detail_layer * 0.2  # 细节层补充
        )

        # 应用空间平滑
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 1.5)

        # === 最终归一化 ===
        if heatmap.max() > 0:
            # 移除背景偏置
            background_value = np.percentile(heatmap, 10)  # 底部10%作为背景
            heatmap = heatmap - background_value
            heatmap = np.clip(heatmap, 0, None)

            # 使用自适应归一化
            if heatmap.max() > 0:
                # 找到有意义的最大值（去除异常值）
                p99 = np.percentile(heatmap[heatmap > 0], 99)
                if p99 > 0:
                    heatmap = heatmap / p99
                    heatmap = np.clip(heatmap, 0, 1)

                    # 应用gamma校正增强对比度
                    gamma = 1.5  # > 1 会降低中间值，增强对比度
                    heatmap = np.power(heatmap, gamma)

        # 确保有一些变化
        if heatmap.std() < 0.01:  # 如果太平坦
            print("  Warning: Heatmap too flat, adding noise")
            noise = np.random.randn(height, width) * 0.05
            heatmap = heatmap + noise
            heatmap = np.clip(heatmap, 0, 1)

        print(f"  Heatmap final: min={heatmap.min():.3f}, max={heatmap.max():.3f}, "
              f"mean={heatmap.mean():.3f}, std={heatmap.std():.3f}")
        print(f"  Pixels > 0.3: {(heatmap > 0.3).sum()} ({(heatmap > 0.3).sum() / (height * width) * 100:.1f}%)")
        print(f"  Pixels > 0.5: {(heatmap > 0.5).sum()} ({(heatmap > 0.5).sum() / (height * width) * 100:.1f}%)")
        print(f"  Pixels > 0.7: {(heatmap > 0.7).sum()} ({(heatmap > 0.7).sum() / (height * width) * 100:.1f}%)")

        return heatmap

    def adaptive_point_generation_fast(self,
                                       sfm_points: np.ndarray,
                                       semantic_masks: List[np.ndarray],
                                       heatmap: np.ndarray,
                                       coverage_threshold: float = 0.1,
                                       images: List[np.ndarray] = None) -> np.ndarray:
        """
        基于修复后热力图的自适应点生成
        """
        virtual_points = []
        height, width = semantic_masks[0].shape

        # 目标点数
        target_min_points = 2000
        target_max_points = 3000

        print(f"Target virtual points: {target_min_points}-{target_max_points}")

        # 创建现有点的密度图
        density_map = np.zeros((height, width))
        if hasattr(self, 'camera_params') and self.camera_params and len(sfm_points) > 0:
            projected = self.project_points_to_view(sfm_points, self.camera_params[0])
            for p in projected:
                if 0 <= p[0] < width and 0 <= p[1] < height:
                    x, y = int(p[0]), int(p[1])
                    cv2.circle(density_map, (x, y), 15, 1.0, -1)

        # 策略1：高重要性低密度区域（优先级最高）
        print("Strategy 1: High importance, low density regions...")
        high_importance_points = []

        # 使用多级阈值
        importance_levels = [0.7, 0.5, 0.3]
        points_per_level = [1000, 800, 500]

        for level_idx, (threshold, max_points) in enumerate(zip(importance_levels, points_per_level)):
            if len(virtual_points) >= target_min_points:
                break

            # 找到该重要性级别且低密度的区域
            importance_mask = (heatmap > threshold) & (density_map < 0.3)

            if importance_mask.sum() > 0:
                # 在这些区域采样
                y_coords, x_coords = np.where(importance_mask)

                # 根据热力图值加权采样
                weights = heatmap[importance_mask]
                weights = weights / weights.sum()

                num_samples = min(max_points, len(y_coords), target_min_points - len(virtual_points))

                if num_samples > 0:
                    indices = np.random.choice(len(y_coords), num_samples, replace=False, p=weights)

                    for idx in indices:
                        px, py = x_coords[idx], y_coords[idx]

                        # 添加小的随机偏移
                        px += np.random.uniform(-2, 2)
                        py += np.random.uniform(-2, 2)
                        px = np.clip(px, 0, width - 1)
                        py = np.clip(py, 0, height - 1)

                        # 估计深度
                        if len(sfm_points) > 0:
                            z = sfm_points[:, 2].mean() + np.random.uniform(-1.5, 1.5)
                        else:
                            z = 5.0 + np.random.uniform(-1.5, 1.5)

                        point_3d = self._simple_unproject(np.array([px, py]), z)
                        high_importance_points.append(point_3d)

        virtual_points.extend(high_importance_points)
        print(f"  Generated {len(high_importance_points)} high-importance points")

        # 策略2：语义边界点
        if len(virtual_points) < target_min_points:
            print("Strategy 2: Semantic boundary points...")
            boundary_points = []

            for mask in semantic_masks[:3]:
                if mask.max() > 0:
                    # 找边界
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
                    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
                    boundary = dilated - eroded

                    # 只在重要边界上采样
                    boundary_importance = boundary * heatmap
                    important_boundary = boundary_importance > 0.2

                    if important_boundary.sum() > 0:
                        y_coords, x_coords = np.where(important_boundary)

                        num_samples = min(200, len(y_coords))
                        indices = np.random.choice(len(y_coords), num_samples, replace=False)

                        for idx in indices:
                            px, py = x_coords[idx], y_coords[idx]

                            if len(sfm_points) > 0:
                                z = sfm_points[:, 2].mean() + np.random.uniform(-1, 1)
                            else:
                                z = 5.0 + np.random.uniform(-1, 1)

                            point_3d = self._simple_unproject(np.array([px, py]), z)
                            boundary_points.append(point_3d)

            virtual_points.extend(boundary_points[:target_min_points - len(virtual_points)])
            print(f"  Generated {len(boundary_points)} boundary points")

        # 策略3：填充稀疏区域
        if len(virtual_points) < target_min_points:
            print("Strategy 3: Filling sparse regions...")
            remaining = target_min_points - len(virtual_points)

            # 在中等重要性区域均匀采样
            medium_importance = (heatmap > 0.2) & (heatmap <= 0.5) & (density_map < 0.5)

            if medium_importance.sum() > 0:
                y_coords, x_coords = np.where(medium_importance)

                num_samples = min(remaining, len(y_coords))
                indices = np.random.choice(len(y_coords), num_samples, replace=False)

                for idx in indices:
                    px, py = x_coords[idx], y_coords[idx]

                    if len(sfm_points) > 0:
                        z = sfm_points[:, 2].mean() + np.random.uniform(-2, 2)
                    else:
                        z = 5.0 + np.random.uniform(-2, 2)

                    point_3d = self._simple_unproject(np.array([px, py]), z)
                    virtual_points.append(point_3d)

        # 限制最大数量
        if len(virtual_points) > target_max_points:
            # 根据重要性排序并选择
            all_points = np.array(virtual_points)
            # 重新投影并获取重要性分数
            projected = self.project_points_to_view(all_points, self.camera_params[0])
            importance_scores = []

            for p in projected:
                if 0 <= p[0] < width and 0 <= p[1] < height:
                    importance_scores.append(heatmap[int(p[1]), int(p[0])])
                else:
                    importance_scores.append(0)

            # 选择最重要的点
            importance_scores = np.array(importance_scores)
            top_indices = np.argsort(importance_scores)[-target_max_points:]
            virtual_points = [virtual_points[i] for i in top_indices]

        print(f"Total virtual points generated: {len(virtual_points)}")

        return np.array(virtual_points) if virtual_points else np.empty((0, 3))

    def _simple_unproject(self, point_2d: np.ndarray, depth: float) -> np.ndarray:
        """简化的反投影"""
        # 假设的相机内参
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0

        x = (point_2d[0] - cx) * depth / fx
        y = (point_2d[1] - cy) * depth / fy

        return np.array([x, y, depth])

    def compute_semantic_importance_batch(self,
                                          points: np.ndarray,
                                          heatmap: np.ndarray,
                                          camera_params: List[Dict]) -> np.ndarray:
        """
        批量计算语义重要性 - 避免逐点循环
        """
        num_points = len(points)
        importance_scores = np.zeros(num_points)

        # 只使用第一个视图进行快速估计
        if camera_params:
            # 批量投影所有点
            projected = self.project_points_to_view(points, camera_params[0])

            # 批量查询热力图值
            for i, p in enumerate(projected):
                if 0 <= p[0] < heatmap.shape[1] and 0 <= p[1] < heatmap.shape[0]:
                    x, y = int(p[0]), int(p[1])
                    importance_scores[i] = heatmap[y, x]
                else:
                    importance_scores[i] = 0.5  # 默认值
        else:
            importance_scores[:] = 0.5

        return importance_scores

    def initialize_gaussians(self,
                             images: List[np.ndarray],
                             sfm_points: np.ndarray,
                             camera_params: List[Dict],
                             text_prompts: Optional[List[str]] = None,
                             image_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        优化的语义增强高斯初始化
        """
        self.camera_params = camera_params

        print("Starting optimized semantic initialization...")
        print(f"  Number of images: {len(images)}")
        print(f"  Number of SfM points: {len(sfm_points)}")
        print(f"  Camera params available: {camera_params is not None}")
        total_start = time.time()

        try:
            # 1. 快速语义分割
            print("Step 1: Fast semantic segmentation...")
            seg_start = time.time()
            semantic_masks = self.multi_view_sam_segmentation_fast(images, image_names)
            print(f"  Completed in {time.time() - seg_start:.2f}s")

            # 2. 快速热力图生成
            print("Step 2: Fast heatmap generation...")
            heat_start = time.time()
            heatmap = self.generate_semantic_heatmap_fast(
                semantic_masks,
                images,
                text_prompts or ["object", "detail", "texture"],
                image_names
            )
            print(f"  Completed in {time.time() - heat_start:.2f}s")
            # 添加热力图统计
            print(f"  Heatmap stats: min={heatmap.min():.3f}, max={heatmap.max():.3f}, mean={heatmap.mean():.3f}")
            print(f"  Pixels > 0.3: {(heatmap > 0.3).sum()}")
            print(f"  Pixels > 0.5: {(heatmap > 0.5).sum()}")

            # 保存热力图用于调试
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap='hot')
            plt.colorbar()
            plt.title('Semantic Heatmap')
            plt.savefig('debug_heatmap.png')
            plt.close()
            print("  Saved heatmap to debug_heatmap.png")

            # 3. 快速点生成
            print("Step 3: Fast point generation...")
            point_start = time.time()
            virtual_points = self.adaptive_point_generation_fast(
                sfm_points,
                semantic_masks,
                heatmap,
                coverage_threshold=0.1
            )
            print(f"  Completed in {time.time() - point_start:.2f}s")

            # 合并点
            all_points = np.vstack([sfm_points, virtual_points]) if len(virtual_points) > 0 else sfm_points
            print(f"Total points: {len(all_points)} (SfM: {len(sfm_points)}, Virtual: {len(virtual_points)})")

            # 4. 批量计算语义重要性
            print("Step 4: Initializing parameters...")
            param_start = time.time()
            semantic_importance = self.compute_semantic_importance_batch(
                all_points, heatmap, camera_params
            )

            # 5. 快速初始化其他参数
            gaussian_params = {
                'positions': torch.from_numpy(all_points).float().cuda(),
                'colors': self._initialize_colors_fast(all_points, images[0], camera_params[0]),
                'opacities': self._initialize_opacities_fast(semantic_importance),
                'scales': self._initialize_scales_fast(all_points),
                'rotations': self._initialize_rotations_fast(len(all_points)),
                'semantic_importance': torch.from_numpy(semantic_importance).float().cuda(),
                'semantic_labels': torch.zeros(len(all_points), dtype=torch.long).cuda()
            }
            print(f"  Completed in {time.time() - param_start:.2f}s")

            print(f"\nTotal initialization time: {time.time() - total_start:.2f}s")

        finally:
            # 无论是否成功，都释放模型内存
            self.cleanup_models()

        return gaussian_params

    def _initialize_colors_fast(self, points: np.ndarray, image: np.ndarray, camera_params: Dict) -> torch.Tensor:
        """快速颜色初始化 - 只使用一个视图"""
        colors = []
        projected = self.project_points_to_view(points, camera_params)

        h, w = image.shape[:2]
        for p in projected:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                x, y = int(p[0]), int(p[1])
                color = image[y, x] / 255.0
            else:
                color = np.array([0.5, 0.5, 0.5])
            colors.append(color)

        return torch.from_numpy(np.array(colors)).float().cuda()

    # 修改semantic_initializer.py中的_initialize_opacities_fast
    def _initialize_opacities_fast(self, importance: np.ndarray) -> torch.Tensor:
        """使用原始3DGS的初始化策略"""
        # 原始3DGS统一初始化为0.1
        # 不管语义重要性如何，都从0.1开始
        opacities = torch.ones(len(importance), device="cuda") * 0.1

        # 转换到logit空间（这是inverse_sigmoid的作用）
        # logit(x) = log(x/(1-x))
        logit_opacities = torch.logit(opacities.clamp(0.01, 0.99))

        return logit_opacities

    def _initialize_scales_fast(self, points: np.ndarray) -> torch.Tensor:
        """快速尺度初始化 - 使用KD树的近似版本"""
        # 简化：使用固定尺度或基于场景范围
        if len(points) > 1:
            scene_scale = np.std(points, axis=0).mean() * 0.01
        else:
            scene_scale = 0.01

        scales = np.ones((len(points), 3)) * scene_scale
        log_scales = np.log(scales + 1e-6)
        return torch.from_numpy(log_scales).float().cuda()

    def _initialize_rotations_fast(self, num_points: int) -> torch.Tensor:
        """快速旋转初始化"""
        rots = np.zeros((num_points, 4))
        rots[:, 0] = 1.0  # 单位四元数
        return torch.from_numpy(rots).float().cuda()

    def project_points_to_view(self, points_3d: np.ndarray, camera_params: Dict) -> np.ndarray:
        """批量投影3D点到2D"""
        K = camera_params['K']
        R = camera_params['R']
        t = camera_params['t']

        # 批量计算
        points_cam = points_3d @ R.T + t.T
        points_2d_homo = points_cam @ K.T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

        return points_2d

    def initialize_from_full_pointcloud_with_semantic(self,
                                                      images: List[np.ndarray],
                                                      point_cloud,
                                                      camera_params: List[Dict],
                                                      text_prompts: Optional[List[str]] = None,
                                                      image_names: Optional[List[str]] = None,
                                                      important_labels: Optional[List[str]] = None) -> Dict:
        """
        使用完整点云初始化，但添加语义重要性标记
        保持原始3DGS的初始化策略，只是额外添加语义信息
        """
        print("\n=== 语义增强初始化（使用完整点云）===")
        print(f"  点云点数: {len(point_cloud.points)}")
        print(f"  策略: 保持3DGS原始初始化 + 添加语义重要性")

        # 新增：打印重要标签
        if important_labels:
            print(f"  重要标签集合: {important_labels}")

        # 步骤1：使用所有点云点
        all_points = point_cloud.points
        all_colors = point_cloud.colors
        num_points = len(all_points)

        # 步骤2：计算语义重要性
        print("\n计算语义重要性...")
        semantic_importance = np.ones(num_points) * 0.3  # 默认值改为0.3

        # 使用精确语义投影
        if self.use_preprocessed_masks and image_names:
            print("  使用精确语义投影...")

            # 准备数据
            label_maps = {}
            masks_data_all = {}

            # 加载所有需要的掩码数据
            for img_idx in range(min(10, len(images))):  # 限制使用的视图数
                if img_idx >= len(image_names):
                    break

                img_name = image_names[img_idx]
                base_name = os.path.splitext(img_name)[0] if isinstance(img_name, str) else f"image_{img_idx:04d}"

                label_path = os.path.join(self.preprocessed_masks_dir, f"{base_name}_labels.npy")
                masks_path = os.path.join(self.preprocessed_masks_dir, f"{base_name}_masks.pkl")

                if os.path.exists(label_path) and os.path.exists(masks_path):
                    try:
                        label_map = np.load(label_path)
                        with open(masks_path, 'rb') as f:
                            masks_data = pickle.load(f)

                        label_maps[img_idx] = label_map
                        masks_data_all[img_idx] = masks_data

                        print(f"    加载视图{img_idx}: {len(masks_data['masks'])}个掩码")
                    except Exception as e:
                        print(f"    加载视图{img_idx}失败: {e}")

            if len(label_maps) > 0:
                # 获取初始尺度（用于投影）
                from simple_knn._C import distCUDA2
                positions_tensor = torch.from_numpy(all_points).float().cuda()
                dist2 = torch.clamp_min(distCUDA2(positions_tensor), 0.0000001)
                scales = torch.sqrt(dist2).cpu().numpy()

                # 使用精确投影方法
                semantic_importance = self.assign_semantic_importance_precise(
                    all_points,
                    scales[:, None].repeat(3, axis=1),  # 转换为3D尺度
                    images,
                    camera_params[:min(10, len(camera_params))],
                    label_maps,
                    masks_data_all,
                    important_labels=important_labels
                )

                # 缓存语义数据供后续更新使用
                self.cached_sam_masks = label_maps
                self.cached_masks_data = masks_data_all
                self.cached_camera_params = camera_params[:min(10, len(camera_params))]

                print(f"\n精确语义分配完成:")
                print(f"  覆盖点数: {(semantic_importance > 0.3).sum()} / {num_points}")
                print(f"  平均重要性: {semantic_importance.mean():.3f}")
            else:
                print("  没有可用的掩码数据，使用默认语义重要性")
                semantic_importance = np.ones(num_points) * 0.3
        else:
            semantic_importance = np.ones(num_points) * 0.3

        # 基于3D空间特征的轻微调整
        print("  基于3D空间特征调整...")

        # 计算点云密度
        from scipy.spatial import cKDTree
        tree = cKDTree(all_points)
        k_neighbors = min(20, num_points - 1)
        distances, _ = tree.query(all_points, k=k_neighbors + 1)
        densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
        density_normalized = (densities - densities.min()) / (densities.max() - densities.min() + 1e-6)

        # 高度因子
        heights = all_points[:, 1]
        height_normalized = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)

        # 颜色变化因子
        color_variance = all_colors.std(axis=1)
        color_var_normalized = (color_variance - color_variance.min()) / (
                    color_variance.max() - color_variance.min() + 1e-6)

        # 综合空间因子（降低权重，映射到较小范围）
        spatial_factor = density_normalized * 0.4 + height_normalized * 0.3 + color_var_normalized * 0.3
        spatial_factor = spatial_factor * 0.2 + 0.4  # 映射到[0.4, 0.6]范围

        # 判断语义覆盖率
        semantic_coverage = (semantic_importance > 0.3).sum() / num_points

        # 保护高语义重要性的点
        high_semantic_mask = semantic_importance > 0.7
        num_high_semantic = high_semantic_mask.sum()

        if num_high_semantic > 0:
            print(f"  保护{num_high_semantic}个高语义重要性点")
            # 高语义点保持原值，不被空间特征影响
            preserved_high = semantic_importance[high_semantic_mask].copy()

        if semantic_coverage < 0.2:  # 语义覆盖率很低
            print(f"  语义覆盖率低({semantic_coverage * 100:.1f}%)，轻微增加空间特征权重")
            # 对没有语义信息的点，使用空间特征
            no_semantic_mask = semantic_importance <= 0.3
            semantic_importance[no_semantic_mask] = spatial_factor[no_semantic_mask] * 0.5 + 0.25
        else:
            print(f"  语义覆盖率正常({semantic_coverage * 100:.1f}%)，保持语义主导")
            # 只对没有语义信息的点做轻微调整
            no_semantic_mask = semantic_importance <= 0.3
            semantic_importance[no_semantic_mask] = spatial_factor[no_semantic_mask] * 0.4 + 0.3

        # 恢复高语义点的原值
        if num_high_semantic > 0:
            semantic_importance[high_semantic_mask] = preserved_high

        # 添加小噪声以增加多样性
        noise = np.random.normal(0, 0.01, num_points)
        semantic_importance = semantic_importance + noise

        # 最终裁剪到合理范围
        semantic_importance = np.clip(semantic_importance, 0.2, 0.95)

        # 统计
        high_count = np.sum(semantic_importance > 0.7)
        medium_count = np.sum((semantic_importance >= 0.5) & (semantic_importance <= 0.7))
        low_count = np.sum(semantic_importance < 0.5)

        print(f"\n语义重要性分布:")
        print(f"  高 (>0.7): {high_count} ({high_count / num_points * 100:.1f}%)")
        print(f"  中 (0.5-0.7): {medium_count} ({medium_count / num_points * 100:.1f}%)")
        print(f"  低 (<0.5): {low_count} ({low_count / num_points * 100:.1f}%)")
        print(f"  平均值: {semantic_importance.mean():.3f}")
        print(f"  标准差: {semantic_importance.std():.3f}")

        # 步骤3：创建高斯参数
        print("\n初始化高斯参数...")

        positions = torch.from_numpy(all_points).float().cuda()

        from utils.sh_utils import RGB2SH
        rgb_colors = torch.from_numpy(all_colors).float().cuda()

        features = torch.zeros((num_points, 3, (self.sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = RGB2SH(rgb_colors)
        features[:, 3:, 1:] = 0.0

        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        features_rest = features[:, :, 1:].transpose(1, 2).contiguous()

        opacities = torch.ones(num_points, device="cuda") * 0.1
        logit_opacities = torch.logit(opacities.clamp(0.01, 0.99))

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(positions), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rotations = torch.zeros((num_points, 4), device="cuda")
        rotations[:, 0] = 1.0

        gaussian_params = {
            'positions': positions,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'opacities': logit_opacities,
            'scales': scales,
            'rotations': rotations,
            'semantic_importance': torch.from_numpy(semantic_importance).float().cuda(),
            'semantic_labels': torch.zeros(num_points, dtype=torch.long).cuda()
        }

        print(f"✓ 完成初始化：{num_points}个高斯点，SH阶数={self.sh_degree}")

        self.cleanup_models()
        return gaussian_params

    def debug_projection_alignment(self, positions, camera, image_shape, label_map):
        """调试投影对齐问题"""
        h, w = image_shape

        # 投影所有点
        points_2d, depths, _ = self.project_gaussians_with_scale(
            positions[:1000],  # 只测试前1000个点
            np.ones((1000, 3)),  # 虚拟尺度
            camera,
            image_shape
        )

        # 统计投影分布
        in_image = 0
        depth_positive = 0

        for i, (pt2d, depth) in enumerate(zip(points_2d, depths)):
            if depth > 0:
                depth_positive += 1
                if 0 <= pt2d[0] < w and 0 <= pt2d[1] < h:
                    in_image += 1

                    # 打印前几个点的详细信息
                    if i < 5:
                        x, y = int(pt2d[0]), int(pt2d[1])
                        label = label_map[y, x]
                        print(f"    Point {i}: 3D={positions[i][:3]}, "
                              f"2D=({x},{y}), depth={depth:.2f}, label={label}")

        print(f"  Projection stats: {depth_positive}/1000 positive depth, "
              f"{in_image}/1000 in image bounds")

        # 检查点云范围
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        print(f"  Point cloud range: [{pos_min}] to [{pos_max}]")

        # 检查相机参数
        if 'R' in camera and 't' in camera:
            print(f"  Camera R:\n{camera['R']}")
            print(f"  Camera t: {camera['t']}")
            print(f"  Camera K:\n{camera['K']}")

        # ========== 在这里添加新代码 ==========
        # 统计3D点在不同标签上的分布
        # 首先需要获取重要标签ID（从调用处传入）
        # 注意：这需要修改方法签名来传入important_label_ids

        # 统计落在不同标签上的点数
        label_distribution = {}
        # 使用更多的点来获得更准确的统计
        sample_size = min(10000, len(positions))
        points_2d_full, depths_full, _ = self.project_gaussians_with_scale(
            positions[:sample_size],
            np.ones((sample_size, 3)),
            camera,
            image_shape
        )

        for i in range(sample_size):
            if depths_full[i] > 0 and 0 <= points_2d_full[i][0] < w and 0 <= points_2d_full[i][1] < h:
                x, y = int(points_2d_full[i][0]), int(points_2d_full[i][1])
                label = label_map[y, x]
                label_distribution[label] = label_distribution.get(label, 0) + 1

        # 打印标签分布
        total_count = sum(label_distribution.values())
        print(f"\n  Label distribution (sampling {sample_size} points):")
        sorted_labels = sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        for label, count in sorted_labels:
            percentage = count / total_count * 100
            print(f"    Label {label}: {count} points ({percentage:.1f}%)")

    def assign_semantic_importance_precise(self, positions, scales, images,
                                           camera_params, label_maps, masks_data_all,
                                           important_labels=None):
        """更精确的语义重要性分配 - 深度调试版本"""

        num_points = len(positions)
        device = positions.device if isinstance(positions, torch.Tensor) else 'cpu'

        # 初始化累积分数和权重
        semantic_scores = np.zeros(num_points, dtype=np.float32)
        vote_counts = np.zeros(num_points, dtype=np.float32)

        # 添加统计
        points_in_important_masks = 0
        points_with_clip_boost = 0

        # CLIP标签缓存
        clip_label_cache = {}

        # 处理每个视图
        for cam_idx, cam in enumerate(camera_params[:min(10, len(camera_params))]):
            if cam_idx not in label_maps:
                continue

            label_map = label_maps[cam_idx]
            masks_data = masks_data_all.get(cam_idx, None)
            if masks_data is None:
                continue

            h, w = label_map.shape

            # 为当前视图的掩码计算CLIP标签
            print(f"  View {cam_idx}: Computing CLIP labels for masks...")
            view_clip_labels = self._compute_clip_labels_for_masks(
                images[cam_idx] if cam_idx < len(images) else None,
                masks_data,
                important_labels
            )

            # 构建label_map值到掩码索引的映射
            label_to_mask_idx = {}
            important_label_ids = set()

            for mask_idx, (mask_info, segmentation) in enumerate(zip(
                    masks_data['masks'], masks_data['segmentations']
            )):
                # 找到这个掩码在label_map中的值
                mask_label_value = mask_idx + 1  # 通常是索引+1
                label_to_mask_idx[mask_label_value] = mask_idx

                # 检查是否是重要掩码
                mask_key = f"0_{mask_idx}"
                if mask_key in view_clip_labels and view_clip_labels[mask_key]['is_important']:
                    important_label_ids.add(mask_label_value)
                    clip_label_cache[f"{cam_idx}_{mask_idx}"] = view_clip_labels[mask_key]

            # 统计重要掩码覆盖
            important_mask_pixels = sum((label_map == lid).sum() for lid in important_label_ids)
            total_image_pixels = h * w
            important_coverage = important_mask_pixels / total_image_pixels
            print(f"  View {cam_idx}: Important masks cover {important_coverage * 100:.1f}% of image")
            print(f"  View {cam_idx}: Important label IDs in label_map: {important_label_ids}")

            # 深度调试：检查label_map的值分布
            unique_labels = np.unique(label_map)
            print(f"  View {cam_idx}: Unique labels in label_map: {unique_labels[:20]}...")  # 只打印前20个
            print(
                f"  View {cam_idx}: Max label value: {label_map.max()}, Non-zero pixels: {(label_map > 0).sum()}/{h * w}")

            # 投影点到当前视图
            points_2d, depths, projected_scales = self.project_gaussians_with_scale(
                positions, scales, cam, (h, w)
            )

            print(f"  View {cam_idx}: Processing {len(positions)} gaussians")

            if cam_idx == 0:  # 只在第一个视图调试
                print("\n  === Debugging projection alignment ===")
                self.debug_projection_alignment(
                    positions[:1000] if isinstance(positions, np.ndarray) else positions[:1000].cpu().numpy(),
                    cam,
                    (h, w),
                    label_map
                )

                # ========== 添加调试代码位置2：标签映射检查 ==========
            if cam_idx == 0:
                print("\n  === Checking label mapping ===")
                # 检查哪些标签对应truck相关的掩码
                for mask_idx, mask_info in enumerate(masks_data['masks'][:10]):
                    # 检查这个掩码在label_map中的实际覆盖
                    label_value = mask_idx + 1
                    pixels_with_label = (label_map == label_value).sum()

                    # 获取CLIP标签
                    mask_key = f"0_{mask_idx}"
                    clip_info = view_clip_labels.get(mask_key, {})

                    print(f"    Mask {mask_idx}: area={mask_info['area']}, "
                          f"label_id={label_value}, pixels={pixels_with_label}, "
                          f"CLIP={clip_info.get('label', 'N/A')}, "
                          f"important={clip_info.get('is_important', False)}")
                print("")

            # 深度调试：检查投影点的分布
            valid_projections = 0
            for i in range(len(points_2d)):
                if depths[i] > 0 and 0 <= points_2d[i][0] < w and 0 <= points_2d[i][1] < h:
                    valid_projections += 1
            print(f"  View {cam_idx}: Valid projections: {valid_projections}/{len(points_2d)}")

            # 采样一些投影点检查它们的位置
            sample_indices = np.random.choice(len(points_2d), min(100, len(points_2d)), replace=False)
            sample_labels = []
            for idx in sample_indices:
                if depths[idx] > 0 and 0 <= points_2d[idx][0] < w and 0 <= points_2d[idx][1] < h:
                    x, y = int(points_2d[idx][0]), int(points_2d[idx][1])
                    label = label_map[y, x]
                    sample_labels.append(label)

            label_counts = {}
            for label in sample_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"  View {cam_idx}: Sample point labels distribution: {dict(sorted(label_counts.items())[:10])}")

            # 检查重要标签在采样中的出现
            important_in_sample = sum(1 for label in sample_labels if label in important_label_ids)
            print(f"  View {cam_idx}: Important labels in sample: {important_in_sample}/{len(sample_labels)}")

            # 统计当前视图的命中情况
            view_points_in_masks = 0
            view_points_with_boost = 0

            # 批处理处理每个点
            batch_size = 10000
            for batch_start in range(0, num_points, batch_size):
                batch_end = min(batch_start + batch_size, num_points)

                for pt_idx in range(batch_start, batch_end):
                    if depths[pt_idx] <= 0:
                        continue

                    center_2d = points_2d[pt_idx]
                    if not (0 <= center_2d[0] < w and 0 <= center_2d[1] < h):
                        continue

                    # 直接检查中心点的掩码
                    center_x, center_y = int(center_2d[0]), int(center_2d[1])
                    center_label_id = label_map[center_y, center_x]

                    # 检查是否落在任何掩码内
                    if center_label_id > 0:
                        view_points_in_masks += 1

                        # 检查是否是重要掩码
                        if center_label_id in important_label_ids:
                            view_points_with_boost += 1
                            points_with_clip_boost += 1

                            # 打印前几个命中的点的详细信息
                            if view_points_with_boost <= 5:
                                mask_idx = label_to_mask_idx.get(center_label_id)
                                if mask_idx is not None and f"{cam_idx}_{mask_idx}" in clip_label_cache:
                                    label_info = clip_label_cache[f"{cam_idx}_{mask_idx}"]
                                    print(
                                        f"    Hit! Point {pt_idx} at ({center_x}, {center_y}) -> label {center_label_id} -> mask {mask_idx} -> {label_info['label']}")

                    # 原始的高斯覆盖采样
                    scale_2d = projected_scales[pt_idx]
                    radius = min(scale_2d * 2.0, 50)

                    samples = self.sample_gaussian_coverage(
                        center_2d, radius, (h, w)
                    )

                    # 累积语义投票
                    total_weight = 0.0
                    weighted_score = 0.0
                    found_important = False

                    for sample_x, sample_y, weight in samples:
                        label_id = label_map[sample_y, sample_x]

                        if label_id > 0:
                            # 使用映射找到对应的掩码索引
                            mask_idx = label_to_mask_idx.get(label_id)

                            if mask_idx is not None and mask_idx < len(masks_data['masks']):
                                mask_info = masks_data['masks'][mask_idx]

                                # 基础分数
                                base_importance = 0.3

                                # 掩码贡献
                                area_factor = min(mask_info['area'] / 10000, 1.0)
                                score_factor = mask_info['score']
                                mask_importance = area_factor * 0.15 + score_factor * 0.15

                                # CLIP标签加成
                                clip_boost = 0.0
                                mask_key = f"{cam_idx}_{mask_idx}"
                                if mask_key in clip_label_cache:
                                    label_info = clip_label_cache[mask_key]
                                    if label_info['is_important'] and label_info['score'] > 0.25:
                                        clip_boost = 0.4
                                        found_important = True

                                importance = base_importance + mask_importance + clip_boost
                                importance = min(importance, 1.0)

                                weighted_score += weight * importance
                                total_weight += weight

                    if found_important:
                        points_in_important_masks += 1

                    if total_weight > 0:
                        semantic_scores[pt_idx] += weighted_score
                        vote_counts[pt_idx] += total_weight

            print(
                f"  View {cam_idx}: {view_points_in_masks} points in masks, {view_points_with_boost} in important masks")

        # 打印总体统计
        print(f"\n  Overall statistics:")
        print(f"    Points in important masks (at least once): {points_in_important_masks}")
        print(f"    Points with CLIP boost (center check): {points_with_clip_boost}")

        # 归一化并应用多视图一致性
        final_scores = np.ones(num_points, dtype=np.float32) * 0.3

        valid_mask = vote_counts > 0
        if valid_mask.sum() > 0:
            final_scores[valid_mask] = semantic_scores[valid_mask] / vote_counts[valid_mask]

            # 应用多视图一致性平滑
            final_scores = self.enforce_multiview_consistency(
                final_scores, vote_counts, valid_mask
            )

        # 统计信息
        high_importance = (final_scores > 0.7).sum()
        medium_importance = ((final_scores >= 0.5) & (final_scores <= 0.7)).sum()
        low_importance = (final_scores < 0.5).sum()

        print(f"\n  Semantic assignment with CLIP labels:")
        print(f"    Points with valid scores: {valid_mask.sum()}")
        print(f"    High importance (>0.7): {high_importance} ({high_importance / num_points * 100:.1f}%)")
        print(f"    Medium importance (0.5-0.7): {medium_importance} ({medium_importance / num_points * 100:.1f}%)")
        print(f"    Low importance (<0.5): {low_importance} ({low_importance / num_points * 100:.1f}%)")
        print(
            f"    Score distribution: min={final_scores.min():.3f}, max={final_scores.max():.3f}, mean={final_scores.mean():.3f}")

        return final_scores

    def _compute_clip_labels_for_masks(self, image, masks_data, important_labels):
        """使用CLIP为掩码计算标签 - 只使用重要标签"""

        if image is None or not hasattr(self, 'clip_model'):
            return {}

        clip_labels = {}
        min_area_threshold = 500  # 最小像素面积阈值

        if important_labels is None or len(important_labels) == 0:
            important_labels = ["face", "text", "sign", "person", "car"]

        # 方案1：只使用重要标签作为CLIP提示
        text_prompts = list(important_labels)

        print(f"    Computing CLIP labels for {len(masks_data['masks'])} masks...")
        print(f"    Important labels: {important_labels}")

        # 批量处理掩码
        valid_regions = []
        valid_indices = []

        for i, (mask_info, segmentation) in enumerate(zip(
                masks_data['masks'][:50],  # 限制处理数量
                masks_data['segmentations'][:50]
        )):
            # 只处理足够大的掩码
            if mask_info['area'] < min_area_threshold:
                continue

            # 提取掩码区域
            bbox = mask_info['bbox']
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h

            # 确保边界有效
            if w < 20 or h < 20:
                continue

            # 提取图像区域
            region_img = image[y_min:y_max, x_min:x_max]
            if region_img.size > 0:
                region_tensor = torch.from_numpy(region_img.copy()).permute(2, 0, 1)
                valid_regions.append(region_tensor)
                valid_indices.append(i)

        if not valid_regions:
            return clip_labels

        # 批量计算CLIP相似度
        try:
            similarities = self.compute_clip_features_batch(valid_regions, text_prompts)

            # 为每个掩码分配标签
            for idx, mask_idx in enumerate(valid_indices):
                sim_scores = similarities[idx]
                best_label_idx = np.argmax(sim_scores)
                best_label = text_prompts[best_label_idx]
                best_score = float(sim_scores[best_label_idx])

                # 因为只有重要标签，所以识别出的都是重要的
                # 但需要置信度阈值来过滤低质量匹配
                is_important = best_score > 0.25  # 置信度阈值

                # 存储结果
                mask_key = f"{0}_{mask_idx}"  # 临时键，调用者会更新
                clip_labels[mask_key] = {
                    'label': best_label,
                    'score': best_score,
                    'is_important': is_important,
                    'area': masks_data['masks'][mask_idx]['area']
                }

                if is_important:
                    print(f"      Mask {mask_idx}: {best_label} (score: {best_score:.3f}) - IMPORTANT")

        except Exception as e:
            print(f"    Warning: CLIP computation failed: {e}")

        return clip_labels

    def project_gaussians_with_scale(self, positions, scales, camera, image_size):
        """投影3D高斯到2D，适配3DGS坐标系"""

        # 转换为numpy
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        if isinstance(scales, torch.Tensor):
            scales = scales.cpu().numpy()

        R = camera['R']
        t = camera['t']
        K = camera['K']
        h, w = image_size

        # 验证内参矩阵
        if abs(K[0, 0]) < 10 or abs(K[1, 1]) < 10:  # 焦距太小
            print(f"WARNING: Focal length too small: fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")
            # 使用默认焦距
            K[0, 0] = w
            K[1, 1] = h

        # 世界坐标到相机坐标
        points_cam = positions @ R.T + t.reshape(1, 3)

        # 深度（相机坐标系的z）
        depths = points_cam[:, 2]

        # 投影到图像平面
        points_2d_homo = points_cam @ K.T

        # 透视除法
        valid_mask = depths > 0.1  # 只处理相机前方的点
        points_2d = np.zeros((len(points_2d_homo), 2))
        points_2d[valid_mask] = points_2d_homo[valid_mask, :2] / points_2d_homo[valid_mask, 2:3]

        # 无效点设为图像外
        points_2d[~valid_mask] = [-1, -1]

        # 估计投影尺度
        focal_mean = (K[0, 0] + K[1, 1]) / 2.0
        mean_scales = np.mean(scales, axis=1) if scales.ndim > 1 else scales

        # 投影尺度与深度成反比
        projected_scales = np.zeros(len(positions))
        projected_scales[valid_mask] = (mean_scales[valid_mask] * focal_mean) / np.abs(depths[valid_mask])
        projected_scales = np.clip(projected_scales, 0.5, 100.0)

        return points_2d, depths, projected_scales

    def sample_gaussian_coverage(self, center, radius, image_size):
        """在高斯覆盖范围内采样点"""
        h, w = image_size
        samples = []

        # 限制采样范围
        radius = min(radius, 50)  # 最大半径50像素

        # 确定采样边界
        x_min = max(0, int(center[0] - radius))
        x_max = min(w - 1, int(center[0] + radius))
        y_min = max(0, int(center[1] - radius))
        y_max = min(h - 1, int(center[1] + radius))

        # 如果范围太小，至少采样中心点
        if x_max <= x_min or y_max <= y_min:
            cx, cy = int(center[0]), int(center[1])
            if 0 <= cx < w and 0 <= cy < h:
                samples.append((cx, cy, 1.0))
            return samples

        # 采样步长（根据半径调整）
        step = max(1, int(radius / 5))  # 大约采样25个点

        # 在范围内采样
        for y in range(y_min, y_max + 1, step):
            for x in range(x_min, x_max + 1, step):
                # 计算到中心的距离
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                if dist <= radius:
                    # 高斯权重
                    weight = np.exp(-0.5 * (dist / (radius * 0.5)) ** 2)
                    samples.append((x, y, weight))

        # 如果没有采样到任何点，至少返回中心点
        if len(samples) == 0:
            cx, cy = int(center[0]), int(center[1])
            if 0 <= cx < w and 0 <= cy < h:
                samples.append((cx, cy, 1.0))

        return samples

    def enforce_multiview_consistency(self, scores, vote_counts, valid_mask):
        """强制多视图一致性"""

        # 使用高斯滤波平滑（简化的空间一致性）
        # 这里简化处理：对邻近点的分数进行平滑

        smoothed_scores = scores.copy()

        # 找出需要平滑的点（在多个视图中可见）
        multi_view_mask = (vote_counts > 1.5) & valid_mask  # 至少1.5个视图

        if multi_view_mask.sum() > 100:
            # 对高置信度的点，减少变化
            high_confidence = vote_counts > 3.0
            smoothed_scores[high_confidence] = (
                    scores[high_confidence] * 0.8 +
                    scores[valid_mask].mean() * 0.2
            )

        return smoothed_scores

    def update_semantic_importance(self, gaussians, iteration, scene_extent):
        """定期重新计算语义重要性"""

        # 检查是否有缓存的语义数据
        if not hasattr(self, 'cached_sam_masks') or self.cached_sam_masks is None:
            return False

        # 检查更新条件
        if iteration % 1000 != 0 or iteration >= 10000:
            return False

        print(f"\n[Semantic Update] Updating semantic importance at iteration {iteration}")

        with torch.no_grad():
            # 获取当前位置和尺度
            current_positions = gaussians._xyz.detach().cpu().numpy()
            current_scales = torch.exp(gaussians._scaling.detach()).cpu().numpy()

            # 重新计算语义重要性
            new_importance = self.assign_semantic_importance_precise(
                current_positions,
                current_scales,
                None,  # 图像已经在缓存中
                self.cached_camera_params,
                self.cached_sam_masks,
                self.cached_masks_data
            )

            new_importance_tensor = torch.from_numpy(new_importance).float().cuda()

            # 平滑更新（避免突变）
            alpha = 0.3  # 更新率
            old_importance = gaussians._semantic_importance
            updated_importance = alpha * new_importance_tensor + (1 - alpha) * old_importance

            # 计算变化统计
            change = torch.abs(updated_importance - old_importance)
            significant_changes = (change > 0.1).sum().item()

            print(f"  Significant changes: {significant_changes} / {len(updated_importance)}")
            print(f"  Max change: {change.max().item():.3f}")
            print(f"  Mean change: {change.mean().item():.3f}")

            # 更新
            gaussians._semantic_importance = updated_importance

            # 记录位置用于移动检测
            if not hasattr(gaussians, '_last_update_positions'):
                gaussians._last_update_positions = gaussians._xyz.clone()
            else:
                # 计算移动距离
                movement = torch.norm(
                    gaussians._xyz - gaussians._last_update_positions, dim=1
                )
                moved_mask = movement > 0.1 * scene_extent

                if moved_mask.sum() > 100:
                    print(f"  {moved_mask.sum()} gaussians moved significantly, adjusting importance")
                    # 对移动过远的点降低重要性
                    gaussians._semantic_importance[moved_mask] *= 0.8

                gaussians._last_update_positions = gaussians._xyz.clone()

            # 确保范围合理
            gaussians._semantic_importance = torch.clamp(
                gaussians._semantic_importance, min=0.2, max=0.95
            )

        return True