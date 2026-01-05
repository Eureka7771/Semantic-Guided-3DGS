#!/usr/bin/env python3
"""
SAM-RPS: 语义增强的自适应密度控制 (重构版)
回归原始3DGS设计，语义作为增强而非替代
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gc


@dataclass
class DensificationConfig:
    """简化的密度控制配置"""
    # 原始3DGS参数
    grad_threshold: float = 0.0004  # 原始论文值
    percent_dense: float = 0.01  # 场景范围的1%
    opacity_cull: float = 0.005  # 最小不透明度
    # 新增：最小保留不透明度
    min_opacity_after_reset: float = 0.01  # 重置后的最小值
    prune_opacity_threshold: float = 0.005  # 剪枝阈值

    # 密度控制间隔
    densification_interval: int = 100
    opacity_reset_interval: int = 3000

    # 语义增强参数（可选）
    semantic_weight: float = 0.15  # 语义权重，不超过30%
    protected_labels: List[str] = None

    # 内存管理（简化）
    max_gaussians: int = 1100000  # 安全上限
    prune_extent: float = 0.1  # 场景10%以上视为过大

    def __post_init__(self):
        if self.protected_labels is None:
            self.protected_labels = ["face", "text", "sign"]


class SemanticAdaptiveDensification:
    """语义增强的自适应密度控制 - 重构版"""

    def __init__(self, config: DensificationConfig = None):
        self.config = config or DensificationConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 统计信息
        self.stats = {
            'total_splits': 0,
            'total_clones': 0,
            'total_prunes': 0,
            'semantic_boosts': 0
        }

        # 场景信息 - 初始化为None，只设置一次
        self.scene_extent = None
        self._scene_extent_fixed = False  # 标记是否已经固定
        self._original_scene_extent = None  # 添加这行来保存原始值

    def set_scene_extent(self, scene_extent_value: float):
        """设置场景范围 - 只能设置一次！"""
        if self._scene_extent_fixed:
            print(
                f"[RPS] Warning: Scene extent already fixed at {self.scene_extent:.3f}, ignoring new value {scene_extent_value:.3f}")
            return

        self.scene_extent = scene_extent_value
        self._original_scene_extent = scene_extent_value
        self._scene_extent_fixed = True
        print(f"[RPS] Scene extent fixed at: {self.scene_extent:.3f}")
        print(f"[RPS] Split threshold will be: {0.01 * self.scene_extent:.4f}")

    def compute_densification_stats(self,
                                    gaussians: Dict[str, torch.Tensor],
                                    viewspace_gradients: torch.Tensor,
                                    visibility_filter: torch.Tensor,
                                    radii: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算密度控制统计（原始3DGS方式）"""

        # 1. 标准梯度累积（原始3DGS）
        grad_norms = torch.zeros_like(gaussians['xyz_gradient_accum'])
        if visibility_filter.sum() > 0:
            grad_norms[visibility_filter] = torch.norm(
                viewspace_gradients[visibility_filter, :2], dim=-1, keepdim=True
            )

        # 2. 更新累积梯度
        gaussians['xyz_gradient_accum'][visibility_filter] += grad_norms[visibility_filter]
        gaussians['denom'][visibility_filter] += 1

        # 3. 计算平均梯度
        avg_grads = torch.zeros_like(gaussians['xyz_gradient_accum'])
        mask = gaussians['denom'] > 0
        avg_grads[mask] = gaussians['xyz_gradient_accum'][mask] / gaussians['denom'][mask]

        # 4. 更新最大2D半径（用于屏幕空间剪枝）
        gaussians['max_radii2D'][visibility_filter] = torch.max(
            gaussians['max_radii2D'][visibility_filter],
            radii[visibility_filter]
        )

        return {
            'avg_gradients': avg_grads,
            'visibility': visibility_filter
        }

    def densify_and_prune(self,
                          gaussians: Dict[str, torch.Tensor],
                          stats: Dict[str, torch.Tensor],
                          extent: float,
                          max_screen_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        执行密度控制和剪枝（内存感知版本）
        """
        grads = stats['avg_gradients'].squeeze()
        current_num_gaussians = len(gaussians['positions'])

        # 内存安全阈值
        HARD_LIMIT = 1100000  # 28万硬限制
        SOFT_LIMIT = 1100000  # 25万软限制

        if current_num_gaussians >= SOFT_LIMIT:
            # 临时禁用语义权重
            original_weight = self.config.semantic_weight
            self.config.semantic_weight = 0  # 完全禁用语义增强

            result = self._optimization_only_mode(gaussians, stats, extent, max_screen_size)

            # 恢复设置
            self.config.semantic_weight = original_weight
            return result

        print(f"\n[Densify] Current gaussians: {current_num_gaussians}")

        # === 策略选择 ===
        if current_num_gaussians >= HARD_LIMIT:
            # 达到硬限制：只优化，不增长
            print(f"[Densify] At hard limit, optimization only mode")
            return self._optimization_only_mode(gaussians, stats, extent, max_screen_size)

        elif current_num_gaussians > SOFT_LIMIT:
            # 接近限制：平衡模式
            print(f"[Densify] Near limit, balanced mode")
            return self._balanced_densification(gaussians, stats, extent, max_screen_size,
                                                available_space=HARD_LIMIT - current_num_gaussians)

        else:
            # 正常模式
            return self._normal_densification(gaussians, stats, extent, max_screen_size)

    def _normal_densification(self, gaussians: Dict[str, torch.Tensor],
                              stats: Dict[str, torch.Tensor],
                              extent: float,
                              max_screen_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """正常密度控制模式（原始3DGS逻辑）- 优化内存版本"""

        grads = stats['avg_gradients'].squeeze()

        # 1. 找出需要密度控制的点（高梯度）
        densify_mask = grads >= self.config.grad_threshold

        # 添加调试输出
        print(f"\n[Densify Debug] Checking densification:")
        print(f"  Points meeting grad threshold: {densify_mask.sum()}")
        print(f"  Total points: {len(grads)}")

        # 记录哪些点被修改了
        modified_mask = torch.zeros_like(densify_mask, dtype=torch.bool)

        if densify_mask.sum() == 0:
            # 没有需要密度控制的点，只执行剪枝
            pruned_gaussians = self._prune_gaussians(gaussians, extent, max_screen_size)

            # 只重置被剪枝影响的梯度（剪枝后索引会变化，所以需要全部重置）
            if len(pruned_gaussians['positions']) < len(gaussians['positions']):
                # 有点被剪枝了，需要重新初始化累积器以匹配新大小
                num_points = len(pruned_gaussians['positions'])
                pruned_gaussians['xyz_gradient_accum'] = torch.zeros((num_points, 1), device=self.device)
                pruned_gaussians['denom'] = torch.zeros((num_points, 1), device=self.device)
                # max_radii2D 也需要调整大小
                if 'max_radii2D' in pruned_gaussians:
                    # 保留未被剪枝点的max_radii2D值
                    # 注意：这里假设剪枝操作保持了顺序
                    pruned_gaussians['max_radii2D'] = torch.zeros(num_points, device=self.device)

            return pruned_gaussians

        # 2. 获取尺度信息
        scales = torch.exp(gaussians['scales'])
        max_scales = scales.max(dim=-1)[0]

        # 3. 计算尺度阈值
        scale_threshold = self.config.percent_dense * extent

        # 4. 分类：分裂 vs 克隆
        split_mask = densify_mask & (max_scales > scale_threshold)
        clone_mask = densify_mask & (max_scales <= scale_threshold)

        # 记录被分裂和克隆的点
        modified_mask = split_mask | clone_mask

        # ==== 语义增强部分（可选）====
        if 'semantic_importance' in gaussians and self.config.semantic_weight > 0:
            semantic_scores = gaussians['semantic_importance']
            high_semantic_mask = semantic_scores > 0.6
            semantic_threshold = self.config.grad_threshold * (1 - self.config.semantic_weight)

            # 找出满足语义增强条件的点
            semantic_boost_mask = high_semantic_mask & (grads >= semantic_threshold) & ~densify_mask

            if semantic_boost_mask.sum() > 0:
                print(f"  ✓ Adding {semantic_boost_mask.sum()} points via semantic boost!")

                # 更新掩码
                densify_mask = densify_mask | semantic_boost_mask

                # 更新分裂/克隆掩码
                enhanced_split = semantic_boost_mask & (max_scales > scale_threshold)
                enhanced_clone = semantic_boost_mask & (max_scales <= scale_threshold)
                split_mask = split_mask | enhanced_split
                clone_mask = clone_mask | enhanced_clone

                # 更新修改掩码
                modified_mask = modified_mask | semantic_boost_mask

                self.stats['semantic_boosts'] += semantic_boost_mask.sum().item()

        # 保存原始的梯度累积器（用于保留未修改点的值）
        original_grad_accum = gaussians['xyz_gradient_accum'].clone()
        original_denom = gaussians['denom'].clone()
        original_max_radii = gaussians.get('max_radii2D', None)
        if original_max_radii is not None:
            original_max_radii = original_max_radii.clone()

        # 5. 执行分裂
        if split_mask.sum() > 0:
            gaussians = self._split_gaussians(gaussians, split_mask, extent)
            self.stats['total_splits'] += split_mask.sum().item()
            print(f"\n[Densify] Split {split_mask.sum()} large gaussians")

        # 6. 执行克隆
        if clone_mask.sum() > 0:
            gaussians = self._clone_gaussians(gaussians, clone_mask)
            self.stats['total_clones'] += clone_mask.sum().item()
            print(f"[Densify] Clone {clone_mask.sum()} small gaussians")

        # 7. 剪枝
        pre_prune_count = len(gaussians['positions'])
        gaussians = self._prune_gaussians(gaussians, extent, max_screen_size)
        post_prune_count = len(gaussians['positions'])
        pruned = pre_prune_count != post_prune_count

        # 8. 智能重置梯度累积器
        num_points = len(gaussians['positions'])

        if pruned or split_mask.sum() > 0:
            # 如果有剪枝或分裂，需要完全重新初始化（因为索引变了）
            gaussians['xyz_gradient_accum'] = torch.zeros((num_points, 1), device=self.device)
            gaussians['denom'] = torch.zeros((num_points, 1), device=self.device)

            # max_radii2D 需要相应处理
            if 'max_radii2D' in gaussians:
                # 对于新增的点，初始化为0；对于保留的点，尽量保持原值
                # 但由于索引可能变化，这里简化处理，重新初始化
                if original_max_radii is not None and len(original_max_radii) == num_points and not pruned:
                    # 没有剪枝，只有克隆，可以部分保留
                    gaussians['max_radii2D'][:len(original_max_radii)] = original_max_radii
                else:
                    # 有剪枝或大小不匹配，重新初始化
                    gaussians['max_radii2D'] = torch.zeros(num_points, device=self.device)
        else:
            # 只有克隆操作，可以保留大部分原始值
            # 为原始点保留梯度，新克隆的点设为0
            num_original = len(original_grad_accum)
            num_new = num_points - num_original

            if num_new > 0:
                # 扩展累积器
                new_grad_accum = torch.zeros((num_points, 1), device=self.device)
                new_denom = torch.zeros((num_points, 1), device=self.device)

                # 保留原始点的值，但被克隆的点要重置
                keep_mask = ~clone_mask  # 未被克隆的点
                new_grad_accum[:num_original][keep_mask] = original_grad_accum[keep_mask]
                new_denom[:num_original][keep_mask] = original_denom[keep_mask]

                gaussians['xyz_gradient_accum'] = new_grad_accum
                gaussians['denom'] = new_denom

                # 处理max_radii2D
                if 'max_radii2D' in gaussians and original_max_radii is not None:
                    new_max_radii = torch.zeros(num_points, device=self.device)
                    new_max_radii[:num_original] = original_max_radii
                    gaussians['max_radii2D'] = new_max_radii
            else:
                # 没有新增点，只重置被修改的点
                gaussians['xyz_gradient_accum'][modified_mask] = 0
                gaussians['denom'][modified_mask] = 0
                # max_radii2D 不需要重置

        # 清理临时变量
        del original_grad_accum, original_denom
        if original_max_radii is not None:
            del original_max_radii

        # 立即清理CUDA缓存
        torch.cuda.empty_cache()

        return gaussians

    def _optimization_only_mode(self, gaussians: Dict[str, torch.Tensor],
                                stats: Dict[str, torch.Tensor],
                                extent: float,
                                max_screen_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        优化模式：达到硬限制后的策略
        - 保持正常剪枝标准
        - 剪枝产生的空间用于有限的分裂/克隆
        """
        initial_count = len(gaussians['positions'])

        # 步骤1：正常剪枝（不提高阈值）
        gaussians = self._prune_gaussians(gaussians, extent, max_screen_size)

        pruned_count = initial_count - len(gaussians['positions'])
        print(f"[Optimization Mode] Pruned {pruned_count} gaussians")

        if pruned_count > 0:
            # 步骤2：利用剪枝空出的空间进行有限的密度控制
            available_space = pruned_count

            # 重新计算梯度（基于剪枝后的高斯）
            if 'xyz_gradient_accum' not in gaussians or gaussians['denom'].sum() == 0:
                print(f"[Optimization Mode] No gradient information available")
                self._ensure_gradient_accumulators(gaussians)
                return gaussians

            avg_grads = torch.zeros_like(gaussians['xyz_gradient_accum'])
            mask = gaussians['denom'] > 0
            avg_grads[mask] = gaussians['xyz_gradient_accum'][mask] / gaussians['denom'][mask]
            grads = avg_grads.squeeze()

            # 找出最需要分裂/克隆的高斯（梯度最大的）
            densify_mask = grads >= self.config.grad_threshold

            if densify_mask.sum() > 0:
                # 获取梯度最大的候选
                candidate_grads = grads[densify_mask]
                num_candidates = min(densify_mask.sum().item(), available_space)

                if num_candidates > 0:
                    _, top_indices = torch.topk(candidate_grads, min(num_candidates, len(candidate_grads)))

                    # 创建受限的密度控制掩码
                    limited_mask = torch.zeros_like(densify_mask)
                    densify_indices = densify_mask.nonzero().squeeze(-1)
                    if densify_indices.dim() == 0:
                        densify_indices = densify_indices.unsqueeze(0)
                    limited_mask[densify_indices[top_indices]] = True

                    # 分类并执行
                    scales = torch.exp(gaussians['scales'])
                    max_scales = scales.max(dim=-1)[0]
                    scale_threshold = self.config.percent_dense * extent

                    # 计算训练进度以决定分裂优先级
                    training_progress = getattr(self, 'current_iteration', 20000) / getattr(self, 'max_iterations',
                                                                                            30000)

                    if training_progress > 0.7:  # 后期优先分裂
                        split_priority = 0.8
                    elif training_progress > 0.3:
                        split_priority = 0.6
                    else:
                        split_priority = 0.5

                    # 分配操作
                    split_mask = limited_mask & (max_scales > scale_threshold)
                    clone_mask = limited_mask & (max_scales <= scale_threshold)

                    # 根据优先级调整
                    target_splits = int(available_space * split_priority)
                    target_clones = available_space - target_splits

                    # 限制数量
                    if split_mask.sum() > target_splits:
                        split_grads = grads[split_mask]
                        _, top_split_indices = torch.topk(split_grads, min(target_splits, len(split_grads)))
                        final_split_mask = torch.zeros_like(split_mask)
                        split_indices = split_mask.nonzero().squeeze(-1)
                        if split_indices.dim() == 0:
                            split_indices = split_indices.unsqueeze(0)
                        final_split_mask[split_indices[top_split_indices]] = True
                        split_mask = final_split_mask

                    if clone_mask.sum() > target_clones:
                        clone_grads = grads[clone_mask]
                        _, top_clone_indices = torch.topk(clone_grads, min(target_clones, len(clone_grads)))
                        final_clone_mask = torch.zeros_like(clone_mask)
                        clone_indices = clone_mask.nonzero().squeeze(-1)
                        if clone_indices.dim() == 0:
                            clone_indices = clone_indices.unsqueeze(0)
                        final_clone_mask[clone_indices[top_clone_indices]] = True
                        clone_mask = final_clone_mask

                    # 执行操作
                    if split_mask.sum() > 0:
                        gaussians = self._split_gaussians(gaussians, split_mask, extent)
                        self.stats['total_splits'] += split_mask.sum().item()

                    if clone_mask.sum() > 0:
                        gaussians = self._clone_gaussians(gaussians, clone_mask)
                        self.stats['total_clones'] += clone_mask.sum().item()

                    print(f"[Optimization Mode] Used {available_space} freed space: "
                          f"{split_mask.sum()} splits, {clone_mask.sum()} clones "
                          f"(priority: {split_priority:.0%} split)")

                    # 关键修复：验证所有张量大小一致
                    final_count = len(gaussians['positions'])

                    # 确保所有辅助张量大小匹配
                    if 'xyz_gradient_accum' in gaussians and len(gaussians['xyz_gradient_accum']) != final_count:
                        # 扩展或截断到正确大小
                        if len(gaussians['xyz_gradient_accum']) < final_count:
                            padding = torch.zeros((final_count - len(gaussians['xyz_gradient_accum']), 1),
                                                  device=self.device)
                            gaussians['xyz_gradient_accum'] = torch.cat([gaussians['xyz_gradient_accum'], padding],
                                                                        dim=0)
                        else:
                            gaussians['xyz_gradient_accum'] = gaussians['xyz_gradient_accum'][:final_count]

                    if 'denom' in gaussians and len(gaussians['denom']) != final_count:
                        if len(gaussians['denom']) < final_count:
                            padding = torch.zeros((final_count - len(gaussians['denom']), 1), device=self.device)
                            gaussians['denom'] = torch.cat([gaussians['denom'], padding], dim=0)
                        else:
                            gaussians['denom'] = gaussians['denom'][:final_count]

                    if 'max_radii2D' in gaussians and len(gaussians['max_radii2D']) != final_count:
                        if len(gaussians['max_radii2D']) < final_count:
                            padding = torch.zeros(final_count - len(gaussians['max_radii2D']), device=self.device)
                            gaussians['max_radii2D'] = torch.cat([gaussians['max_radii2D'], padding], dim=0)
                        else:
                            gaussians['max_radii2D'] = gaussians['max_radii2D'][:final_count]

        final_count = len(gaussians['positions'])

        # 强制重新创建所有梯度累积器（释放旧内存）
        gaussians['xyz_gradient_accum'] = torch.zeros((final_count, 1), device=self.device)
        gaussians['denom'] = torch.zeros((final_count, 1), device=self.device)

        # max_radii2D 也需要调整大小
        if 'max_radii2D' in gaussians:
            gaussians['max_radii2D'] = gaussians['max_radii2D'][:final_count].contiguous()

        # 强制垃圾回收
        torch.cuda.empty_cache()

        return gaussians

    def _balanced_densification(self, gaussians: Dict[str, torch.Tensor],
                                stats: Dict[str, torch.Tensor],
                                extent: float,
                                max_screen_size: Optional[int],
                                available_space: int) -> Dict[str, torch.Tensor]:
        """
        平衡模式：接近限制时的策略
        - 正常剪枝
        - 限制增长数量
        - 优先处理高梯度区域
        """
        initial_count = len(gaussians['positions'])

        # 先执行正常剪枝
        gaussians = self._prune_gaussians(gaussians, extent, max_screen_size)
        current_count = len(gaussians['positions'])

        print(f"[Balanced Mode] After pruning: {initial_count} -> {current_count}")

        # 更新可用空间
        actual_available = min(available_space, 900000 - current_count)

        if actual_available <= 0:
            print(f"[Balanced Mode] No space available after pruning")
            self._ensure_gradient_accumulators(gaussians)
            return gaussians

        # 重要：剪枝后需要重新获取梯度信息
        # 因为剪枝改变了高斯的数量，原始的stats已经不匹配了
        # 所以我们需要基于当前的gaussians重新计算

        # 检查是否有有效的梯度信息
        if 'xyz_gradient_accum' not in gaussians or gaussians['denom'].sum() == 0:
            print(f"[Balanced Mode] No gradient information available")
            return gaussians

        # 重新计算平均梯度（基于剪枝后的高斯）
        avg_grads = torch.zeros_like(gaussians['xyz_gradient_accum'])
        mask = gaussians['denom'] > 0
        avg_grads[mask] = gaussians['xyz_gradient_accum'][mask] / gaussians['denom'][mask]
        grads = avg_grads.squeeze()

        # 正常密度控制，但限制数量
        densify_mask = grads >= self.config.grad_threshold

        if densify_mask.sum() == 0:
            self._ensure_gradient_accumulators(gaussians)
            return gaussians

        # 如果需要的操作数超过可用空间，按梯度优先级选择
        needed_operations = densify_mask.sum().item()
        if needed_operations > actual_available:
            print(f"[Balanced Mode] Limiting operations from {needed_operations} to {actual_available}")
            # 选择梯度最大的点
            candidate_grads = grads[densify_mask]
            _, top_indices = torch.topk(candidate_grads, min(actual_available, len(candidate_grads)))

            limited_mask = torch.zeros_like(densify_mask)
            densify_indices = densify_mask.nonzero().squeeze(-1)
            if densify_indices.dim() == 0:  # 处理只有一个点的情况
                densify_indices = densify_indices.unsqueeze(0)
            limited_mask[densify_indices[top_indices]] = True
            densify_mask = limited_mask

        # 执行正常的分裂/克隆
        scales = torch.exp(gaussians['scales'])
        max_scales = scales.max(dim=-1)[0]
        scale_threshold = self.config.percent_dense * extent

        # 这里的掩码现在基于剪枝后的大小
        split_mask = densify_mask & (max_scales > scale_threshold)
        clone_mask = densify_mask & (max_scales <= scale_threshold)

        # 语义增强（降低权重）
        if 'semantic_importance' in gaussians and self.config.semantic_weight > 0:
            # 在接近限制时降低语义权重
            reduced_weight = self.config.semantic_weight * 0.3
            semantic_scores = gaussians['semantic_importance']

            # 只对极高语义重要性的点降低阈值
            high_semantic_mask = semantic_scores > 0.8
            semantic_threshold = self.config.grad_threshold * (1 - reduced_weight)
            semantic_boost_mask = high_semantic_mask & (grads >= semantic_threshold) & ~densify_mask

            # 限制语义增强的数量
            max_boost = max(1, actual_available // 10)  # 最多10%，至少1个
            if semantic_boost_mask.sum() > max_boost:
                boost_grads = grads[semantic_boost_mask]
                _, top_boost_indices = torch.topk(boost_grads, min(max_boost, len(boost_grads)))
                final_boost_mask = torch.zeros_like(semantic_boost_mask)
                boost_indices = semantic_boost_mask.nonzero().squeeze(-1)
                if boost_indices.dim() == 0:
                    boost_indices = boost_indices.unsqueeze(0)
                final_boost_mask[boost_indices[top_boost_indices]] = True
                semantic_boost_mask = final_boost_mask

            # 更新掩码
            if semantic_boost_mask.sum() > 0:
                enhanced_split = semantic_boost_mask & (max_scales > scale_threshold)
                enhanced_clone = semantic_boost_mask & (max_scales <= scale_threshold)
                split_mask = split_mask | enhanced_split
                clone_mask = clone_mask | enhanced_clone
                self.stats['semantic_boosts'] += semantic_boost_mask.sum().item()

        # 执行操作
        if split_mask.sum() > 0:
            gaussians = self._split_gaussians(gaussians, split_mask, extent)
            self.stats['total_splits'] += split_mask.sum().item()

        if clone_mask.sum() > 0:
            gaussians = self._clone_gaussians(gaussians, clone_mask)
            self.stats['total_clones'] += clone_mask.sum().item()

        # 重置梯度
        self._ensure_gradient_accumulators(gaussians)

        final_count = len(gaussians['positions'])
        print(f"[Balanced Mode] Complete: {current_count} -> {final_count} "
              f"(available: {actual_available}, split: {split_mask.sum()}, clone: {clone_mask.sum()})")

        return gaussians

    def _reset_gradients_for_modified(self, gaussians: Dict[str, torch.Tensor],
                                      modified_mask: torch.Tensor):
        """只重置被修改的高斯的梯度 - 原始3DGS方式"""
        if modified_mask.sum() == 0:
            return

        # 只重置被修改的高斯
        gaussians['xyz_gradient_accum'][modified_mask] = 0
        gaussians['denom'][modified_mask] = 0
        # 注意：max_radii2D不需要重置，它是累积最大值

    def _ensure_gradient_accumulators(self, gaussians: Dict[str, torch.Tensor]):
        """确保梯度累积器存在且大小正确 - 保留已有的累积值"""
        num_points = len(gaussians['positions'])

        # 处理 xyz_gradient_accum
        if 'xyz_gradient_accum' not in gaussians:
            # 只在不存在时创建
            gaussians['xyz_gradient_accum'] = torch.zeros((num_points, 1), device=self.device)
        elif len(gaussians['xyz_gradient_accum']) < num_points:
            # 如果需要扩展（新增了高斯），保留旧值，为新高斯添加零
            old_size = len(gaussians['xyz_gradient_accum'])
            new_accum = torch.zeros((num_points, 1), device=self.device)
            new_accum[:old_size] = gaussians['xyz_gradient_accum']
            gaussians['xyz_gradient_accum'] = new_accum
        elif len(gaussians['xyz_gradient_accum']) > num_points:
            # 如果需要缩小（不应该发生，因为剪枝已经处理了）
            gaussians['xyz_gradient_accum'] = gaussians['xyz_gradient_accum'][:num_points]

        # 处理 denom
        if 'denom' not in gaussians:
            gaussians['denom'] = torch.zeros((num_points, 1), device=self.device)
        elif len(gaussians['denom']) < num_points:
            old_size = len(gaussians['denom'])
            new_denom = torch.zeros((num_points, 1), device=self.device)
            new_denom[:old_size] = gaussians['denom']
            gaussians['denom'] = new_denom
        elif len(gaussians['denom']) > num_points:
            gaussians['denom'] = gaussians['denom'][:num_points]

        # 处理 max_radii2D
        if 'max_radii2D' not in gaussians:
            gaussians['max_radii2D'] = torch.zeros(num_points, device=self.device)
        elif len(gaussians['max_radii2D']) < num_points:
            old_size = len(gaussians['max_radii2D'])
            new_radii = torch.zeros(num_points, device=self.device)
            new_radii[:old_size] = gaussians['max_radii2D']
            gaussians['max_radii2D'] = new_radii
        elif len(gaussians['max_radii2D']) > num_points:
            gaussians['max_radii2D'] = gaussians['max_radii2D'][:num_points]

    def _split_gaussians(self,
                         gaussians: Dict[str, torch.Tensor],
                         mask: torch.Tensor,
                         extent: float) -> Dict[str, torch.Tensor]:
        """分裂大尺度高斯（原始3DGS实现 - 修复版）"""
        selected_pts = mask.nonzero().squeeze(-1)
        n = len(selected_pts)

        if n == 0:
            return gaussians

        # 采样数（原始论文使用2）
        n_samples = 2

        # 在高斯分布内采样新位置
        selected_scales = torch.exp(gaussians['scales'][selected_pts])

        # 限制分裂后的尺度（防止产生过大的新高斯）
        max_allowed_scale = extent * 0.02  # 最大允许尺度为场景的2%

        # 确保max_allowed_scale是tensor并在正确的设备上
        max_allowed_scale_tensor = torch.tensor(max_allowed_scale, device=selected_scales.device)
        selected_scales = torch.min(selected_scales, max_allowed_scale_tensor)

        selected_rots = gaussians['rotations'][selected_pts]

        # 生成采样点
        stds = selected_scales.repeat(n_samples, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        # 应用旋转
        rots = self._build_rotation(selected_rots).repeat(n_samples, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = new_xyz + gaussians['positions'][selected_pts].repeat(n_samples, 1)

        # 新高斯的属性 - 修复：使用固定缩放因子1.6而不是除以n_samples
        new_scaling = torch.log(selected_scales.repeat(n_samples, 1) / 1.6)

        # 额外限制：确保新尺度不会太大
        max_log_scale = torch.log(max_allowed_scale_tensor)
        new_scaling = torch.clamp(new_scaling, max=max_log_scale)

        # 其他属性保持不变
        new_rotation = gaussians['rotations'][selected_pts].repeat(n_samples, 1)
        new_features_dc = gaussians['features_dc'][selected_pts].repeat(n_samples, 1, 1)
        new_features_rest = gaussians['features_rest'][selected_pts].repeat(n_samples, 1, 1)
        new_opacity = gaussians['opacities'][selected_pts].repeat(n_samples, 1)

        # 处理语义属性
        new_semantic_importance = None
        new_semantic_labels = None
        new_is_exploratory = None

        if 'semantic_importance' in gaussians:
            new_semantic_importance = gaussians['semantic_importance'][selected_pts].repeat(n_samples)
        if 'semantic_labels' in gaussians:
            new_semantic_labels = gaussians['semantic_labels'][selected_pts].repeat(n_samples)
        if 'is_exploratory' in gaussians:
            new_is_exploratory = gaussians['is_exploratory'][selected_pts].repeat(n_samples)

        # 合并新旧高斯
        keep_mask = ~mask
        pruned_gaussians = self._prune_by_mask(gaussians, keep_mask)

        # 添加新高斯
        result = {
            'positions': torch.cat([pruned_gaussians['positions'], new_xyz], dim=0),
            'scales': torch.cat([pruned_gaussians['scales'], new_scaling], dim=0),
            'rotations': torch.cat([pruned_gaussians['rotations'], new_rotation], dim=0),
            'features_dc': torch.cat([pruned_gaussians['features_dc'], new_features_dc], dim=0),
            'features_rest': torch.cat([pruned_gaussians['features_rest'], new_features_rest], dim=0),
            'opacities': torch.cat([pruned_gaussians['opacities'], new_opacity], dim=0),
        }

        # 添加语义属性
        if new_semantic_importance is not None:
            result['semantic_importance'] = torch.cat([
                pruned_gaussians['semantic_importance'], new_semantic_importance
            ], dim=0)
        if new_semantic_labels is not None:
            result['semantic_labels'] = torch.cat([
                pruned_gaussians['semantic_labels'], new_semantic_labels
            ], dim=0)
        if new_is_exploratory is not None:
            result['is_exploratory'] = torch.cat([
                pruned_gaussians['is_exploratory'], new_is_exploratory
            ], dim=0)

        # 新增：处理梯度累积器
        num_old = len(pruned_gaussians['positions'])
        num_new = len(new_xyz)

        # 保留未被分裂高斯的梯度，新高斯梯度为0
        if 'xyz_gradient_accum' in gaussians:
            old_grad_accum = gaussians['xyz_gradient_accum'][keep_mask]
            new_grad_accum = torch.zeros((num_new, 1), device=self.device)
            result['xyz_gradient_accum'] = torch.cat([old_grad_accum, new_grad_accum], dim=0)

        if 'denom' in gaussians:
            old_denom = gaussians['denom'][keep_mask]
            new_denom = torch.zeros((num_new, 1), device=self.device)
            result['denom'] = torch.cat([old_denom, new_denom], dim=0)

        if 'max_radii2D' in gaussians:
            old_radii = gaussians['max_radii2D'][keep_mask]
            new_radii = torch.zeros(num_new, device=self.device)
            result['max_radii2D'] = torch.cat([old_radii, new_radii], dim=0)

        if 'optimizer' in gaussians and gaussians['optimizer'] is not None:
            optimizer = gaussians['optimizer']
            # 获取被分裂的参数索引
            removed_indices = mask.nonzero().squeeze(-1)

            # 清理旧状态
            for group in optimizer.param_groups:
                old_param = group['params'][0]
                if old_param in optimizer.state:
                    # 删除被分裂高斯的优化器状态
                    del optimizer.state[old_param]

            result['optimizer'] = optimizer

        return result

    def _clone_gaussians(self, gaussians: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """克隆小尺度高斯（原始3DGS实现）"""
        selected_pts = mask.nonzero().squeeze(-1)
        n = len(selected_pts)

        if n == 0:
            return gaussians

        # 直接复制，不添加任何偏移！
        new_xyz = gaussians['positions'][selected_pts]
        # 添加微小扰动！
        perturbation = torch.randn_like(new_xyz) * 0.001 * self.scene_extent
        new_xyz = new_xyz + perturbation
        new_scaling = gaussians['scales'][selected_pts]
        new_rotation = gaussians['rotations'][selected_pts]
        new_features_dc = gaussians['features_dc'][selected_pts]
        new_features_rest = gaussians['features_rest'][selected_pts]
        new_opacity = gaussians['opacities'][selected_pts]

        # 处理语义属性（如果存在）
        new_semantic_importance = None
        new_semantic_labels = None
        new_is_exploratory = None

        if 'semantic_importance' in gaussians:
            new_semantic_importance = gaussians['semantic_importance'][selected_pts]
        if 'semantic_labels' in gaussians:
            new_semantic_labels = gaussians['semantic_labels'][selected_pts]
        if 'is_exploratory' in gaussians:
            new_is_exploratory = gaussians['is_exploratory'][selected_pts]

        # 合并新旧高斯
        result = {
            'positions': torch.cat([gaussians['positions'], new_xyz], dim=0),
            'scales': torch.cat([gaussians['scales'], new_scaling], dim=0),
            'rotations': torch.cat([gaussians['rotations'], new_rotation], dim=0),
            'features_dc': torch.cat([gaussians['features_dc'], new_features_dc], dim=0),
            'features_rest': torch.cat([gaussians['features_rest'], new_features_rest], dim=0),
            'opacities': torch.cat([gaussians['opacities'], new_opacity], dim=0),
        }

        # 添加语义属性（如果存在）
        if new_semantic_importance is not None:
            result['semantic_importance'] = torch.cat([
                gaussians['semantic_importance'], new_semantic_importance
            ], dim=0)
        if new_semantic_labels is not None:
            result['semantic_labels'] = torch.cat([
                gaussians['semantic_labels'], new_semantic_labels
            ], dim=0)
        if new_is_exploratory is not None:
            result['is_exploratory'] = torch.cat([
                gaussians['is_exploratory'], new_is_exploratory
            ], dim=0)

        # 处理梯度累积器（新高斯的梯度从0开始）
        if 'xyz_gradient_accum' in gaussians:
            new_grad_accum = torch.zeros((len(new_xyz), 1), device=self.device)
            result['xyz_gradient_accum'] = torch.cat([gaussians['xyz_gradient_accum'], new_grad_accum], dim=0)

        if 'denom' in gaussians:
            new_denom = torch.zeros((len(new_xyz), 1), device=self.device)
            result['denom'] = torch.cat([gaussians['denom'], new_denom], dim=0)

        if 'max_radii2D' in gaussians:
            new_radii = torch.zeros(len(new_xyz), device=self.device)
            result['max_radii2D'] = torch.cat([gaussians['max_radii2D'], new_radii], dim=0)

        # 处理优化器（如果存在）
        if 'optimizer' in gaussians:
            result['optimizer'] = gaussians['optimizer']

        return result

    def _prune_gaussians(self,
                         gaussians: Dict[str, torch.Tensor],
                         extent: float,
                         max_screen_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """剪枝低质量高斯（原始3DGS逻辑）"""
        prune_mask = torch.zeros(len(gaussians['positions']), dtype=torch.bool, device=self.device)

        # 1. 不透明度剪枝 - 原始3DGS只有这个
        opacity_mask = torch.sigmoid(gaussians['opacities'].squeeze()) < self.config.opacity_cull
        prune_mask = prune_mask | opacity_mask

        # 2. 屏幕空间剪枝 - 原始3DGS的第二个条件
        if max_screen_size is not None and 'max_radii2D' in gaussians:
            big_points_vs = gaussians['max_radii2D'] > max_screen_size
            prune_mask = prune_mask | big_points_vs

        # 就这样！没有其他条件！

        # 执行剪枝
        if prune_mask.sum() > 0:
            self.stats['total_prunes'] += prune_mask.sum().item()
            keep_mask = ~prune_mask
            return self._prune_by_mask(gaussians, keep_mask)

        return gaussians

    def _prune_by_mask(self, gaussians: Dict[str, torch.Tensor],
                       keep_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据掩码剪枝高斯 - 修复优化器状态管理"""
        pruned = {}

        # 剪枝所有张量
        for key, value in gaussians.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == len(keep_mask):
                pruned[key] = value[keep_mask]
            else:
                pruned[key] = value

        # ===== 修复：正确清理优化器状态 =====
        if 'optimizer' in gaussians and gaussians['optimizer'] is not None:
            optimizer = gaussians['optimizer']

            # 这是关键：不能直接操作原优化器，需要更新参数引用
            for group in optimizer.param_groups:
                if len(group["params"]) == 1:
                    old_param = group["params"][0]
                    param_name = group.get("name", "unknown")

                    # 只处理需要剪枝的参数
                    if old_param.shape[0] == len(keep_mask):
                        # 1. 先删除旧的优化器状态
                        if old_param in optimizer.state:
                            del optimizer.state[old_param]

                        # 2. 创建新参数（已剪枝）
                        new_param = old_param[keep_mask]
                        new_param = nn.Parameter(new_param.requires_grad_(True))

                        # 3. 更新参数引用 - 这是关键！
                        group["params"][0] = new_param

                        # 4. 优化器状态会在下次step时自动创建

            pruned['optimizer'] = optimizer

        return pruned

    def _build_rotation(self, r: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵"""
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] +
                          r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device=self.device)

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def reset_opacity(self, gaussians: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """重置不透明度（原始3DGS方式）"""
        opacities_new = torch.min(
            gaussians['opacities'],
            torch.logit(torch.tensor([0.01], device=self.device))
        )
        gaussians['opacities'] = opacities_new
        return gaussians

    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_splits': 0,
            'total_clones': 0,
            'total_prunes': 0,
            'semantic_boosts': 0
        }