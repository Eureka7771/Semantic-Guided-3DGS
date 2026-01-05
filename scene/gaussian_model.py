#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Dict, List, Optional, Tuple

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # 语义增强属性
        self._semantic_importance = torch.empty(0)
        self._semantic_labels = torch.empty(0)
        self._is_exploratory = torch.empty(0)
        self._creation_iter = torch.empty(0)

        # 训练参数
        self.training_args = None
        self.exposure_mapping = {}  # 空字典，避免保存时报错

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float):
        """从点云创建高斯模型 - 符合原始3DGS设计"""
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 原始3DGS方式：使用k近邻距离作为初始尺度
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # 原始3DGS：单位四元数初始化
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 原始3DGS：统一初始化不透明度为0.1
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 设置参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # 初始化梯度累积器
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 初始化语义属性（您的扩展）
        self._semantic_importance = torch.ones(self.get_xyz.shape[0], device="cuda") * 0.5
        self._semantic_labels = torch.zeros(self.get_xyz.shape[0], dtype=torch.long, device="cuda")
        self._is_exploratory = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        self._creation_iter = torch.zeros(self.get_xyz.shape[0], dtype=torch.long, device="cuda")

    def create_from_semantic_init(self, params: Dict[str, torch.Tensor], spatial_lr_scale: float):
        """从语义初始化参数创建高斯模型"""
        self.spatial_lr_scale = spatial_lr_scale

        num_points = params['positions'].shape[0]

        # 初始化基本参数
        self._xyz = nn.Parameter(params['positions'].requires_grad_(True))

        # 处理SH特征（兼容两种格式）
        if 'features_dc' in params and 'features_rest' in params:
            # 新格式：已经分离的DC和rest
            self._features_dc = nn.Parameter(params['features_dc'].requires_grad_(True))
            self._features_rest = nn.Parameter(params['features_rest'].requires_grad_(True))
        elif 'colors' in params:
            # 旧格式：只有RGB颜色，需要转换
            colors = params['colors']
            colors = torch.clamp(colors, 0.0, 1.0)
            sh_dc = RGB2SH(colors)
            self._features_dc = nn.Parameter(sh_dc.unsqueeze(1).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(
                torch.zeros((num_points, (self.max_sh_degree + 1) ** 2 - 1, 3),
                            dtype=torch.float32, device="cuda").requires_grad_(True)
            )

        # 初始化其他参数
        if params['opacities'].dim() == 1:
            self._opacity = nn.Parameter(params['opacities'].unsqueeze(-1).requires_grad_(True))
        else:
            self._opacity = nn.Parameter(params['opacities'].requires_grad_(True))

        self._scaling = nn.Parameter(params['scales'].requires_grad_(True))
        self._rotation = nn.Parameter(params['rotations'].requires_grad_(True))

        # 初始化辅助属性
        self.max_radii2D = torch.zeros((num_points), device="cuda")
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
        self.denom = torch.zeros((num_points, 1), device="cuda")

        # 初始化语义属性
        self._semantic_importance = params.get('semantic_importance', torch.ones(num_points, device="cuda") * 0.5)
        self._semantic_labels = params.get('semantic_labels', torch.zeros(num_points, dtype=torch.long, device="cuda"))
        self._is_exploratory = params.get('is_exploratory', torch.zeros(num_points, dtype=torch.bool, device="cuda"))
        self._creation_iter = torch.zeros(num_points, dtype=torch.long, device="cuda")

        self.active_sh_degree = 0
        self.optimizer = None

        print(f"✓ Created {num_points} gaussians from semantic initialization")
        print(f"  SH degree: {self.max_sh_degree}")
        print(f"  Features DC shape: {self._features_dc.shape}")
        print(f"  Features rest shape: {self._features_rest.shape}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.training_args = training_args

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        """重置不透明度（原始3DGS方式 - 正确版本）"""
        # 获取当前的不透明度（sigmoid空间）
        opacities_current = self.get_opacity

        # 原始3DGS：将所有不透明度与0.01取最小值
        opacities_new = torch.min(opacities_current, torch.ones_like(opacities_current) * 0.01)

        # 转换回logit空间
        opacities_new = self.inverse_opacity_activation(opacities_new)

        # 更新参数
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        print(f"[Reset Opacity] Reset {(opacities_current > 0.01).sum()} gaussians to 0.01")

    def get_params_dict(self) -> Dict[str, torch.Tensor]:
        """获取所有参数的字典形式"""
        params_dict = {
            'positions': self._xyz,
            'scales': self._scaling,
            'rotations': self._rotation,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'opacities': self._opacity,
            # 梯度累积器
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'max_radii2D': self.max_radii2D
        }

        # 确保语义相关参数总是存在于字典中
        num_points = len(self._xyz)

        # semantic_importance - 总是添加到字典
        if hasattr(self, '_semantic_importance') and len(self._semantic_importance) > 0:
            params_dict['semantic_importance'] = self._semantic_importance
        else:
            # 如果不存在或为空，创建默认值
            if not hasattr(self, '_semantic_importance'):
                self._semantic_importance = torch.ones(num_points, device="cuda") * 0.5
            params_dict['semantic_importance'] = self._semantic_importance

        # semantic_labels - 总是添加到字典
        if hasattr(self, '_semantic_labels') and len(self._semantic_labels) > 0:
            params_dict['semantic_labels'] = self._semantic_labels
        else:
            # 如果不存在或为空，创建默认值
            if not hasattr(self, '_semantic_labels'):
                self._semantic_labels = torch.zeros(num_points, dtype=torch.long, device="cuda")
            params_dict['semantic_labels'] = self._semantic_labels

        # is_exploratory - 总是添加到字典
        if hasattr(self, '_is_exploratory') and len(self._is_exploratory) > 0:
            params_dict['is_exploratory'] = self._is_exploratory
        else:
            # 如果不存在或为空，创建默认值
            if not hasattr(self, '_is_exploratory'):
                self._is_exploratory = torch.zeros(num_points, dtype=torch.bool, device="cuda")
            params_dict['is_exploratory'] = self._is_exploratory

        # creation_iter - 总是添加到字典
        if hasattr(self, '_creation_iter') and len(self._creation_iter) > 0:
            params_dict['creation_iter'] = self._creation_iter
        else:
            # 如果不存在或为空，创建默认值
            if not hasattr(self, '_creation_iter'):
                self._creation_iter = torch.zeros(num_points, dtype=torch.long, device="cuda")
            params_dict['creation_iter'] = self._creation_iter

        return params_dict

    def update_from_params_dict(self, params: Dict[str, torch.Tensor]):
        """从参数字典更新高斯模型"""
        # 更新基本参数
        if 'positions' in params:
            self._xyz = nn.Parameter(params['positions'].requires_grad_(True))
        if 'scales' in params:
            self._scaling = nn.Parameter(params['scales'].requires_grad_(True))
        if 'rotations' in params:
            self._rotation = nn.Parameter(params['rotations'].requires_grad_(True))
        if 'features_dc' in params:
            self._features_dc = nn.Parameter(params['features_dc'].requires_grad_(True))
        if 'features_rest' in params:
            self._features_rest = nn.Parameter(params['features_rest'].requires_grad_(True))
        if 'opacities' in params:
            self._opacity = nn.Parameter(params['opacities'].requires_grad_(True))

        # 更新梯度累积器
        if 'xyz_gradient_accum' in params:
            self.xyz_gradient_accum = params['xyz_gradient_accum']
        if 'denom' in params:
            self.denom = params['denom']
        if 'max_radii2D' in params:
            self.max_radii2D = params['max_radii2D']

        # 获取新的高斯数量
        num_gaussians = len(self._xyz)
        device = self._xyz.device

        # 确保所有辅助张量尺寸匹配
        # 1. 语义重要性
        if 'semantic_importance' in params:
            self._semantic_importance = params['semantic_importance']
        elif hasattr(self, '_semantic_importance'):
            if len(self._semantic_importance) != num_gaussians:
                if len(self._semantic_importance) < num_gaussians:
                    # 新增的高斯使用默认值
                    new_count = num_gaussians - len(self._semantic_importance)
                    new_importance = torch.ones(new_count, device=device) * 0.5
                    self._semantic_importance = torch.cat([self._semantic_importance, new_importance])
                else:
                    self._semantic_importance = self._semantic_importance[:num_gaussians]
        else:
            self._semantic_importance = torch.ones(num_gaussians, device=device) * 0.5

        # 2. 语义标签
        if 'semantic_labels' in params:
            self._semantic_labels = params['semantic_labels']
        elif hasattr(self, '_semantic_labels'):
            if len(self._semantic_labels) != num_gaussians:
                if len(self._semantic_labels) < num_gaussians:
                    new_count = num_gaussians - len(self._semantic_labels)
                    new_labels = torch.zeros(new_count, dtype=torch.long, device=device)
                    self._semantic_labels = torch.cat([self._semantic_labels, new_labels])
                else:
                    self._semantic_labels = self._semantic_labels[:num_gaussians]
        else:
            self._semantic_labels = torch.zeros(num_gaussians, dtype=torch.long, device=device)

        # 3. 探索性标记
        if 'is_exploratory' in params:
            self._is_exploratory = params['is_exploratory']
        elif hasattr(self, '_is_exploratory'):
            if len(self._is_exploratory) != num_gaussians:
                if len(self._is_exploratory) < num_gaussians:
                    new_count = num_gaussians - len(self._is_exploratory)
                    new_exploratory = torch.zeros(new_count, dtype=torch.bool, device=device)
                    self._is_exploratory = torch.cat([self._is_exploratory, new_exploratory])
                else:
                    self._is_exploratory = self._is_exploratory[:num_gaussians]
        else:
            self._is_exploratory = torch.zeros(num_gaussians, dtype=torch.bool, device=device)

        # 4. 创建迭代记录
        if 'creation_iter' in params:
            self._creation_iter = params['creation_iter']
        elif hasattr(self, '_creation_iter'):
            if len(self._creation_iter) != num_gaussians:
                if len(self._creation_iter) < num_gaussians:
                    new_count = num_gaussians - len(self._creation_iter)
                    # 使用一个默认值，稍后会在训练循环中更新
                    new_creation = torch.zeros(new_count, dtype=torch.long, device=device)
                    self._creation_iter = torch.cat([self._creation_iter, new_creation])
                else:
                    self._creation_iter = self._creation_iter[:num_gaussians]
        else:
            self._creation_iter = torch.zeros(num_gaussians, dtype=torch.long, device=device)

        # 更新优化器参数引用
        if self.optimizer is not None:
            self._update_optimizer_params()

    def _update_optimizer_params(self):
        """更新优化器中的参数引用"""
        param_mapping = {
            'xyz': self._xyz,
            'f_dc': self._features_dc,
            'f_rest': self._features_rest,
            'opacity': self._opacity,
            'scaling': self._scaling,
            'rotation': self._rotation
        }

        for group in self.optimizer.param_groups:
            name = group['name']
            if name in param_mapping:
                group['params'][0] = param_mapping[name]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """累积梯度统计"""
        # 计算梯度
        grad = viewspace_point_tensor.grad[update_filter, :2]
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)

        # 累积
        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1

    def _prune_optimizer_state(self, mask):
        """清理被剪枝高斯的优化器状态"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            old_tensor = group["params"][0]
            if old_tensor.shape[0] == len(mask):
                # 创建新的参数
                new_tensor = old_tensor[mask]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

                # 清理旧的优化器状态
                old_state = self.optimizer.state.pop(old_tensor, None)
                if old_state is not None:
                    # 为新参数创建对应的状态
                    new_state = {}
                    for key, value in old_state.items():
                        if isinstance(value, torch.Tensor) and value.shape[0] == len(mask):
                            new_state[key] = value[mask]
                        else:
                            new_state[key] = value
                    self.optimizer.state[group["params"][0]] = new_state

        return optimizable_tensors

    def compress_optimizer_state(self):
        """压缩优化器状态，删除死引用"""
        if self.optimizer is None:
            return

        # 收集当前活跃的参数
        active_params = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                active_params.add(p)

        # 删除死状态
        dead_states = []
        for param in list(self.optimizer.state.keys()):
            if param not in active_params:
                dead_states.append(param)

        for param in dead_states:
            del self.optimizer.state[param]

        if dead_states:
            print(f"[Optimizer] Cleaned {len(dead_states)} dead states")
            torch.cuda.empty_cache()