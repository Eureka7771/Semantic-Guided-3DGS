#!/usr/bin/env python3
"""
稳定高质量GPU版SAM预处理
简化实现，避免尺寸不匹配问题
"""

import os
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import cv2
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import gc


@dataclass
class MaskData:
    """存储单个mask的数据"""
    segmentation: np.ndarray
    area: int
    bbox: Tuple[int, int, int, int]
    score: float
    point: Tuple[int, int]

    def to_dict(self):
        return {
            'area': self.area,
            'bbox': self.bbox,
            'score': self.score,
            'point': self.point,
        }


class StableSAMPreprocessor:
    """稳定的高质量SAM预处理器"""

    def __init__(self,
                 sam_checkpoint: str = "checkpoints/sam/sam_vit_h_4b8939.pth",
                 model_type: str = "vit_h",
                 quality: str = "high"):
        """初始化"""
        print(f"Loading SAM model: {model_type}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quality = quality

        # 加载模型
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.eval()
        if self.device == "cuda":
            self.sam = self.sam.cuda()

        self.predictor = SamPredictor(self.sam)

        # 质量设置
        self.quality_configs = {
            "high": {
                "grid_size": 20,  # 更密集的网格
                "score_thresh": 0.85,
                "min_area": 100,
                "use_multimask": True,
                "refine_masks": True
            },
            "medium": {
                "grid_size": 30,
                "score_thresh": 0.88,
                "min_area": 200,
                "use_multimask": True,
                "refine_masks": False
            },
            "fast": {
                "grid_size": 50,
                "score_thresh": 0.90,
                "min_area": 400,
                "use_multimask": False,
                "refine_masks": False
            }
        }

        self.config = self.quality_configs[quality]
        print(f"✓ SAM initialized on {self.device} with {quality} quality")

    def process_image(self, image_path: str, save_visualization: bool = True) -> Dict:
        """处理单张图像"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            #  添加：自动缩放大图像
            TARGET_WIDTH = 1264
            TARGET_HEIGHT = 832

            if h != TARGET_HEIGHT or w != TARGET_WIDTH:
                print(f"  Resizing from {w}x{h} to {TARGET_WIDTH}x{TARGET_HEIGHT} (matching T&T dataset)")
                img_rgb = cv2.resize(img_rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
                h, w = TARGET_HEIGHT, TARGET_WIDTH

            print(f"Processing {os.path.basename(image_path)} ({w}x{h})...")

            # 设置SAM图像
            self.predictor.set_image(img_rgb)

            # 生成采样点
            grid_size = self.config["grid_size"]
            points = []

            # 规则网格
            for y in range(grid_size // 2, h, grid_size):
                for x in range(grid_size // 2, w, grid_size):
                    points.append([x, y])

            # 高质量模式：添加额外采样点
            if self.quality == "high":
                # 边缘采样
                for i in range(0, w, grid_size * 2):
                    points.extend([[i, 0], [i, h - 1]])
                for i in range(0, h, grid_size * 2):
                    points.extend([[0, i], [w - 1, i]])

                # 中心区域加密
                center_x, center_y = w // 2, h // 2
                for dy in range(-100, 101, 20):
                    for dx in range(-100, 101, 20):
                        nx, ny = center_x + dx, center_y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            points.append([nx, ny])

            points = np.array(points)
            print(f"  Using {len(points)} sample points")

            # 预测masks
            all_masks = []

            for point in tqdm(points, desc="  Generating masks", leave=False):
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([1]),
                    multimask_output=self.config["use_multimask"]
                )

                if self.config["use_multimask"]:
                    # 选择最好的mask
                    best_idx = scores.argmax()
                    mask = masks[best_idx]
                    score = scores[best_idx]
                else:
                    mask = masks[0]
                    score = scores[0]

                # 质量过滤
                if score > self.config["score_thresh"] and mask.sum() > self.config["min_area"]:
                    # 计算bbox
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        bbox = (
                            int(x_indices.min()),
                            int(y_indices.min()),
                            int(x_indices.max() - x_indices.min()),
                            int(y_indices.max() - y_indices.min())
                        )

                        mask_data = MaskData(
                            segmentation=mask,
                            area=int(mask.sum()),
                            bbox=bbox,
                            score=float(score),
                            point=(int(point[0]), int(point[1]))
                        )
                        all_masks.append(mask_data)

            # 后处理：去重
            final_masks = self._postprocess_masks(all_masks, h, w)

            # 创建label map
            label_map = np.zeros((h, w), dtype=np.int32)
            for i, mask_data in enumerate(final_masks):
                label_map[mask_data.segmentation] = i + 1

            print(f"  Found {len(final_masks)} masks after postprocessing")

            # 保存可视化
            if save_visualization and len(final_masks) > 0:
                vis_path = image_path.replace('.jpg', '_sam_vis.jpg').replace('.png', '_sam_vis.png')
                self._save_visualization(img_rgb, final_masks, label_map, vis_path)

            return {
                'masks': final_masks,
                'label_map': label_map,
                'image_size': (h, w),
                'num_masks': len(final_masks)
            }

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _postprocess_masks(self, masks: List[MaskData], h: int, w: int) -> List[MaskData]:
        """后处理masks：去重和优化"""
        if not masks:
            return []

        # 按分数排序
        masks.sort(key=lambda x: x.score, reverse=True)

        # NMS去重
        keep_masks = []
        used_area = np.zeros((h, w), dtype=bool)

        for mask_data in masks:
            # 计算重叠
            intersection = mask_data.segmentation & used_area
            overlap_ratio = intersection.sum() / mask_data.area if mask_data.area > 0 else 0

            # 保留重叠较少的mask
            if overlap_ratio < 0.5:
                # 高质量模式：优化mask边界
                if self.config["refine_masks"]:
                    kernel = np.ones((3, 3), np.uint8)
                    # 先腐蚀再膨胀（开运算），去除噪点
                    refined = cv2.morphologyEx(
                        mask_data.segmentation.astype(np.uint8),
                        cv2.MORPH_OPEN, kernel
                    )
                    # 再膨胀然后腐蚀（闭运算），填充小孔
                    refined = cv2.morphologyEx(
                        refined, cv2.MORPH_CLOSE, kernel
                    )
                    mask_data.segmentation = refined.astype(bool)
                    mask_data.area = int(refined.sum())

                # 再次检查面积
                if mask_data.area > self.config["min_area"]:
                    keep_masks.append(mask_data)
                    used_area |= mask_data.segmentation

                # 限制总数
                if len(keep_masks) >= 500:  # 最多500个masks
                    break

        return keep_masks

    def _save_visualization(self, image: np.ndarray, masks: List[MaskData],
                            label_map: np.ndarray, save_path: str):
        """保存可视化"""
        try:
            h, w = image.shape[:2]

            # 创建彩色mask
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

            # 生成颜色
            np.random.seed(42)
            colors = []
            for i in range(len(masks) + 1):
                color = np.random.randint(60, 255, size=3)
                colors.append(color)
            colors[0] = [0, 0, 0]  # 背景

            # 填充颜色
            for label in range(1, len(masks) + 1):
                colored_mask[label_map == label] = colors[label]

            # 混合
            alpha = 0.4
            vis = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

            # 添加采样点（可选）
            if self.quality == "high" and len(masks) < 100:
                for mask_data in masks:
                    # 画采样点
                    cv2.circle(vis, mask_data.point, 2, (255, 255, 255), -1)
                    # 画边界框
                    x, y, w, h = mask_data.bbox
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

            # 添加统计信息
            text = f"Masks: {len(masks)} | Quality: {self.quality}"
            cv2.putText(vis, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 保存
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_bgr)

        except Exception as e:
            print(f"Warning: Failed to save visualization: {e}")

    def process_dataset(self, dataset_path: str, output_dir: str,
                        image_folder: str = "images", max_images: Optional[int] = None):
        """处理整个数据集"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        # 查找图像
        image_dir = os.path.join(dataset_path, image_folder)
        if not os.path.exists(image_dir):
            for folder in ["images", "rgb", "color"]:
                test_dir = os.path.join(dataset_path, folder)
                if os.path.exists(test_dir):
                    image_dir = test_dir
                    break

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        if max_images:
            image_files = image_files[:max_images]

        print(f"\nProcessing {len(image_files)} images from {image_dir}")
        print(f"Quality mode: {self.quality}")

        # 处理
        all_metadata = {}
        success_count = 0

        for img_file in tqdm(image_files, desc="Processing dataset"):
            image_path = os.path.join(image_dir, img_file)
            image_name = os.path.splitext(img_file)[0]

            result = self.process_image(image_path)

            if result and result['num_masks'] > 0:
                # 保存结果
                label_map_path = os.path.join(masks_dir, f"{image_name}_labels.npy")
                np.save(label_map_path, result['label_map'])

                masks_data_path = os.path.join(masks_dir, f"{image_name}_masks.pkl")
                with open(masks_data_path, 'wb') as f:
                    pickle.dump({
                        'masks': [m.to_dict() for m in result['masks']],
                        'segmentations': [m.segmentation for m in result['masks']]
                    }, f)

                # 移动可视化
                base_name = os.path.splitext(image_path)[0]  # 去掉扩展名
                vis_src = base_name + '_sam_vis.jpg'  # 重新添加扩展名
                if os.path.exists(vis_src):
                    vis_dst = os.path.join(vis_dir, f"{image_name}_vis.jpg")
                    if os.path.exists(vis_dst):
                        os.remove(vis_dst)
                    os.rename(vis_src, vis_dst)

                all_metadata[image_name] = {
                    'num_masks': result['num_masks'],
                    'image_size': result['image_size'],
                    'label_map_path': label_map_path,
                    'masks_data_path': masks_data_path
                }
                success_count += 1

            # 定期清理内存
            if success_count % 10 == 0:
                torch.cuda.empty_cache()

        # 保存元数据
        metadata_path = os.path.join(output_dir, "sam_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'dataset_path': dataset_path,
                'num_images': len(all_metadata),
                'images': all_metadata,
                'preprocessing_config': {
                    'model_type': 'vit_h',
                    'quality': self.quality,
                    'device': self.device,
                    'config': self.config
                }
            }, f, indent=2)

        print(f"\n✓ Preprocessing complete!")
        print(f"  Successfully processed {success_count}/{len(image_files)} images")
        print(f"  Results saved to: {output_dir}")

        return success_count > 0

    def cleanup(self):
        """清理资源"""
        del self.predictor
        del self.sam
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('--quality', type=str, default='high',
                        choices=['high', 'medium', 'fast'])
    parser.add_argument('--checkpoint', type=str,
                        default="checkpoints/sam/sam_vit_h_4b8939.pth")
    parser.add_argument('--max_images', type=int, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.source_path, "sam_preprocessed")

    print(f"\n{'=' * 60}")
    print(f"Stable High-Quality SAM Preprocessing")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.source_path}")
    print(f"Output: {args.output_dir}")
    print(f"Quality: {args.quality}")
    print(f"{'=' * 60}\n")

    # 检查checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: SAM checkpoint not found at {args.checkpoint}")
        print("Please download from: https://github.com/facebookresearch/segment-anything")
        return

    # 初始化
    preprocessor = StableSAMPreprocessor(
        sam_checkpoint=args.checkpoint,
        quality=args.quality
    )

    # 处理
    preprocessor.process_dataset(
        dataset_path=args.source_path,
        output_dir=args.output_dir,
        max_images=args.max_images
    )

    # 清理
    preprocessor.cleanup()

    print("\n✓ All done!")


if __name__ == "__main__":
    main()