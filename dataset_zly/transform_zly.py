# transform.py

import math
import random
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
from typing import List, Tuple, Optional
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import ColorJitter as TorchColorJitter
import torchvision.transforms as T
class RandomHorizontalFlip:
    """
    随机水平翻转图像和掩码，并调整点击点。
    支持多个点击点 (B, 2)。
    """
    def __init__(self, p: float = 0.5, consistent_transform: bool = True):
        self.p = p
        self.consistent_transform = consistent_transform  # 参数保留但不使用

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            image = sample['image']
            mask = sample['mask']
            click_points = sample['click_point']  # (B, 2)

            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

            width = image.shape[1]
            click_points[:, 0] = width - click_points[:, 0] - 1

            sample['image'] = image
            sample['mask'] = mask
            sample['click_point'] = click_points

        return sample


class RandomAffine:
    """
    Randomly apply affine transformations to image and mask, and adjust click points.
    Supports multiple click points (B, 2).
    """
    def __init__(self, p=0.5, degrees: float = 0, shear: float = 0, scale: float = 1.0,
                 translate: Optional[float] = None, image_interpolation: str = 'bilinear',
                 consistent_transform: bool = True):
        self.p = p
        self.degrees = degrees
        self.shear = shear
        self.scale = scale
        self.translate = translate
        self.image_interpolation = image_interpolation
        self.consistent_transform = consistent_transform  # Not used but kept for compatibility

        if image_interpolation == 'bilinear':
            self.image_interp_mode = InterpolationMode.BILINEAR
        elif image_interpolation == 'bicubic':
            self.image_interp_mode = InterpolationMode.BICUBIC
        elif image_interpolation == 'nearest':
            self.image_interp_mode = InterpolationMode.NEAREST
        else:
            raise ValueError(f"Unsupported image_interpolation mode: {image_interpolation}")

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        # Generate random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        scale = random.uniform(self.scale[0], self.scale[1]) if isinstance(self.scale, (tuple, list)) else self.scale

        max_dx = self.translate * sample['image'].shape[1] if self.translate else 0
        max_dy = self.translate * sample['image'].shape[0] if self.translate else 0
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy))) if self.translate else (0, 0)

        image = sample['image']
        mask = sample['mask']
        click_points = sample['click_point']  # (B, 2)

        # Convert images to PIL
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)

        # Compute the center of the image
        center = (image_pil.width * 0.5, image_pil.height * 0.5)

        # Apply affine transformation to image
        image_transformed = F.affine(
            image_pil,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=(shear_x, shear_y),
            interpolation=self.image_interp_mode,
            fill=self._get_fill_color(),
            center=center
        )

        # Apply affine transformation to mask
        mask_transformed = F.affine(
            mask_pil,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=(shear_x, shear_y),
            interpolation=InterpolationMode.NEAREST,
            fill=0,
            center=center
        )

        image = np.array(image_transformed)
        mask = (np.array(mask_transformed) > 127).astype(np.uint8)  # Convert back to binary mask
        # Adjust click points
        affine_matrix = self.get_full_affine_matrix(center, angle, translations, scale, (shear_x, shear_y))
        #Add a column of ones for homogeneous coordinates
        ones = np.ones((click_points.shape[0], 1))
        click_points_homogeneous = np.hstack([click_points, ones])
        transformed_points = click_points_homogeneous @ affine_matrix.T
        sample['image'] = image
        sample['mask'] = mask
        sample['click_point'] = transformed_points[:, :2]

        return sample

    def _get_fill_color(self):
        return (123, 116, 103)

    def get_full_affine_matrix(self, center, angle, translate, scale, shear):
    # center: (center_x, center_y)
    # angle: in degrees
    # translate: (translate_x, translate_y)
    # scale: scalar or tuple (scale_x, scale_y)
    # shear: float or tuple (shear_x, shear_y), in degrees
        angle_rad = math.radians(angle)
        shear_x_rad, shear_y_rad = math.radians(shear[0]), math.radians(shear[1])
        # Translation matrices to move the center to the origin and back
        T_center_inv = np.array([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])

        T_center = np.array([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])

        # Final translation
        T_translate = np.array([
            [1, 0, translate[0]],
            [0, 1, translate[1]],
            [0, 0, 1]
        ])

        # Scaling matrix
        S = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])

        # Shear matrix
        Sh = np.array([
            [1, math.tan(shear_x_rad), 0],
            [math.tan(shear_y_rad), 1, 0],
            [0, 0, 1]
        ])

        # Rotation matrix
        R = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        M = T_translate @ T_center @ R @ Sh @ S @ T_center_inv
        return M


class RandomResizeAPI:
    """
    随机调整图像和掩码的大小，并调整点击点。
    支持多个点击点 (B, 2)。
    """
    def __init__(self, sizes: List[int], square: bool = False, consistent_transform: bool = True):
        self.sizes = sizes  # 可选的目标大小列表,但是时常只有(1024,1024)
        self.square = square
        self.consistent_transform = consistent_transform  # 参数保留但不使用

    def __call__(self, sample: dict) -> dict:
        size = random.choice(self.sizes)
        image = sample['image']
        mask = sample['mask']
        click_points = sample['click_point']  # (B, 2)
        # square 暂时没用
        original_height, original_width = image.shape[:2]
        target_size = size if not self.square else size

        # 计算缩放因子
        scale_y = target_size / original_height
        scale_x = target_size / original_width

        # 调整图像和掩码的大小
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # 调整点击点
        click_points[:, 0] = click_points[:, 0] * scale_x
        click_points[:, 1] = click_points[:, 1] * scale_y

        sample['image'] = image
        sample['mask'] = mask
        sample['click_point'] = click_points

        return sample


class ColorJitter:
    """
    随机改变图像的亮度、对比度、饱和度和色调。
    """
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0,
                 saturation: float = 0.0, hue: Optional[float] = None, consistent_transform: bool = True):
        self.color_jitter = TorchColorJitter(brightness, contrast, saturation, [-hue,hue] if hue is not None else 0)
        self.consistent_transform = consistent_transform  
    
    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        image = Image.fromarray(image)
        image = self.color_jitter(image)
        image = np.array(image)
        sample['image'] = image
        return sample


class RandomGrayscale:
    """
    以一定概率将图像转换为灰度图。
    """
    def __init__(self, p: float = 0.1, consistent_transform: bool = True):
        self.p = p
        self.consistent_transform = consistent_transform  # 参数保留但不使用

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            image = sample['image']
            # 转换为灰度图并保持RGB格式
            image = Image.fromarray(image).convert('L')
            image = ImageOps.colorize(image, black="black", white="white")
            image = np.array(image)
            sample['image'] = image
        return sample


class Pad:
    """
    使用指定的填充参数对图像和掩码进行填充，并调整点击点。
    支持多个点击点 (B, 2)。
    """
    def __init__(self, padding: Tuple[int, int, int, int]):
        """
        Args:
            padding (tuple): (left, top, right, bottom)
        """
        self.padding = padding

    def __call__(self, sample: dict) -> dict:
        left, top, right, bottom = self.padding
        image = sample['image']
        mask = sample['mask']
        click_points = sample['click_point']  # (B, 2)

        # 填充图像
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # 填充掩码
        mask = cv2.copyMakeBorder(
            mask.astype(np.uint8),
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        # 调整点击点
        click_points[:, 0] = click_points[:, 0] + left
        click_points[:, 1] = click_points[:, 1] + top

        sample['image'] = image
        sample['mask'] = mask
        sample['click_point'] = click_points

        return sample


class ToTensorAPI:
    """
    将图像和掩码转换为PyTorch张量，并保持点击点为 (B, 2) 的张量。
    """
    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        mask = sample['mask']
        click_points = sample['click_point']  # (B, 2)

        # 将图像转换为张量并归一化到[0,1]
        image = F.to_tensor(Image.fromarray(image))

        # 将掩码转换为张量
        mask = torch.from_numpy(mask).float()

        # 将点击点转换为张量
        click_points = torch.from_numpy(click_points).float()

        sample['image'] = image
        sample['mask'] = mask
        sample['click_point'] = click_points

        return sample


class NormalizeAPI:
    """
    使用指定的均值和标准差对图像张量进行归一化。
    """
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        image = F.normalize(image, mean=self.mean, std=self.std)
        sample['image'] = image
        return sample


class ComposeAPI:
    """
    将多个转换组合在一起。
    """
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
