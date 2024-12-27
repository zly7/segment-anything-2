import math
import os
import random
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pycocotools.mask as rletools
import cv2
from loguru import logger
from sam2.utils.transforms import SAM2Transforms
from scipy.ndimage import binary_erosion, binary_dilation
import psutil
from .transform_zly import (
    ComposeAPI,
    RandomHorizontalFlip,
    RandomResizeAPI,
    Pad,
    ToTensorAPI,
    NormalizeAPI,
    RandomGrayscale,
    ColorJitter,
    RandomAffine
)

class COCOPersonDataset(Dataset):
    def __init__(self, annotation_file, images_dir, use_SAM2_transform=True, augment=False,
                 img_output_size=1024, max_length_for_validate=-1, enable_negative_sample=True):
        """
        Initializes the COCO Person Dataset with memory optimization.
        """
        self.annotation_file = annotation_file
        self.images_dir = images_dir
        self.use_SAM2_transform = use_SAM2_transform
        self.augment = augment
        self.img_output_size = img_output_size
        self.max_length_for_validate = max_length_for_validate
        self.enable_negative_sample = enable_negative_sample
        self.whether_use_nonperson = False

        # Initialize transforms
        self.sam2_transform = SAM2Transforms(resolution=img_output_size, mask_threshold=0)
        
        # 优化的数据结构
        self.annotations = []  # [(img_path, ann)]格式
        self.image_to_anns = {}  # {img_path: [ann]}格式，用于负样本生成
        self.nonperson_masks = {}  # {img_path: ann}格式，仅存储非人物标注
        
        self._log_memory("Before loading data")
        self.load_data()
        self._setup_transforms()
        self._log_memory("After loading data")

    def _log_memory(self, message=""):
        """内存使用监控"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"{message} - Memory RSS: {memory_info.rss / 1024 / 1024:.2f} MB")

    def _setup_transforms(self):
        """设置数据转换管道"""
        if self.augment:
            self.transform_pipeline = ComposeAPI([
                RandomHorizontalFlip(p=0.5, consistent_transform=True),
                RandomAffine(p=0.5, degrees=25, shear=20, image_interpolation='bilinear', consistent_transform=True),
                RandomResizeAPI(sizes=[self.img_output_size], square=True, consistent_transform=True),
                ColorJitter(brightness=0.1, contrast=0.03, saturation=0.03, hue=None, consistent_transform=True),
                RandomGrayscale(p=0.05, consistent_transform=True),
                ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=None, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_pipeline = ComposeAPI([
                RandomResizeAPI(sizes=[self.img_output_size], square=True, consistent_transform=True),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_data(self):
        """优化的数据加载函数"""
        logger.info("Starting to load data...")
        
        with open(self.annotation_file, 'r') as f:
            coco = json.load(f)

        # 获取类别信息
        categories = {category['id']: category['name'] for category in coco['categories']}
        person_category_ids = [cat_id for cat_id, name in categories.items() if name.lower() == 'person']
        
        if not person_category_ids:
            raise ValueError("No 'person' category found in the annotation file.")
        
        if 'nonperson' in categories.values():
            nonperson_category_id = [cat_id for cat_id, name in categories.items() 
                                   if name.lower() == 'nonperson'][0]
            self.whether_use_nonperson = True

        # 创建图片ID到路径的映射
        id_to_path = {}
        for image in coco['images']:
            img_path = os.path.join(self.images_dir, image['file_name'])
            if os.path.exists(img_path):
                id_to_path[image['id']] = img_path

        # 处理标注信息
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in id_to_path:
                continue

            img_path = id_to_path[img_id]
            if ann['category_id'] in person_category_ids:
                self.annotations.append((img_path, ann))
                if img_path not in self.image_to_anns:
                    self.image_to_anns[img_path] = []
                self.image_to_anns[img_path].append(ann)
            elif self.whether_use_nonperson and ann['category_id'] == nonperson_category_id:
                self.nonperson_masks[img_path] = ann

        # 清理不必要的数据
        del coco
        
        logger.success(f"Successfully loaded {len(self.annotations)} positive samples from {len(id_to_path)} images.")

        if self.max_length_for_validate is not None and self.max_length_for_validate > 0:
            random.shuffle(self.annotations)
            self.annotations = self.annotations[:self.max_length_for_validate]
            logger.info(f"Using {len(self.annotations)} samples for validation.")

    def __len__(self):
        return len(self.annotations)

    # def __getitem__(self, idx):
    #     """获取数据样本"""
    #     start_time = time.time()
    #     max_retries = 10
    #     for attempt in range(1, max_retries + 1):
    #         if self.enable_negative_sample and (int(time.time() * 1000) % 2) == 0:
    #             sample = self.get_negative_sample(idx)
    #         else:
    #             sample = self.get_positive_sample(idx)
                
    #         h, w = self.img_output_size, self.img_output_size
    #         click_point = sample['click_point'].squeeze(0).numpy()
    #         x, y = click_point
            
    #         if 0 <= x < w and 0 <= y < h:
    #             break
    #         else:
    #             if attempt > 3:
    #                 logger.warning(f"Attempt {attempt}: click_point ({x}, {y}) out of image bounds ({w}, {h}). Resampling.")
                    
    #     if attempt > 9:
    #         logger.error(f"Failed to obtain a valid sample after {max_retries} attempts for index {idx}.")
            
    #     spend_time = time.time() - start_time
    #     if spend_time > 1.0:
    #         logger.debug(f"COCOPerson dataset, Time taken for sample {idx}: {spend_time:.2f} seconds.")
            
    #     return sample
    def __getitem__(self, idx):
        # debug only positive sample
        sample = self.get_positive_sample(idx)
        return sample

    def get_positive_sample(self, idx):
        """获取正样本"""
        img_path, ann = self.annotations[idx]
        start_time = time.time()
        img = Image.open(img_path)
        spend_time = time.time() - start_time
        if spend_time > 1.0:
            logger.debug(f"Time taken for reading image idx: {idx}, image path {img_path}: {spend_time:.2f} seconds.")
        start_time = time.time()
        image = np.asarray(img.convert("RGB"))
        spend_time = time.time() - start_time
        if spend_time > 1.0:
            logger.debug(f"Time taken for converting image idx: {idx}, image path {img_path}: {spend_time:.2f} seconds.")
        start_time = time.time()
        binary_mask = self.decode_mask(ann)
        spend_time = time.time() - start_time
        if spend_time > 1.0:
            logger.debug(f"Time taken for decoding mask idx: {idx}, image path {img_path}: {spend_time:.2f} seconds.")
        eroded_mask = binary_erosion(binary_mask, structure=np.ones((3, 3)))
        ys, xs = np.where(eroded_mask)
        
        if len(ys) == 0:
            logger.warning(f"No pixels found in mask for index {idx} in image {img_path}. the ann is {ann}")
            # return self.get_negative_sample(idx)
            return self.get_positive_sample(idx + 1)
            
        random_idx = random.randint(0, len(ys) - 1)
        click_point = (xs[random_idx], ys[random_idx])

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': binary_mask,
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([1.0]),
            'track_id': int(ann["id"]),
            'point_label': torch.tensor([1.0]),
        }

        return self.process_sample(sample)

    def get_negative_sample(self, idx):
        """获取负样本"""
        img_path = self.annotations[idx % len(self.annotations)][0]
        start_time = time.time()
        img = Image.open(img_path)
        spend_time = time.time() - start_time
        if spend_time > 1.0:
            logger.debug(f"Time taken for reading image idx: {idx}, image path {img_path} in negative mask: {spend_time:.2f} seconds.")
        start_time = time.time()
        image = np.asarray(img.convert("RGB"))
        spend_time = time.time() - start_time
        if spend_time > 1.0:
            logger.debug(f"Time taken for converting image idx: {idx}, image path {img_path} in negative mask: {spend_time:.2f} seconds.")

        # 获取非人物区域
        if img_path in self.nonperson_masks:
            non_person_mask = self.decode_mask(self.nonperson_masks[img_path])
            non_person_indices = np.where(non_person_mask)
        else:
            if self.whether_use_nonperson:
                logger.warning(f"For the nonperson dataset, No nonperson mask found for image {img_path}. Using the combination of negative mask!")
            
            # 生成负样本区域
            start_time = time.time()
            masks_of_person = []
            if img_path in self.image_to_anns:
                for ann in self.image_to_anns[img_path]:
                    mask = self.decode_mask(ann)
                    masks_of_person.append(mask)
            spend_time = time.time() - start_time
            if spend_time > 1.0:
                logger.debug(f"Time taken for decoding masks idx: {idx} in negative mask: {spend_time:.2f} seconds.")

            if masks_of_person:
                combined_person_mask = np.any(masks_of_person, axis=0)
                combined_person_dilation = binary_dilation(combined_person_mask, structure=np.ones((3, 3)))
                non_person_indices = np.where(~combined_person_dilation)
            else:
                combined_exclude_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                non_person_indices = np.where(~combined_exclude_mask)

        if non_person_indices[0].size == 0:
            logger.warning(f"No non-person pixels found in image {img_path}. Using no dilation")
            combined_exclude_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            non_person_indices = np.where(~combined_exclude_mask)
            if non_person_indices[0].size == 0:
                return self.get_positive_sample(idx)

        random_idx = random.randint(0, len(non_person_indices[0]) - 1)
        click_point = (math.floor(non_person_indices[1][random_idx]), 
                      math.floor(non_person_indices[0][random_idx]))

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([0.0]),
            'track_id': -1,
            'point_label': torch.tensor([1.0]),
        }

        return self.process_sample(sample)

    def decode_mask(self, ann):
        """解码mask"""
        if 'segmentation' not in ann:
            raise KeyError("Annotation does not contain 'segmentation' field.")

        if isinstance(ann['segmentation'], list):
            rles = rletools.frPyObjects(ann['segmentation'], ann['height'], ann['width'])
            binary_mask = rletools.decode(rles)
        elif isinstance(ann['segmentation'], dict):
            binary_mask = rletools.decode(ann['segmentation'])
        else:
            raise ValueError("Unknown segmentation format.")

        if binary_mask.ndim == 3:
            binary_mask = binary_mask[:, :, 0]
        return binary_mask.astype(bool)

    def process_sample(self, sample):
        """应用转换"""
        return self.transform_pipeline(sample)

    def __del__(self):
        """清理资源"""
        self.annotations.clear()
        self.image_to_anns.clear()
        self.nonperson_masks.clear()