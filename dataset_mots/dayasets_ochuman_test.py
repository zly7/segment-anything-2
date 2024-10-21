import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
class OCHumanSegmentationTest(Dataset):
    def __init__(self, ann_file, img_dir, transforms=None):
        """
        Args:
            ann_file (str): Path to the COCO format annotation file.
            img_dir (str): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image (PIL Image): The image corresponding to the index.
            img_path (str): The file path to the image.
            rle_masks (list): List of compressed RLE masks for each instance.
        """
        # Get image ID and metadata
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Collect compressed RLE masks
        rle_masks = []
        for ann in anns:
            segm = ann['segmentation']
            if isinstance(segm, list):
                # Polygon - convert to RLE
                rles = maskUtils.frPyObjects(segm, img_info['height'], img_info['width'])
                rle = maskUtils.merge(rles)
            elif isinstance(segm['counts'], list):
                # Uncompressed RLE - compress it
                rle = maskUtils.frPyObjects(segm, img_info['height'], img_info['width'])
            else:
                # Already compressed RLE
                rle = segm
            rle_masks.append(rle)

        # Apply any transformations
        if self.transforms is not None:
            image = self.transforms(image)
        image_array = np.array(image)
        return image_array, img_path, rle_masks