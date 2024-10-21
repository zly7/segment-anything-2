from PIL import Image
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from typing import Any, Dict, List, Optional, Tuple

import yaml
# import sys
# sys.path.append("/home/zly/multi_ob/segment-anything-2")
from sam2.build_sam import build_sam2_for_self_train
from sam2.utils.transforms import SAM2Transforms
from trainer_ddp import PedestrainSAM2
from sam2.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_masks,
    uncrop_points,
    remove_small_regions,
)


import numpy as np
import torch
from torchvision.ops.boxes import batched_nms  # type: ignore
from typing import Any, Dict, List, Optional, Tuple

class PedestrianSAM2AutomaticMaskGenerator:
    def __init__(
        self,
        model: PedestrainSAM2,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        person_score_thresh: float = 0.5,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ):
        """
        Simplified automatic mask generator for pedestrian segmentation using PedestrainSAM2.

        Arguments:
            model (PedestrainSAM2): The PedestrainSAM2 model to use for mask prediction.
            points_per_side (int): The number of points to be sampled
                along one side of the image. The total number of points is
                points_per_side**2.
            points_per_batch (int): Sets the number of points run simultaneously
                by the model. Higher numbers may be faster but use more GPU memory.
            person_score_thresh (float): Threshold for person classification score.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize
                the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculating the stability score.
            mask_threshold (float): Threshold for binarizing the mask logits.
            box_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks.
            min_mask_region_area (int): If >0, postprocessing will be applied
                to remove disconnected regions and holes in masks with area smaller
                than min_mask_region_area. Requires opencv.
            output_mode (str): The form masks are returned in. Can be 'binary_mask',
                'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
        """
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.model = model
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.person_score_thresh = person_score_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        # Build the point grid for the entire image
        self.point_grid = self._build_point_grid(points_per_side)

    def _build_point_grid(self, points_per_side: int) -> np.ndarray:
        """
        Builds a grid of points over the image.

        Arguments:
            points_per_side (int): Number of points per side.

        Returns:
            np.ndarray: Array of point coordinates.
        """
        coords = np.linspace(0, 1, points_per_side)
        grid = np.meshgrid(coords, coords)
        point_grid = np.stack(grid, axis=-1).reshape(-1, 2)
        return point_grid

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
            image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
            List[Dict[str, Any]]: A list over records for masks.
        """
        orig_size = image.shape[:2]
        self.model._orig_hw = [orig_size]  # Set the original image size

        # Preprocess the image
        image_tensor = self.model._transforms(image).to(self.model.device).unsqueeze(0)  # Shape: (1, 3, H, W)

        # Build point coordinates
        points_scale = np.array(orig_size)[::-1][None, :]  # (1, 2)
        points_for_image = self.point_grid * points_scale  # Shape: (N, 2)
        point_labels = np.ones((points_for_image.shape[0],), dtype=np.int32)  # Positive labels

        # Process points in batches
        data = MaskData()
        num_points = points_for_image.shape[0]
        for i in range(0, num_points, self.points_per_batch):
            points_batch = points_for_image[i:i+self.points_per_batch]
            labels_batch = point_labels[i:i+self.points_per_batch]

            # Call forward method
            prd_masks, person_logits = self.model.forward(
                image_tensor,
                points_batch,
                labels_batch,
            )
            # Process outputs
            masks = prd_masks.squeeze(1)  # Shape: (B, H, W)
            person_scores = torch.sigmoid(person_logits).squeeze(1)  # Shape: (B,)
            points = torch.tensor(points_batch, device=self.model.device)  # (B, 2)

            # Filter out non-person masks
            keep_mask = person_scores > self.person_score_thresh
            if not torch.any(keep_mask):
                continue

            # Apply filters
            masks = masks[keep_mask]
            person_scores = person_scores[keep_mask]
            points = points[keep_mask]

            # Threshold masks and calculate boxes
            masks = masks > self.mask_threshold
            boxes = batched_mask_to_box(masks)

            # Calculate stability score
            stability_score = calculate_stability_score(
                masks, self.mask_threshold, self.stability_score_offset
            )

            # Filter by stability score
            if self.stability_score_thresh > 0.0:
                keep = stability_score >= self.stability_score_thresh
                masks = masks[keep]
                person_scores = person_scores[keep]
                boxes = boxes[keep]
                points = points[keep]
                stability_score = stability_score[keep]

            # Apply min_mask_region_area post-processing
            if self.min_mask_region_area > 0:
                new_masks = []
                new_boxes = []
                new_person_scores = []
                new_points = []
                new_stability_score = []
                for idx in range(masks.shape[0]):
                    mask = masks[idx].cpu().numpy()
                    mask, _ = remove_small_regions(
                        mask, self.min_mask_region_area, mode="holes"
                    )
                    mask, _ = remove_small_regions(
                        mask, self.min_mask_region_area, mode="islands"
                    )
                    if mask.sum() > 0:
                        new_masks.append(torch.from_numpy(mask).to(self.model.device))
                        new_boxes.append(boxes[idx])
                        new_person_scores.append(person_scores[idx])
                        new_points.append(points[idx])
                        new_stability_score.append(stability_score[idx])

                if len(new_masks) == 0:
                    continue

                masks = torch.stack(new_masks, dim=0)
                boxes = torch.stack(new_boxes, dim=0)
                person_scores = torch.stack(new_person_scores, dim=0)
                points = torch.stack(new_points, dim=0)
                stability_score = torch.stack(new_stability_score, dim=0)

            # Convert masks to RLE
            rles = mask_to_rle_pytorch(masks)

            # Prepare MaskData
            batch_data = MaskData(
                masks=masks,
                boxes=boxes,
                scores=person_scores,
                points=points,
                stability_score=stability_score,
                rles=rles,
            )

            data.cat(batch_data)
            del batch_data

        # Remove duplicate masks using NMS
        if len(data["boxes"]) > 0:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["scores"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()

        # Prepare final annotations
        anns = []
        for idx in range(len(data["rles"])):
            ann = {
                "segmentation": data["rles"][idx],
                "area": area_from_rle(data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
                "score": data["scores"][idx].item(),
                "point_coords": data["points"][idx].tolist(),
                "stability_score": data["stability_score"][idx].item(),
            }
            anns.append(ann)

        return anns

if __name__ == "__main__":
    config_path = "./train_config/train_large.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    device_index = 0
    sam2_model, missing_keys, unexpected_keys = build_sam2_for_self_train(
        model_config_path_for_build_sam2,
        sam2_checkpoint,
        device=torch.device('cuda', device_index),
        apply_postprocessing=True,
    )
    # Initialize the PedestrainSAM2 model
    pedestrian_sam2 = PedestrainSAM2(
        model=sam2_model,
        config=config,
        device_index=device_index,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )

    # Initialize the PedestrianSAM2AutomaticMaskGenerator
    mask_generator = PedestrianSAM2AutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_side=32,
        points_per_batch=64,
        person_score_thresh=0.5,
        stability_score_thresh=0.95,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        min_mask_region_area=0,
    )
    image = Image.open("/data2/zly/CrowdSeg/OCHuman/images/000009.jpg")
    image_array = np.array(image)
    # Generate masks
    masks = mask_generator.generate(image_array)