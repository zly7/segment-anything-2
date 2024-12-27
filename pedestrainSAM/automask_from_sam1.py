# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ProcessPoolExecutor
import os
import time
from PIL import Image
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import yaml
from sam2.build_sam import build_sam2_for_self_train
from sam2.utils.transforms import SAM2Transforms
from .trainer_ddp import PedestrainSAM2
from PIL import Image, ImageDraw, ImageFont
from sam2.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    mask_to_rle_pytorch_one_by_one,
)
from loguru import logger

class PedestrainSamAutomaticMaskGenerator:
    def __init__(
        self,
        model: PedestrainSAM2,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        all_use_same_point_grid: bool = True, # 就是比如第一层和第二层都是使用[64*64]
        output_mode: str = "binary_mask",
        person_probability_thresh: float = 0.7,
        vis: bool = False,
        vis_detailed_process_prabability=0.1,
        replaced_immediate_path = "images",
        vis_immediate_folder_to_replace_images = "pred_images",
        vis_resize_width = -1,
        loguru_path: str = "./",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
            length_for_points_grid = len(self.point_grids)
            if all_use_same_point_grid is True:
                self.point_grids = [self.point_grids[0] for _ in range(length_for_points_grid)]
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = model
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.person_probability_thresh = person_probability_thresh
        self.vis = vis
        self.vis_detailed_process_prabability = vis_detailed_process_prabability
        self.replaced_immediate_path = replaced_immediate_path
        self.vis_immediate_folder_to_replace_images = vis_immediate_folder_to_replace_images
        self.vis_resize_width = vis_resize_width
        logger.add(loguru_path, rotation="1 day", retention="7 days", level="DEBUG")

    @torch.no_grad()
    def generate(self, image: np.ndarray, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image,image_path)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        
        # Collect masks for combining
        if self.vis:
            # Convert the original image to PIL Image
            original_image = Image.fromarray(image)
            # Collect all masks as a list
            all_masks = [rle_to_mask(rle) for rle in mask_data["rles"]]
            # Combine all masks into one image
            self.combine_masks(all_masks, original_image, image_path)

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                "person_probability": mask_data["person_probability"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray, image_path: Optional[str]) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size,image_path)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        layer_idx: int,
        orig_size: Tuple[int, ...],
        image_path: Optional[str],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[layer_idx] * points_scale

        # Compute embeddings for all points in points_for_image
        in_points = torch.as_tensor(points_for_image[:, None, :], device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[:2], dtype=torch.int, device=self.predictor.device)
        # Normalize points
        transformed_points = self.predictor._transforms.transform_coords(
            in_points, normalize=True, orig_hw=cropped_im_size
        )

        # Compute prompt embeddings for all points
        concat_points = (transformed_points, in_labels)
        sparse_embeddings, dense_embeddings = self.predictor.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        )

        # Now we can pass the embeddings to _process_batch
        batch_size = self.points_per_batch
        num_points = in_points.shape[0]
        data = MaskData()
        for i in range(0, num_points, batch_size):
            batch_points = in_points[i:i+batch_size]
            batch_transformed_points = transformed_points[i:i+batch_size]
            batch_labels = in_labels[i:i+batch_size]
            batch_sparse_embeddings = sparse_embeddings[i:i+batch_size]
            batch_dense_embeddings = dense_embeddings[i:i+batch_size]
            batch_data = self._process_batch(
                batch_points,
                batch_transformed_points,
                batch_labels,
                batch_sparse_embeddings,
                batch_dense_embeddings,
                cropped_im_size,
                crop_box,
                orig_size,
                cropped_image=cropped_im if self.vis else None,
                image_path=image_path if self.vis else None,
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch(
        self,
        points: torch.Tensor,
        transformed_points: torch.Tensor,
        labels: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        cropped_image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
    ) -> MaskData:
        '''
        默认返回是0/1的
        '''
        orig_h, orig_w = orig_size

        # Run model on this batch
        prd_masks, iou_preds, person_logits, low_res_masks = self.predictor.predict_torch(
            point_coords=transformed_points[:, None, :],
            point_labels=labels[:, None],
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            predict_logit=True,
        )        
        sigmoid_person_logits = torch.sigmoid(person_logits.flatten(0, 1))
        data = MaskData(
            masks=prd_masks.flatten(0, 1), # 融合前两维度
            iou_preds=iou_preds.flatten(0, 1),
            person_probability=sigmoid_person_logits,  # Store person_logits
            points=points.repeat_interleave(prd_masks.shape[1], dim=0),
        )
        del prd_masks
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor._transforms.mask_threshold, self.stability_score_offset
        )
        # Importantly, we threshold the masks here
        data["masks"] = data["masks"] > self.predictor._transforms.mask_threshold
        # Filter by person logits
        keep_person = data["person_probability"] >= self.person_probability_thresh
        data.filter(keep_person)
        logger.info(f"经过人概率大于{self.person_probability_thresh}筛选,剩下{len(data["masks"])}个")
        # Filter by predicted IoU
        keep_mask = data["iou_preds"] > self.pred_iou_thresh
        data.filter(keep_mask)
        logger.info(f"经过预测分数大于{self.pred_iou_thresh}筛选,剩下{len(data["masks"])}个")
        # Filter stability score
        keep_mask = data["stability_score"] >= self.stability_score_thresh
        data.filter(keep_mask)
        logger.info(f"经过稳定性大于{self.stability_score_thresh}筛选,剩下{len(data["masks"])}个")
        data["boxes"] = batched_mask_to_box(data["masks"])
        if self.vis and np.random.rand() < self.vis_detailed_process_prabability:
            start_time = time.time()
            self.visualize_masks(
                data["masks"],
                data["iou_preds"],
                data["person_probability"],
                data["stability_score"],  # Pass stability_score here
                cropped_image,
                image_path,
            )
            logger.info(f"可视化耗时: {time.time() - start_time:.2f}s")
        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch_one_by_one(data["masks"])

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
    
    @staticmethod
    def overlay_mask_on_image(
        image: Image.Image,
        mask: np.ndarray,
        iou_pred: float,
        person_logit: float,
        stability_score: float,  # New parameter
        font: ImageFont.ImageFont
    ) -> Image.Image:
        # Create a color mask with transparency
        color = tuple(np.random.randint(0, 256, size=3).tolist()) + (128,)
        colored_mask = Image.new("RGBA", image.size, color)
        
        # Convert mask to PIL Image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
        mask_pil = mask_pil.convert("L")
        colored_mask.putalpha(mask_pil)

        # Composite the mask onto the image
        overlay = Image.alpha_composite(image.convert("RGBA"), colored_mask)
        
        # Draw text
        draw = ImageDraw.Draw(overlay)
        text_line1 = f"iou_pred: {iou_pred:.2f}, person_logit: {person_logit:.2f}"
        text_line2 = f"stability_score: {stability_score:.2f}"
        
        x, y = 10, 10
        line_spacing = 20
        draw.text((x, y), text_line1, fill=(255, 0, 0, 255), font=font)
        draw.text((x, y + line_spacing), text_line2, fill=(255, 0, 0, 255), font=font)
        
        return overlay

    # Define a helper function to process each mask in parallel
    def process_mask(self, idx, mask, iou_pred, person_logit, stability_score, image, font, save_dir, person_probability_thresh):
        overlay = self.overlay_mask_on_image(image, mask, iou_pred, person_logit, stability_score, font)
        immediate_dir = "person" if person_logit > person_probability_thresh else "others"
        save_path = os.path.join(save_dir, immediate_dir, f"{idx}.jpg")
        overlay = overlay.convert("RGB")
        overlay.save(save_path)

        # Save the mask image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
        mask_pil = mask_pil.convert("L")
        mask_save_path = os.path.join(save_dir, immediate_dir, f"{idx}_mask.jpg")
        mask_pil.save(mask_save_path)
        
    def visualize_masks(
        self,
        masks: torch.Tensor,
        iou_preds: torch.Tensor,
        person_logits: torch.Tensor,
        stability_scores: torch.Tensor,  # Assuming stability_score is per mask
        cropped_image: np.ndarray,
        image_path: str,
    ) -> None:
        # Convert cropped_image to PIL Image
        image = Image.fromarray(cropped_image)
        if self.vis_resize_width > 1: # resize 0.05s 很快
            new_width = self.vis_resize_width
            new_height = int(self.vis_resize_width * image.size[1] / image.size[0])
            image = image.resize((new_width, new_height), resample=Image.LANCZOS)

            # Resize masks accordingly
            masks_resized = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(),  # Add channel dimension
                size=(new_height, new_width),
                mode='nearest'
            ).squeeze(1).bool()  # Remove channel dimension and convert back to bool
        else:
            masks_resized = masks.bool()
        
        font_path = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"
        font = ImageFont.truetype(font_path, 20)

        # Get save directory and base filename
        save_dir, base_filename = self.get_save_dir_and_base_filename(image_path)
        os.makedirs(os.path.join(save_dir, "person"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "others"), exist_ok=True)

        for idx in range(masks_resized.shape[0]): # 这个一次循环0.2s,实测并行处理会更慢
            mask = masks_resized[idx].cpu().numpy()
            iou_pred = iou_preds[idx].item()
            person_logit = person_logits[idx].item()
            stability_score = stability_scores[idx].item()  # Get stability_score for this mask

            # Overlay mask on image with stability_score
            overlay = self.overlay_mask_on_image(image, mask, iou_pred, person_logit, stability_score, font)
            immediate_dir = "person" if person_logit > self.person_probability_thresh else "others"
            save_path = os.path.join(save_dir, immediate_dir, f"{idx}.jpg")
            overlay = overlay.convert("RGB")
            overlay.save(save_path)  
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
            mask_pil = mask_pil.convert("L")
            # Save the mask image
            mask_save_path = os.path.join(save_dir, immediate_dir, f"{idx}_mask.jpg")
            mask_pil.save(mask_save_path) 

    
    def combine_masks(
        self,
        masks: List[np.ndarray],
        image: Image.Image,
        image_path: str,
    ) -> None:
        # Create a new image for the combined mask
        combined_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        font_path = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"
        font = ImageFont.truetype(font_path, 20)

        for mask_np in masks:
            color = tuple(np.random.randint(0, 256, size=3).tolist()) + (128,)
            colored_mask = Image.new("RGBA", image.size, color)
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
            mask_pil = mask_pil.convert("L")
            colored_mask.putalpha(mask_pil)
            combined_mask = Image.alpha_composite(combined_mask, colored_mask)

        # Overlay the combined mask onto the original image
        overlay = Image.alpha_composite(image.convert("RGBA"), combined_mask)
        # Convert to RGB before saving
        overlay = overlay.convert("RGB")

        # Save the combined image
        save_dir, _ = self.get_save_dir_and_base_filename(image_path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "all.jpg")
        overlay.save(save_path)
    
    def get_save_dir_and_base_filename(self, image_path: str) -> Tuple[str, str]:
        image_dir, image_filename = os.path.split(image_path)
        image_base, _ = os.path.splitext(image_filename)
        # Replace 'images' with 'pred_images' in image_dir
        save_dir = image_dir.replace(self.replaced_immediate_path, self.vis_immediate_folder_to_replace_images)
        save_dir = os.path.join(save_dir, image_base)
        return save_dir, image_base

    
'''
main function 主要是为了测试用的
'''
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

    # Initialize the PedestrainSAM2AutomaticMaskGenerator
    mask_generator = PedestrainSamAutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_batch = 64*64, # 64*64大约是38GB
        points_per_side=32, # 每一条边有多少的控制点
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        vis=True,
    )
    image_path = "/data2/zly/CrowdSeg/OCHuman/images/000008.jpg"
    image = Image.open(image_path)
    image_array = np.array(image)
    # Generate masks
    masks = mask_generator.generate(image_array, image_path=image_path)
    
