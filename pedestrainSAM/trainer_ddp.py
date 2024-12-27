import math
import os
import random
import shutil
import time
import numpy as np
import torchvision
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, ChainedScheduler
from tensorboardX import SummaryWriter
from loguru import logger
from sam2.build_sam import build_sam2, build_sam2_for_self_train
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.amg import calculate_stability_score
from sam2.utils.transforms import SAM2Transforms
from typing import Optional
from dataset_zly import MOTSDataset,COCOPersonDataset, CombinedDataset
from PIL import Image, ImageDraw, ImageFont
from training.loss_fns import sigmoid_focal_loss, dice_loss, iou_loss
from typing import Union
from torchvision import transforms as torch_transforms  # 导入 torchvision 的 transforms 模块
from PIL import Image
import torchvision.transforms as transforms
class PedestrainSAM2(nn.Module):
    def __init__(
        self,
        model: SAM2Base,
        config,
        device_index,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    ):
        super().__init__()
        self.sam2_model = model
        self._transforms = SAM2Transforms(
            resolution=self.sam2_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.config = config
        self.device = torch.device('cuda', device_index)
        self.is_img_set = False
        self.use_low_backbone_feature_direct = config["model"]["use_low_backbone_feature_direct"]

        self.classifier = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 1) # model.fc = nn.Linear(model.fc.in_features, 1)
        # 修改最后的全连接层，输出一个logit表示是否为人

        self.classifier = self.classifier.to(self.device) # don't need half beacause amp autocast
        self.count_for_vis = 0
        self.extracted_images = None
        self.input_image_after_transform = None # for inference
    
    def load_classifier(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        if "classifier" in sd:
            missing_keys, unexpected_keys = self.classifier.load_state_dict(sd["classifier"], strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
            self.classifier.to(self.device)
            logger.success(f"Loaded classifier from {ckpt_path}")
        else:
            logger.warning("Checkpoint is a dictionary but does not contain 'classifier' key.")

    def _extract_and_resize_images(self, image, binary_mask, whether_person:list = None, 
                                   whether_gt:list = None, whether_vis = False, whether_tolerate_no_mask = True, point_coordinate = None):
        """
        提取分割区域并调整图像大小的函数
        image : [b,3,1024,1024] or [b,3,A,B]
        binary_mask : [b, 1 ,1024,1024] or [b,1,256,256] or [b,1,A,B]
        """
        extracted_images = []
        batch_size = image.size(0)
        assert binary_mask.size(0) == batch_size
        assert image.dim() == 4 and binary_mask.dim() == 4
        assert binary_mask.size(1) == 1
        if binary_mask.size(2) == 256 and binary_mask.size(3) == 256:
            binary_mask = F.interpolate(binary_mask, size=(1024, 1024), mode='nearest')
        for i in range(batch_size):
            mask = binary_mask[i]
            img = image[i]
            # 将mask应用到图像上
            masked_img = img * mask
            # 获取mask的边界框
            coords = torch.nonzero(mask[0], as_tuple=False)
            if coords.nelement() == 0:
                resized_img = torch.zeros(3, 224, 224).to(self.device)
                logger.warning(f"Big mistake!!!  No mask found for image {i}")
            else:
                y_min, x_min = coords.min(0)[0]
                y_max, x_max = coords.max(0)[0]
                cropped_img = masked_img[:, y_min:y_max+1, x_min:x_max+1]
                # 不扭曲地调整图像大小到224x224
                resized_img = self._resize_with_padding(cropped_img, 224, 224)
            if whether_vis:
                self.save_image(resized_img, f"./vis_classfication_image/extracted_image_{self.count_for_vis}_gt?_{whether_gt[i]}.png", whether_person[i])
                tensor_filename = f"extracted_image_{self.count_for_vis}_gt?_{whether_gt[i]}.pt"
                tensor_path = os.path.join("./vis_classfication_image", tensor_filename)
                torch.save(resized_img.cpu(), tensor_path)
                self.count_for_vis += 1
            extracted_images.append(resized_img)
        extracted_images = torch.stack(extracted_images)
        return extracted_images

    def _resize_with_padding(self, img, target_height, target_width):
        c, h, w = img.size()
        # 计算放缩比例
        scale = min(target_height / h, target_width / w)
        # 计算放缩后的尺寸
        new_h = math.ceil(h * scale)
        new_w =  math.ceil(w * scale)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        y_offset = max(0, y_offset)
        x_offset = max(0, x_offset)
        resized_img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        # 创建新的图像并填充
        new_img = torch.zeros(c, target_height, target_width).to(img.device)
        y_end = y_offset + new_h
        x_end = x_offset + new_w    
        # 如果计算出的结束位置超出目标尺寸，则进行修正
        if y_end > target_height:
            y_end = target_height
            new_h = y_end - y_offset
        if x_end > target_width:
            x_end = target_width
            new_w = x_end - x_offset
        new_img[:, y_offset:y_end, x_offset:x_end] = resized_img[:, :new_h, :new_w]
        return new_img
    
    def save_image(self, tensor, filename, whether_person=False):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 反归一化 (De-normalize)
        tensor = tensor.clone().detach().to('cpu')
        tensor = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        tensor.clamp_(0, 1)
        image = transforms.ToPILImage()(tensor)
        draw = ImageDraw.Draw(image)
        text = "Person" if whether_person else "No Person"
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()
        x = 10  
        y = 10  
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        image.save(filename)
        
    def _visualize_with_point(self, img, point_coords, gt_info, person_info):
        """
        在图像上绘制点击点并保存
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img.clone().detach().to('cpu')
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img.clamp_(0, 1)
        img = img.permute(1, 2, 0).numpy()  # 将 (C, H, W) 转换为 (H, W, C) 格式
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for point in point_coords:
            x, y = point
            draw.rectangle([x-5, y-5, x+5, y+5], outline="red", width=2, fill="blue")  # 绘制点击点
            draw.text((x+10, y-5), f"GT: {gt_info}, Person: {person_info}", fill="red")  # 添加文字说明
        pil_img.save(f"./vis_classification_image/No_mask_image{self.count_for_vis}.png")
        self.count_for_vis += 1
    
    def forward(self, image:Optional[Union[torch.Tensor, np.ndarray]], point_coords = None , point_labels = None, box = None):
        """
        input:
        image tensor: (B, 3, H, W) (B,3,1024,1024)
        points_coords: (B, N, 2)
        point_labels: (B, N)
        box: (B, 2, 2)
        return: 
        prd_masks: (B, 1/4, H, W)
        person_logits, (B, 1/4)
        iou_predictions, (B, 1/4)
        """
        if not self.config["train"]["transfer_image_in_dataset_use_sam2"]: # This line should be done first
            b, h, w, c = image.shape # 对应测试的时候不一定是已经resize好的image，point_coords是原始的绝对坐标，box是原始的绝对坐标
            assert c == 3
            image = self._transforms.forward_batch_numpy(image)
            assert isinstance(image, torch.Tensor)
            self._orig_hw = [(h, w)]
        else:
            self._orig_hw = [(image.size(2), image.size(3))] # 对应训练集基本都是已经resize好的image,但是point_coords是1024的绝对坐标
        assert image.shape[1] == 3, "Input image must have 3 channels. current image.shape is {}".format(image.shape)
        _features = self._image_encoder(image)
        if self.use_low_backbone_feature_direct:
            img_embed, high_res_features, direct_high_res_features = _features["image_embed"], _features["high_res_feats"], _features["high_res_feats_direct"]
        else:
            img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, None, normalize_coords=True
        )
        if point_coords is not None:
            concat_points = (unnorm_coords, labels)
        else:
            concat_points = None
        # Embed prompts 
        if unnorm_box is not None:
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
            box_labels = box_labels.repeat(unnorm_box.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder( #sparse_embeddings [32,2,256]?
            points=concat_points,
            boxes=None,
            masks=mask_input,
        ) # spare [b,2,256],[6,256,64,64]
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.config["train"]["multimask"],
            repeat_image=False,
            high_res_features=high_res_features,
            direct_high_res_features = direct_high_res_features if self.use_low_backbone_feature_direct else None
        )
        prd_masks = low_res_masks_logits
        binary_mask = (prd_masks > 0.0).float()
        iou_pred_max, max_iou_indices = torch.max(iou_predictions, dim=1)
        binary_mask = binary_mask[range(low_res_masks_logits.shape[0]),max_iou_indices,:,:].unsqueeze(1)
        if not self.sam2_model.training:
            prd_masks = prd_masks[range(low_res_masks_logits.shape[0]),max_iou_indices,:,:].unsqueeze(1)
            iou_predictions = iou_pred_max.unsqueeze(1)
            prd_masks = self._transforms.postprocess_masks(prd_masks, self._orig_hw[-1]) 
        extracted_images = self._extract_and_resize_images(image, binary_mask, point_coordinate=point_coords).detach() # 在训练或者测试的时候就是完全选最好预测的那个
        self.extracted_images = extracted_images
        person_logits = self.classifier(extracted_images)
        
        return prd_masks, person_logits, iou_predictions

    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        if self.use_low_backbone_feature_direct:
            _, vision_feats, _, _, vision_feats_direct = self.sam2_model._prepare_backbone_features_direct(backbone_out)
        else:
            _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed # 只在最后一层加入embedding
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1], "high_res_feats_direct": None}
        if self.use_low_backbone_feature_direct:
            bb_feat_sizes_direct = [(256, 256), (128, 128)]
            direct_feats = [
                feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
                for feat, feat_size in zip(vision_feats_direct[::-1], bb_feat_sizes_direct[::-1])
            ][::-1]
            _features["high_res_feats_direct"] = direct_feats

        return _features

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box
    
    def set_image(self, input_image: Union[Image.Image, np.ndarray]):
        """
        这个输入的image之后是要过Totensor()的
        Prepare the image embeddings for the given image. Only use when inference
        """
        if isinstance(input_image, np.ndarray):
            self._orig_hw = [input_image.shape[:2]]
        elif isinstance(input_image, Image):
            w, h = input_image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")
        input_image = self._transforms(input_image)
        # resize to the origin size
        input_image = input_image[None, ...].to(self.device) # add batch dimension
        self.input_image_after_transform = torch.nn.functional.interpolate(input_image, size=(self._orig_hw[0][0], self._orig_hw[0][1]), mode='bilinear', align_corners=False)
        
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        _features = self._image_encoder(input_image)
        self.img_embed = _features["image_embed"]
        self.high_res_features = _features["high_res_feats"]
        self.direct_high_res_features = _features["high_res_feats_direct"]
        # Compute the dense positional embeddings once
        self.image_pe = self.sam2_model.sam_prompt_encoder.get_dense_pe()
        self.is_img_set = True

    def reset_image(self):
        """
        Clear the stored image and prompt embeddings.
        """
        self.img_embed = None
        self.high_res_features = None
        self.image_pe = None
        self.sparse_embeddings = None
        self.dense_embeddings = None
        self.is_img_set = False
        self.input_image_after_transform = None

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords = None,
        point_labels = None,
        box = None,
        sparse_embeddings=None,
        dense_embeddings=None,
        predict_logit = False,
    ):
        """
        USE_HQ 应该由模型的参数配置决定，不应该由这个函数来决定
        Predict masks for the given prompts.
        prd_masks: (B, 1, H, W)
        iou_predictions: (B, 1)
        person_logits: (B, 1)
        low_res_masks_logits: (B, 1, 256, 256)
        """
        # Use stored embeddings
        img_embed = self.img_embed
        high_res_features = self.high_res_features
        direct_high_res_features = self.direct_high_res_features
        image_pe = self.image_pe

        # If embeddings are provided, use them; otherwise compute them
        if sparse_embeddings is None or dense_embeddings is None:
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords, point_labels, box, None, normalize_coords=True
            )
            if point_coords is not None:
                concat_points = (unnorm_coords, labels)
            else:
                concat_points = None
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=unnorm_box,
                masks=None,
            )

        batch_size = sparse_embeddings.shape[0]
        if batch_size > 1:
            repeat_image = True
        else:
            repeat_image = False

        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (1, 256, 64, 64)
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            direct_high_res_features = direct_high_res_features if self.use_low_backbone_feature_direct else None
        ) 
        low_res_masks_logits = torch.clamp(low_res_masks_logits, -32.0, 32.0)
        iou_pred_max, max_iou_indices = torch.max(iou_predictions, dim=1)
        prd_masks = self._transforms.postprocess_masks(low_res_masks_logits, self._orig_hw[-1]) # 这里如果是原本的图片非常大，这里会消耗很大cuda显存
        prd_masks = prd_masks[range(low_res_masks_logits.shape[0]),max_iou_indices,:,:].unsqueeze(1)
        if predict_logit == False:
            prd_masks = prd_masks > self._transforms.mask_threshold
            
        binary_mask = (prd_masks > 0.0).float()
        batch_size = binary_mask.size(0)
        input_image_expand = self.input_image_after_transform.expand(batch_size, -1, -1, -1)
        extracted_images = self._extract_and_resize_images(input_image_expand, binary_mask).detach()
        self.extracted_images = extracted_images
        person_logits = self.classifier(extracted_images)
        return prd_masks, iou_predictions, person_logits, low_res_masks_logits


class CustomSegLoss(nn.Module):
    def __init__(self, multimask=True):
        super(CustomSegLoss, self).__init__()
        self.multimask = multimask
        
    def forward(self, mask_logits, targets, iou_predictions):
        """
        Args:
            mask_logits: [B, C, H, W] - Predicted mask logits, where C is the number of masks
            targets: [B, H, W] - Ground truth masks
            iou_predictions: [B, C] - Predicted IOU scores (optional)
        Returns:
            total_loss: The combined loss value
            loss_dict: Dictionary containing individual loss components and metrics
        """
        assert mask_logits.dim() == 4, "mask_logits should have 4 dimensions [B, C, H, W]"
        
        B, C, H, W = mask_logits.shape
        
        if self.multimask and targets.dim() == 3:
            # Repeat targets for each mask if multimask is enabled
            targets = targets.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, H, W]
        elif not self.multimask and mask_logits.dim() == 4:
            mask_logits = mask_logits[:, 0, :, :].unsqueeze(1)
            targets = targets.unsqueeze(1)
        
        # Calculate loss for all masks at once
        epsilon = 1e-5
        prd_masks = torch.sigmoid(mask_logits)  # [B, C, H, W]
        
        # Compute segmentation loss for all masks
        seg_loss = -targets * torch.log(prd_masks + epsilon) - \
                  (1 - targets) * torch.log(1 - prd_masks + epsilon)
        seg_loss = seg_loss.mean(dim=(2, 3))  # Mean over H,W dimensions -> [B, C]
        
        # Calculate IoU for all masks
        pred_binary = (prd_masks > 0.5).float()
        inter = (targets * pred_binary).sum(dim=(2, 3))  # [B, C]
        union = targets.sum(dim=(2, 3)) + pred_binary.sum(dim=(2, 3)) - inter  # [B, C]
        ious = inter / (union + epsilon)  # [B, C]
        
        # Select best mask based on segmentation loss
        best_loss_inds = torch.argmin(seg_loss, dim=1)  # [B]
        batch_inds = torch.arange(B, device=seg_loss.device)
        
        # Get losses for best masks
        selected_seg_loss = seg_loss[batch_inds, best_loss_inds].mean()
        selected_iou = ious[batch_inds, best_loss_inds].mean()
        
        score_loss = torch.abs(iou_predictions - ious).mean()
        
        # Total Loss
        total_loss = selected_seg_loss + score_loss * 0.25
        
        # Prepare loss dictionary
        loss_dict = {
            "total_loss": total_loss,
            "seg_loss": selected_seg_loss,
            "focal_loss": selected_seg_loss,  # Keeping naming consistent with original
            "dice_loss": selected_seg_loss,   # Keeping naming consistent with original
            "iou_loss": score_loss * 0.25,
            "real_iou": selected_iou,
            "real_backward_mask_logits_index": best_loss_inds  # Adding index of selected masks
        }
        
        return total_loss, loss_dict

class FocalDiceIoULoss(nn.Module):
    def __init__(self, weight_focal=20.0, weight_dice=1.0, weight_iou=1.0, 
                 focal_alpha=0.25, focal_gamma=2, 
                 iou_use_l1_loss=True,multimask=False):
        super(FocalDiceIoULoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.total_weights = weight_focal + weight_dice + weight_iou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_use_l1_loss = iou_use_l1_loss
        self.multimask = multimask

    def forward(self, mask_logits, targets, iou_predictions=None):
        """
        Args:
            mask_logits: [B, 4 / 1, H, W] - 预测的mask logits,测试的时候大小是1
            targets: [B, H, W] - 真实的mask
            iou_predictions: [B, 4] - 预测的IOU分数（可选）
        Returns:
            总损失
        """
        assert mask_logits.dim() == 4  
        if self.multimask == True and len(targets.size()) == 3: # 兼容老版本的multimask==False
            targets = targets.unsqueeze(1).repeat(1, mask_logits.size(1), 1, 1)
        # Focal Loss
        focal_loss = sigmoid_focal_loss(
            mask_logits,
            targets,
            num_objects=targets.size(0),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=self.multimask
        )
        
        # Dice Loss
        dice_loss_value = dice_loss(
            mask_logits,
            targets,
            num_objects=targets.size(0),
            loss_on_multimask=self.multimask
        )
        
        # IOU Loss
        if iou_predictions is not None:
            if len(targets.size()) == 3:
                targets = targets.unsqueeze(1)
            if len(mask_logits.size()) == 3:
                mask_logits = mask_logits.unsqueeze(1) 
            iou_loss_value = iou_loss(
                mask_logits,
                targets,
                pred_ious=iou_predictions,
                num_objects=targets.size(0),
                loss_on_multimask=self.multimask,
                use_l1_loss=self.iou_use_l1_loss
            )
        else:
            iou_loss_value = torch.tensor(0.0, device=mask_logits.device)
        if self.multimask == True:
            assert focal_loss.dim() == 2
            assert dice_loss_value.dim() == 2
            assert iou_loss_value.dim() == 2
            loss_combo = (
                focal_loss * self.weight_focal
                + dice_loss_value * self.weight_dice
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = focal_loss[batch_inds, best_loss_inds].unsqueeze(1).sum() # sum的原因是所有的实现都除以了num_objects
            loss_dice = dice_loss_value[batch_inds, best_loss_inds].unsqueeze(1).sum()
            loss_iou = iou_loss_value.mean(dim=-1).unsqueeze(1).sum()
            total_loss = (self.weight_focal * loss_mask) + \
                        (self.weight_dice *loss_dice) + \
                        (self.weight_iou * loss_iou)
            pred_binary = (mask_logits[batch_inds, best_loss_inds] > 0.0).float()
            targets_3_dim = targets[:, 0, :, :]
            inter = (targets_3_dim * pred_binary).sum(dim=(1, 2))
            union = targets_3_dim.sum(dim=(1, 2)) + pred_binary.sum(dim=(1, 2)) - inter
            iou = inter / (union + 1e-5)
            real_iou = iou.mean()
            loss_dict = {
                "total_loss": total_loss,
                "focal_loss": loss_mask,
                "dice_loss": loss_dice,
                "iou_loss": loss_iou,
                "real_backward_mask_logits_index": best_loss_inds,
                "real_iou": real_iou
            }
   
        else:
            # 加权总损失
            total_loss = (self.weight_focal * focal_loss) + \
                        (self.weight_dice * dice_loss_value) + \
                        (self.weight_iou * iou_loss_value)
            loss_dict = {
                "total_loss": total_loss,
                "focal_loss": focal_loss,
                "dice_loss": dice_loss_value,
                "iou_loss": iou_loss_value
            }
        return total_loss, loss_dict

class Trainer:
    def __init__(self, wrap_model: PedestrainSAM2, train_loader, val_loader, config, rank, device_index, missing_keys):
        self.rank = rank
        self.device = torch.device('cuda', device_index)
        self.wrap_model = wrap_model
        self.model = wrap_model.module.sam2_model if isinstance(wrap_model, DDP) else wrap_model.sam2_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.valid_epoch = int(config["train"]["valid_every_train_epoch"])
        # Define loss function
        # self.criterion = CustomSegLoss(multimask=config["train"]["multimask"]).to(self.device)
        self.criterion = FocalDiceIoULoss(weight_focal=20.0, weight_dice=1.0, weight_iou=1.0,multimask=config["train"]["multimask"]).to(self.device)
        self.criterion_for_person =  nn.BCEWithLogitsLoss().to(self.device)
        missing_keys_params = []
        other_params = []
        trainable_layers = config['model']['trainable_layers']
        for name, param in self.model.named_parameters():
            if name in missing_keys or any(layer in name for layer in trainable_layers) or ("all" in trainable_layers):
                logger.info(f"{name} params is registered as missing_key_params_list")
                missing_keys_params.append(param)
            else:
                other_params.append(param)
        classifier_params = []
        for name, param in self.wrap_model.classifier.named_parameters():
            param.requires_grad = True
            classifier_params.append(param)
            logger.info(f"Classifier {name} params is registered as classifier parameters")
        for param in other_params:
            param.requires_grad = False
        param_groups = [
            {"params": list(filter(lambda p: p.requires_grad, missing_keys_params)), "lr": float(config["train"]["learning_rate"]) * config["train"]["lr_multiple_for_new_param"]},  # 10倍学习率
            # {"params": classifier_params, "lr": float(config["train"]["learning_rate_for_classifier"]) * config["train"]["lr_multiple_for_new_param"]},
            # {"params": filter(lambda p: p.requires_grad, other_params), "lr": float(config["train"]["learning_rate"])},  # 默认学习率
        ]
        for module in self.wrap_model.classifier.modules():
            # if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.Dropout):
            module.eval()  # 设置 BN 层为评估模式
            for param in module.parameters():
                param.requires_grad = False  # 冻结 BN 参数
            logger.info(f"Frozen layer: {module}")
        param_name_map = {param: name for name, param in self.model.named_parameters()}
        # Now print the param_groups for inspection
        for i, group in enumerate(param_groups):
            logger.info(f"Param group {i}:")
            logger.info(f"Learning rate: {group['lr']}")
            logger.info(f"Number of params: {len(group['params'])}")
            for param in group['params']:
                param_name = param_name_map.get(param, "Unknown")
                logger.info(f"Param name: {param_name}, Param shape: {param.shape}, requires_grad: {param.requires_grad}")
        # Define optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=float(config["train"]["learning_rate"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        #Total number of training steps
        self.num_epochs = config["train"]["num_epochs"]
        self.total_steps = self.num_epochs * len(self.train_loader)
        self.warm_up_step = int(config["train"]["warm_up_step_ratio"] * len(self.train_loader)) # 改成0.1的epoch
        def lr_lambda(current_step):
            if current_step < self.warm_up_step:
                # Linear warm-up
                return float(current_step) / float(max(1, self.warm_up_step))
            else:
                # Cosine annealing with minimum lr = 0.1 (1/10 of max lr)
                progress = float(current_step - self.warm_up_step) / float(max(1, self.total_steps - self.warm_up_step))
                return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Initialize GradScaler for mixed precision
        self.scaler = torch.amp.grad_scaler.GradScaler()
        self.max_norm = config["train"]["max_norm"]

        self.num_epochs = config["train"]["num_epochs"]
        self.save_path = config["train"]["save_path"]
        self.validate_save_path = os.path.join(self.save_path,"validate")
        self.train_save_path = os.path.join(self.save_path,"train")

        # Initialize logging (only on main process)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=config["log"]["tensorboard_log_dir"])
            logger.success(
                f"------------------------------------{self.device}Trainer initialized.---------------------------------"
            )
        self.log_interval_batch = config["log"]["log_interval_batch"]
        self.config = config
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """加载检查点文件进行断点续训"""
        checkpoint_path = self.config["model"]["pretrain_model_path"]
        if self.config["train"]["continue_training"] == False and "sam2.1" in self.config["model"]["pretrain_model_path"]:
            self.start_epoch = 0
            logger.info("Not resuming training. Set epoch to 0.")
            return
        elif  ("sam2.1" not in self.config["model"]["pretrain_model_path"]) and self.config["train"]["continue_training"] == True:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = int(checkpoint["epoch"]) + 1 # 这里的epoch应该要完成的
            self.scaler.load_state_dict(checkpoint["scaler"])
            logger.success(f"Resumed training from epoch {self.start_epoch}")
        else:
            raise NotImplementedError
    
    def train(self):
        start_time = time.time()
        total_steps = self.num_epochs * len(self.train_loader)
        global_step = self.start_epoch * len(self.train_loader) 
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            if self.config["train"]["validate_first"] == True and (epoch % self.valid_epoch == 0):
                val_loss = self.validate(epoch)
            if self.train_loader.sampler is not None and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            focal_loss_epoch = 0.0
            dice_loss_epoch = 0.0
            iou_loss_epoch = 0.0
            correct_predictions = 0
            total_predictions = 0
            stability_score = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                global_step += 1
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v,torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                images = batch["image"] # B, 3, 1024,1024  一般情况下是已经resize成1024的,在数据集就已经SAM2_transform
                gt_masks = batch["mask"]  # (B, 1024, 1024)
                # resize gt_masks to (256,256)
                gt_masks = F.interpolate(gt_masks.unsqueeze(1), size=(256, 256), mode='nearest').squeeze(1)
                click_points = batch["click_point"]  # (B, N, 2) 1024的绝对坐标
                point_labels = batch["point_label"]  # (B, N)
                is_person_labels = batch["is_person"]  # (B, 1)
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs, pred_person_logits, iou_predictions = self.wrap_model.forward(images, click_points, point_labels)
                    if self.config["train"]["vis_for_classfier_image"] == True:
                        binary_mask = (outputs > 0.0).float()
                        person_label_tf = [True if i == 1 else False for i in is_person_labels]
                        extracted_images = self.wrap_model._extract_and_resize_images(images, binary_mask, person_label_tf, [False] * len(person_label_tf), whether_vis=True)
                    with torch.no_grad():
                        person_loss = self.criterion_for_person(pred_person_logits, is_person_labels)
                    mask_loss, loss_dict = self.criterion(outputs, gt_masks, iou_predictions)
                loss = mask_loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item()
                focal_loss_epoch += loss_dict["focal_loss"].item()
                dice_loss_epoch += loss_dict["dice_loss"].item()
                iou_loss_epoch += loss_dict["iou_loss"].item()
                
                # compute the accuracy of classfication
                pred_person_label = (torch.sigmoid(pred_person_logits) > 0.5).long()
                is_person_labels_long = is_person_labels.long()
                correct = (pred_person_label == is_person_labels_long).sum().item()
                correct_predictions += correct
                total_predictions += is_person_labels.size(0)
                stability_score += calculate_stability_score(
                    outputs, self.wrap_model._transforms.mask_threshold,1.0
                ).mean().item()

                # Logging (only on main process)
                if batch_idx % self.log_interval_batch == 0 and self.rank == 0:
                    current_step = epoch * len(self.train_loader) + batch_idx + 1
                    elapsed_time = time.time() - start_time

                    estimated_total_time = elapsed_time / current_step * total_steps
                    remaining_time = estimated_total_time - elapsed_time

                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)

                    logger.info(
                        f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{batch_idx+1}/{len(self.train_loader)}], "
                        f"Mask Loss: {mask_loss.item():.4f}, "
                        f"The Full Remaining Time: {remaining_hours:02d} hours:{remaining_minutes:02d} minutes"
                    )
                if self.rank == 0:
                    self.writer.add_scalar(
                        "Training/Mask Loss",
                        mask_loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Person Loss",
                        person_loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Focal Loss",
                        loss_dict["focal_loss"].item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Dice Loss",
                        loss_dict["dice_loss"].item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/IOU Loss",
                        loss_dict["iou_loss"].item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Prediction Accuray",
                        correct_predictions / total_predictions,
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Real IOU",
                        loss_dict["real_iou"].item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Learning Rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch * len(self.train_loader) + batch_idx,
                    )
                    self.writer.add_scalar(
                        "Training/Stability Score",
                        stability_score / (batch_idx + 1),
                        epoch * len(self.train_loader) + batch_idx,
                    )
                visualize_person_loss = person_loss.item() > 0.25 and epoch >= 10
                visualize_mask_loss = mask_loss.item() > 0.05 and epoch >= 10
                img_paths = batch["image_path"]
                track_ids = batch["track_id"]
                if (visualize_person_loss or visualize_mask_loss) and img_paths is not None:
                    logger.warning(
                        f"Person Loss: {person_loss.item():.4f}, Mask Loss: {mask_loss.item():.4f}, 超出上限！！！可视化"
                    )
                    for i in range(images.size(0)):
                        mask_0_1_2_3 = loss_dict["real_backward_mask_logits_index"][i]
                        mask_to_vis = (outputs[i, mask_0_1_2_3, :] > 0.0).detach().cpu().numpy()
                        self.save_image_and_masks(
                            img_path=img_paths[i],
                            original_image=images[i],
                            output_mask=mask_to_vis,
                            gt_mask=gt_masks[i].detach().cpu().numpy(),
                            click_points=click_points[i],
                            point_labels=point_labels[i],
                            person_class=is_person_labels[i],
                            pred_person_label=pred_person_label[i],
                            epoch_or_step = global_step,
                            track_id=track_ids[i] if track_ids is not None else i,
                            train_save=True,
                        )
                       

            if self.config["train"]["validate_first"] == False and ((epoch + 1) % self.valid_epoch == 0):
                val_loss = self.validate(epoch)
                logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}"
                )
            # Save checkpoint (only on main process)
            if self.rank == 0:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "scheduler": self.scheduler.state_dict(), # DEBUG
                    "epoch": epoch,
                    "global_step": global_step,
                    "classifier": self.wrap_model.classifier.state_dict(),
                }
                torch.save(
                    checkpoint,
                    f"{self.save_path}/model_epoch_{epoch+1}.pth",
                )

                # Record epoch loss
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                avg_epoch_loss_focal = focal_loss_epoch / len(self.train_loader)
                avg_epoch_loss_dice = dice_loss_epoch / len(self.train_loader)
                avg_epoch_loss_iou = iou_loss_epoch / len(self.train_loader)
                self.writer.add_scalar("Epoch/Training Loss", avg_epoch_loss, epoch)
                self.writer.add_scalar("Epoch/Focal Loss", avg_epoch_loss_focal, epoch)
                self.writer.add_scalar("Epoch/Dice Loss", avg_epoch_loss_dice, epoch)
                self.writer.add_scalar("Epoch/IOU Loss", avg_epoch_loss_iou, epoch)
                self.writer.add_scalar("Epoch/Prediction Accuray", correct_predictions / total_predictions, epoch)
                self.writer.add_scalar("Epoch/Real IOU", loss_dict["real_iou"], epoch)
                self.writer.add_scalar("Epoch/Stability Score", stability_score / len(self.train_loader), epoch)
                logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {avg_epoch_loss:.4f}"
                )
        if self.config["train"]["validate_first"] == True and epoch % self.valid_epoch == 0:
            val_loss = self.validate(epoch)
        
    def validate(self, epoch):
        self.model.eval()
        val_mask_loss = 0.0
        val_person_loss = 0.0
        val_focal_loss = 0.0
        val_dice_loss = 0.0
        val_iou_loss = 0.0
        start_time = time.time()
        total_batches = len(self.val_loader)
        correct_predictions = 0
        total_predictions = 0
        real_iou = 0
        stability_score = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)

                images = batch["image"]
                gt_masks = batch["mask"]
                click_points = batch["click_point"]
                point_labels = batch["point_label"]
                is_person_labels = batch["is_person"]  # Retrieve is_person_labels
                image_paths = batch["image_path"]
                track_ids = batch["track_id"]
                b, h, w = gt_masks.shape

                # Forward pass with autocast
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs, pred_person_logits, iou_predictions = self.wrap_model.forward(
                        images, click_points, point_labels
                    )
                    # 计算Person分类损失
                    person_loss = self.criterion_for_person(pred_person_logits, is_person_labels)
                    mask_loss, loss_dict = self.criterion(outputs, gt_masks, iou_predictions)

                # Accumulate losses
                val_mask_loss += mask_loss.item()
                val_person_loss += person_loss.item()
                val_focal_loss += loss_dict["focal_loss"].item()
                val_dice_loss += loss_dict["dice_loss"].item()
                val_iou_loss += loss_dict["iou_loss"].item()
                pred_person_probs = torch.sigmoid(pred_person_logits)
                pred_person_labels = (pred_person_probs > 0.5).long()
                is_person_labels = is_person_labels.long()
                correct = (pred_person_labels == is_person_labels).sum().item()
                correct_predictions += correct
                total_predictions += is_person_labels.size(0)
                real_iou += loss_dict["real_iou"].item()
                stability_score_this_batch = calculate_stability_score(
                    outputs, self.wrap_model._transforms.mask_threshold, 1.0
                ).mean().item()
                stability_score += stability_score_this_batch
                
                if loss_dict["real_iou"] < 0.65:
                    for i in range(b):
                        self.save_image_and_masks(
                            img_path=image_paths[i],
                            original_image=images[i],
                            output_mask=(outputs[i] > 0.0).detach().cpu().numpy(),
                            gt_mask=gt_masks[i].detach().cpu().numpy(),
                            click_points=click_points[i],
                            point_labels=point_labels[i],
                            person_class=is_person_labels[i],
                            pred_person_label=pred_person_labels[i],
                            epoch_or_step=epoch,
                            track_id=track_ids[i] if track_ids is not None else i,
                            train_save=False,
                        )
                self.writer.add_scalar("Validation/Step Real IOU", loss_dict["real_iou"], epoch * len(self.val_loader) + batch_idx)
                self.writer.add_scalar("Validation/Step Correct Prediction", correct / is_person_labels.size(0), epoch * len(self.val_loader) + batch_idx)
                self.writer.add_scalar("Validation/Step Stability Score", stability_score_this_batch, epoch * len(self.val_loader) + batch_idx)
                if (batch_idx + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    batches_remaining = total_batches - (batch_idx + 1)
                    estimated_time_per_batch = elapsed_time / (batch_idx + 1)
                    estimated_remaining_time = batches_remaining * estimated_time_per_batch
                    logger.info(
                        f"Processed {batch_idx + 1}/{total_batches} batches. "
                        f"Elapsed Time: {elapsed_time:.2f} seconds, "
                        f"Estimated Remaining Time: {(estimated_remaining_time // 60):.2f} minutes."
                    )

        # Compute average losses
        val_mask_loss /= len(self.val_loader)
        val_person_loss /= len(self.val_loader)
        val_focal_loss /= len(self.val_loader)
        val_dice_loss /= len(self.val_loader)
        val_iou_loss /= len(self.val_loader)
        real_iou /= len(self.val_loader)
        stability_score /= len(self.val_loader)
        val_loss = val_mask_loss + val_person_loss  # Total validation loss

        # Aggregate validation losses across all processes
        val_mask_loss_tensor = torch.tensor(val_mask_loss).to(self.device)
        val_person_loss_tensor = torch.tensor(val_person_loss).to(self.device)
        if dist.is_initialized():
            dist.all_reduce(val_mask_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_person_loss_tensor, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            val_mask_loss = val_mask_loss_tensor.item() / world_size
            val_person_loss = val_person_loss_tensor.item() / world_size
            val_loss = val_mask_loss + val_person_loss  # Update total loss after aggregation

        # Logging (only on main process)
        if self.rank == 0:
            logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}], "
                f"Validation Total Loss: {val_loss:.4f}, "
                f"Mask Loss: {val_mask_loss:.4f}, "
                f"Person Loss: {val_person_loss:.4f}, "
                f"Persion Accuray: {correct_predictions / total_predictions:.4f}, "
                f"Real IOU: {real_iou:.4f}, "
                f"Stability Score: {stability_score:.4f}"
            )
            # Write losses to TensorBoard
            self.writer.add_scalar("Validation/Total Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/Mask Loss", val_mask_loss, epoch)
            self.writer.add_scalar("Validation/Person Loss", val_person_loss, epoch)
            self.writer.add_scalar("Validation/Focal Loss", val_focal_loss, epoch)
            self.writer.add_scalar("Validation/Dice Loss", val_dice_loss, epoch)
            self.writer.add_scalar("Validation/IOU Loss", val_iou_loss, epoch)
            self.writer.add_scalar("Validation/Prediction Accuray", correct_predictions / total_predictions, epoch)
            self.writer.add_scalar("Validation/Real IOU", real_iou, epoch)
            self.writer.add_scalar("Validation/Stability Score", stability_score, epoch)

        return val_loss

    def blend_image_with_mask(self, original_image, mask_image, mask_color=(255, 0, 0, 100)):
        # Convert mask to RGBA with desired color
        mask_rgba = Image.new('RGBA', original_image.size, mask_color)
        mask_alpha = mask_image.convert('L')  # Convert mask to grayscale
        mask_rgba.putalpha(mask_alpha)
        original_image_rgba = original_image.convert('RGBA')
        blended = Image.alpha_composite(original_image_rgba, mask_rgba)
        return blended

        
    def draw_click_points(self, image, click_points, point_labels, person_class, image_size):
        def draw_star(draw, center, size, fill, outline=None, width=5):
            """
            绘制五角星
            :param draw: ImageDraw 对象
            :param center: 五角星中心 (x, y)
            :param size: 五角星大小（半径）
            :param fill: 填充颜色
            :param outline: 描边颜色
            :param width: 描边宽度
            """
            x, y = center
            
            points = []
            num_points = 5
            angle = math.pi / 2  # 起始角度，五角星朝上
            for i in range(num_points * 2):
                radius = size if i % 2 == 0 else size / 2
                theta = angle + i * (math.pi / num_points)
                px = x + radius * math.cos(theta)
                py = y - radius * math.sin(theta)
                points.append((px, py))
            draw.polygon(points, fill=fill, outline=outline)
            if outline:
                draw.line(points + [points[0]], fill=outline, width=width)
        
        
        draw = ImageDraw.Draw(image)
        orig_w, orig_h = image_size
        # Assuming images are resized to (1024, 1024) during preprocessing
        scale_w = orig_w / 1024
        scale_h = orig_h / 1024

        points = click_points.cpu().numpy()
        labels = point_labels.cpu().numpy()
        person_class = person_class.cpu().numpy()
        for pt, label, pc in zip(points, labels,person_class):
            x = pt[0] * scale_w
            y = pt[1] * scale_h
            if int(pc) == 1:
                color = 'blue'  # Person
            elif int(pc) == 0:
                color = 'cyan'    # Object
            else:
                raise NotImplementedError

            if label > 0:
                # Positive click: draw circle
                size = 16  # 增大尺寸
                draw_star(draw, (x, y), size, fill=color, outline='red', width=2)
            else:
                # Negative click: draw cross
                size = 15
                draw.line((x - size, y - size, x + size, y + size), fill=color, width=2)
                draw.line((x - size, y + size, x + size, y - size), fill=color, width=2)
        return image
    
    def save_image_and_masks(self, img_path, original_image: torch.tensor, output_mask:np.ndarray, gt_mask:np.ndarray, click_points, 
                             point_labels, person_class, pred_person_label, epoch_or_step, track_id, train_save = False):
        '''
        output_mask 应该是[0,1]的ndarray or True/False的ndarray
        '''
        if len(output_mask) == 1:
            output_mask = output_mask.squeeze(0)
        if len(gt_mask) == 1:
            gt_mask = gt_mask.squeeze(0)
        if train_save:
            save_base_dir = self.train_save_path
        else:
            save_base_dir = self.validate_save_path
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Reverse normalization  original_image
        unnormalized_tensor = original_image.clone()
        if unnormalized_tensor.is_cuda:
            unnormalized_tensor = unnormalized_tensor.cpu()
        for t, m, s in zip(unnormalized_tensor, mean, std):
            t.mul_(s).add_(m)
        unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
        original_image = torch_transforms.ToPILImage()(unnormalized_tensor)
        last_three_parts = os.path.join(*img_path.split(os.sep)[-3:])  # e.g., 'MOTS20-09/img1/000426.jpg'
        save_dir = os.path.join(save_base_dir, str(epoch_or_step+1), os.path.dirname(last_three_parts))
        os.makedirs(save_dir, exist_ok=True)

        # Save original image
        save_img_path = os.path.join(save_base_dir, str(epoch_or_step+1), last_three_parts)
        original_image.save(save_img_path)
        
        image_before_augment = Image.open(img_path)
        image_before_augment.save(os.path.join(
            save_base_dir,
            str(epoch_or_step+1),
            os.path.splitext(last_three_parts)[0]+ f'_before_augment.jpg'
        ))

        # Resize masks to original image size
        # Predicted mask
        pred_mask = Image.fromarray((output_mask * 255).astype(np.uint8))
        pred_mask = pred_mask.resize(original_image.size, resample=Image.BILINEAR)
        # Ground truth mask
        gt_mask_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
        gt_mask_img = gt_mask_img.resize(original_image.size, resample=Image.NEAREST)

        # Save predicted mask only image
        save_pred_mask_only_path = os.path.join(
            save_base_dir,
            str(epoch_or_step+1),
            os.path.splitext(last_three_parts)[0]+ f'_{track_id}_pred_mask_only.jpg'
        )
        pred_mask.convert('L').save(save_pred_mask_only_path)

        # Save ground truth mask only image
        save_gt_mask_only_path = os.path.join(
            save_base_dir,
            str(epoch_or_step+1),
            os.path.splitext(last_three_parts)[0] + f'_{track_id}_gt_mask_only.jpg'
        )
        gt_mask_img.convert('L').save(save_gt_mask_only_path)

        # Create blended image of original image and predicted mask
        blended_pred = self.blend_image_with_mask(original_image, pred_mask, mask_color=(255, 0, 0, 100))
        # Draw click points on blended_pred
        blended_pred = self.draw_click_points(blended_pred, click_points, point_labels, person_class, image_size=original_image.size)
        
        draw = ImageDraw.Draw(blended_pred)
        font_path = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)

        # Determine the real label and predicted label
        if int(person_class) == 1:
            real_label_text = 'real label: person'
        else:
            real_label_text = 'real label: background'

        if int(pred_person_label) == 1:
            pred_label_text = 'pred label: person'
        else:
            pred_label_text = 'pred label: background'

        if int(person_class) == int(pred_person_label):
            color = 'green'  # 预测正确，绿色
        else:
            color = 'red'  # 预测错误，红色
        draw.text((10, 10), real_label_text, fill=color, font=font)  # 始终为白色
        draw.text((10, 50), pred_label_text, fill=color, font=font)  # 根据预测情况改变颜色
        # Save blended image with predicted mask
        save_pred_mask_img_path = os.path.join(
            save_base_dir,
            str(epoch_or_step+1),
            os.path.splitext(last_three_parts)[0] + f'_{track_id}_pred_mask_overlay.jpg'
        )
        blended_pred.convert('RGB').save(save_pred_mask_img_path)

        # Create blended image of original image and ground truth mask
        blended_gt = self.blend_image_with_mask(original_image, gt_mask_img, mask_color=(0, 255, 0, 100))
        # Save blended image with ground truth mask
        save_gt_mask_img_path = os.path.join(
            save_base_dir,
            str(epoch_or_step+1),
            os.path.splitext(last_three_parts)[0] + f'_{track_id}_gt_mask_overlay.jpg'
        )
        blended_gt.convert('RGB').save(save_gt_mask_img_path)
        

def get_dataset(config, dataset_type="train", whether_augument=True):
    """
    根据配置文件自动选择并加载单个或混合数据集。
    
    Args:
        config (dict): 配置文件内容
        dataset_type (str): 数据集类型（"train" 或 "val"）

    Returns:
        dataset: 返回加载后的数据集
    """
    main_datasets = []
    aux_datasets = []
    ratios = []

    if config["dataset"].get("use_mix_mode", False):
        dataset_config_list = config["dataset"].get(f"{dataset_type}_datasets", [])
        for dataset_config in dataset_config_list:
            dataset_name = dataset_config["name"]
            dataset_path = dataset_config["path"]
            ratio = dataset_config.get("ratio", 1.0 / len(dataset_config_list))  
            if ratio == 1.0:
                datasets = main_datasets
            else:
                datasets = aux_datasets
                ratios.append(ratio)
            val_length = dataset_config.get("val_length", -1)  # 默认-1，即不限制长度
            
            if dataset_name == "MOTS":
                datasets.append(MOTSDataset(
                    root_path=dataset_path,
                    use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"], 
                    augment=whether_augument,
                    enable_negative_sample=True,
                    max_length_for_validate=val_length if dataset_type == "val" else None
                ))
            elif dataset_name in ["openimagev7", "coco", "samacoco", "crowdhuman"]:
                annotation_file = dataset_config.get("annotation_file", None)
                datasets.append(COCOPersonDataset(
                    images_dir=dataset_path,
                    annotation_file=annotation_file,
                    use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"], 
                    augment= whether_augument,
                    enable_negative_sample=True,
                    max_length_for_validate=val_length if dataset_type == "val" else None
                ))
            else:
                raise ValueError(f"未识别的数据集名称: {dataset_name}")
            
        dataset = CombinedDataset(main_datasets=main_datasets, auxiliary_datasets=aux_datasets, auxiliary_ratio=ratios)
    else:
        # 单一数据集模式
        dataset_name = config["dataset"].get(f"{dataset_type}_dataloader")
        dataset_path = config["dataset"].get(f"{dataset_type}_dataset_root_path")
        annotation_file = config["dataset"].get(f"{dataset_type}_annotation_file", None)
        val_length = config["dataset"].get("val_length", -1)

        if dataset_name == "MOTS":
            dataset = MOTSDataset(
                root_path=dataset_path,
                use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"],
                augment=whether_augument,
                enable_negative_sample=True,
                max_length_for_validate=val_length if dataset_type == "val" else None
            )
        elif dataset_name in ["openimagev7", "coco", "samacoco"]:
            dataset = COCOPersonDataset(
                images_dir=dataset_path,
                annotation_file=annotation_file,
                use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"],
                augment=whether_augument,
                enable_negative_sample=True,
                max_length_for_validate=val_length if dataset_type == "val" else None
            )
        else:
            raise ValueError(f"未识别的数据集名称: {dataset_name}")

    return dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./train_config/train_large.yaml")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()
    print(f"From argument local_rank:{args.local_rank},device_index:{args.device_index}")
    if 'LOCAL_RANK' in os.environ and args.local_rank is None:
        logger.info(f"From os.environ LOCAL_RANK:{os.environ['LOCAL_RANK']}")
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        # DDP 模式
        print(f"!!!!!!!!!!!!!!!!!!!Choose DDP mode,world_size{os.environ['WORLD_SIZE']}!!!!!!!!!!!!!!!!!!!")
        torch.cuda.set_device(args.local_rank)#理论上torchrun 自动分配
        dist.init_process_group(
            backend="nccl", 
            init_method="tcp://10.16.64.8:29500",  # 主节点的 IP 和端口
            world_size=int(os.environ['WORLD_SIZE']),
            rank=int(os.environ['RANK'])
        )
        dist.barrier()
        device_index =args.local_rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.success(f"!!!!!!!!!!!!!!!Come to the DDP mode--rank:{rank}--world_size:{world_size}--device:cuda{device_index}!!!!!!!!!!!!!!!!")
    else:
        # 单 GPU 模式
        device_index = args.device_index
        rank = args.local_rank
        world_size = 1
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    if rank == 0 :
        logger.add(config["log"]["loguru_log_file"])
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
    config["train"]["save_path"] = os.path.join(config["train"]["save_path"] ,formatted_time)
    config["log"]["tensorboard_log_dir"] = os.path.join(config["log"]["tensorboard_log_dir"] ,formatted_time)
    if rank == 0:
        if not os.path.exists(config["train"]["save_path"]):
            os.makedirs(config["train"]["save_path"])
        shutil.copytree("./pedestrainSAM", os.path.join(config["train"]["save_path"],"code", "pedestrainSAM")) # 保存一些代码
        shutil.copytree("./sam2", os.path.join(config["train"]["save_path"],"code", "sam2"))
        shutil.copy(args.config_path, os.path.join(config["train"]["save_path"], os.path.basename(args.config_path)))
    # Build dataset and dataloader
    train_dataset = get_dataset(config, dataset_type="train", whether_augument=False)
    val_dataset = get_dataset(config, dataset_type="val", whether_augument=False)

    if world_size > 1:
        # 多 GPU 模式，使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        # 单 GPU 模式，使用普通的随机采样器
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size_one_gpu"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["train"]["num_workers_dataloader"],
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size_one_gpu"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["train"]["num_workers_dataloader"],
        pin_memory=True,
    )
    # Build model
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model, missing_keys, unexpected_keys = build_sam2_for_self_train(
        model_config_path_for_build_sam2,
        sam2_checkpoint,
        device=torch.device('cuda', device_index),
        apply_postprocessing=True,
    )
    pedestrain_sam_model = PedestrainSAM2(model=sam2_model, config=config,device_index=device_index)
    pedestrain_sam_model.load_classifier(config["model"]["pretrain_classfier_path"])


    # Wrap model with DDP
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        logger.success("Wrapping model with DDP")
        pedestrain_sam_model = DDP(pedestrain_sam_model, device_ids=[device_index], output_device=device_index, find_unused_parameters=True)
    # Create trainer
    trainer = Trainer(
        pedestrain_sam_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        rank=rank,
        device_index=device_index,
        missing_keys=missing_keys
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    print("---------------come into main---------------")
    main()
