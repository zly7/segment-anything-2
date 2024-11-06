import math
import os
import shutil
import time
import numpy as np
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
from sam2.utils.transforms import SAM2Transforms
from typing import Optional
from dataset_zly import MOTSDataset,COCOPersonDataset, CombinedDataset
from PIL import Image, ImageDraw, ImageFont
from training.loss_fns import sigmoid_focal_loss, dice_loss, iou_loss
from typing import Union
from torchvision import transforms as torch_transforms  # 导入 torchvision 的 transforms 模块
print("!!!!!!!!!!!!come to this file!!!!!!!!!!!!!!")
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
        # Freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image:Optional[Union[torch.Tensor, np.ndarray]], point_coords = None , point_labels = None, box = None):
        """
        image tensor: (B, 3, H, W) (B,3,1024,1024)
        points_coords: (B, N, 2)
        point_labels: (B, N)
        box: (B, 2, 2)
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
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, None, normalize_coords=True
        )
        if point_coords is not None:
            concat_points = (unnorm_coords, labels)
        else:
            concat_points = None
        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        ) # spare [b,2,256],[6,256,64,64]
        # if not self.config["train"]["transfer_image_in_dataset_use_sam2"]:
        #     high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in _features["high_res_feats"]] # 这句话直接把[b,32,256,256] 变成[1,32,256,256],想想都觉得铁错
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits, person_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
            use_hq = True,
        )
        if len(person_logits.size()) == 3 and person_logits.size(1) == 1:
            person_logits = person_logits.squeeze(1)
        low_res_masks_logits = torch.clamp(low_res_masks_logits, -32.0, 32.0)
        prd_masks = self._transforms.postprocess_masks(low_res_masks_logits, self._orig_hw[-1])
        return prd_masks, person_logits, iou_predictions

    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
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
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        
        _features = self._image_encoder(input_image)
        self.img_embed = _features["image_embed"]
        self.high_res_features = _features["high_res_feats"]
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

    def predict_torch(
        self,
        point_coords = None,
        point_labels = None,
        box = None,
        sparse_embeddings=None,
        dense_embeddings=None,
        predict_logit = False,
        use_hq = False,
    ):
        """
        Predict masks for the given prompts.
        """
        # Use stored embeddings
        img_embed = self.img_embed
        high_res_features = self.high_res_features
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
                boxes=box,
                masks=None,
            )

        batch_size = sparse_embeddings.shape[0]
        if batch_size > 1:
            repeat_image = True
        else:
            repeat_image = False

        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits, person_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (1, 256, 64, 64)
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            use_hq = use_hq,
            
        )
        if len(person_logits.size()) == 3 and person_logits.size(1) == 1:
            person_logits = person_logits.squeeze(1)
        low_res_masks_logits = torch.clamp(low_res_masks_logits, -32.0, 32.0)
        prd_masks = self._transforms.postprocess_masks(low_res_masks_logits, self._orig_hw[-1])
        if predict_logit == False:
            prd_masks = prd_masks > self._transforms.mask_threshold
        return prd_masks, iou_predictions, person_logits, low_res_masks_logits





class FocalDiceIoULoss(nn.Module):
    def __init__(self, weight_focal=20.0, weight_dice=1.0, weight_iou=1.0, 
                 focal_alpha=0.25, focal_gamma=2, 
                 iou_use_l1_loss=False):
        super(FocalDiceIoULoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.total_weights = weight_focal + weight_dice + weight_iou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_use_l1_loss = iou_use_l1_loss

    def forward(self, mask_logits, targets, iou_predictions=None):
        """
        Args:
            mask_logits: [B, H, W] - 预测的mask logits
            targets: [B, H, W] - 真实的mask
            iou_predictions: [B, 1] - 预测的IOU分数（可选）
        Returns:
            总损失
        """
        # Focal Loss
        focal_loss = sigmoid_focal_loss(
            mask_logits,
            targets,
            num_objects=targets.size(0),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=False
        )
        
        # Dice Loss
        dice = dice_loss(
            mask_logits,
            targets,
            num_objects=targets.size(0),
            loss_on_multimask=False
        )
        
        # IOU Loss
        if iou_predictions is not None:
            if len(targets.size()) == 3:
                targets = targets.unsqueeze(1)
            if len(mask_logits.size()) == 3:
                mask_logits = mask_logits.unsqueeze(1) 
            iou = iou_loss(
                mask_logits,
                targets,
                pred_ious=iou_predictions,
                num_objects=targets.size(0),
                loss_on_multimask=False,
                use_l1_loss=self.iou_use_l1_loss
            )
        else:
            iou = torch.tensor(0.0, device=mask_logits.device)

        # 加权总损失
        total_loss = (self.weight_focal * focal_loss) + \
                     (self.weight_dice * dice) + \
                     (self.weight_iou * iou)
        total_loss = total_loss / self.total_weights
        return total_loss


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
        self.criterion = FocalDiceIoULoss(weight_focal=20.0, weight_dice=1.0, weight_iou=1.0).to(self.device)
        self.criterion_for_person =  nn.BCEWithLogitsLoss().to(self.device)
        missing_keys_params = []
        other_params = []
        trainable_layers = config['model']['trainable_layers']
        for name, param in self.model.named_parameters():
            if name in missing_keys or any(layer in name for layer in trainable_layers):
                logger.info(f"{name} params is registered as missing_key_params_list")
                missing_keys_params.append(param)
            else:
                other_params.append(param)
        for param in other_params:
            param.requires_grad = False
        param_groups = [
            {"params": list(filter(lambda p: p.requires_grad, missing_keys_params)), "lr": float(config["train"]["learning_rate"]) * 10},  # 10倍学习率
            # {"params": filter(lambda p: p.requires_grad, other_params), "lr": float(config["train"]["learning_rate"])},  # 默认学习率
        ]
        # Now print the param_groups for inspection
        for i, group in enumerate(param_groups):
            logger.info(f"Param group {i}:")
            logger.info(f"Learning rate: {group['lr']}")
            logger.info(f"Number of params: {len(group['params'])}")
            for param in group['params']:
                logger.info(f"Param shape: {param.shape}, requires_grad: {param.requires_grad}")
        # Define optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=float(config["train"]["learning_rate"]),
            weight_decay=0.1
        )
        #Total number of training steps
        self.num_epochs = config["train"]["num_epochs"]
        self.total_steps = self.num_epochs * len(self.train_loader)
        self.warm_up_step = config["train"]["warm_up_step"]
        self.warm_up_scheduler =  LambdaLR(self.optimizer, lr_lambda=lambda step: step / self.warm_up_step)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.warm_up_step,
            eta_min=0
        )
        self.scheduler = ChainedScheduler([self.warm_up_scheduler, self.cosine_scheduler])

        # Initialize GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_norm = config["train"]["max_norm"]

        self.num_epochs = config["train"]["num_epochs"]
        self.save_path = config["train"]["save_path"]
        self.validate_save_path = os.path.join(self.save_path,"validate")

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
        if self.config["train"]["continue_training"] == False:
            self.start_epoch = 0
            logger.info("Not resuming training.")
            return
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"]
            self.scaler.load_state_dict(checkpoint["scaler"])
            logger.success(f"Resumed training from epoch {self.start_epoch}")
    
    def train(self):
        start_time = time.time()
        total_steps = (self.num_epochs-self.start_epoch) * len(self.train_loader)
        global_step = 0  # Track the global step
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            if self.config["train"]["validate_first"] == True and (epoch % self.valid_epoch == 0):
                val_loss = self.validate(epoch)
            if self.train_loader.sampler is not None and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                global_step += 1
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v,torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                images = batch["image"] # B, 3, 1024,1024  一般情况下是已经resize成1024的,在数据集就已经SAM2_transform
                gt_masks = batch["mask"]  # (B, 1024, 1024)
                click_points = batch["click_point"]  # (B, N, 2) 1024的绝对坐标
                point_labels = batch["point_label"]  # (B, N)
                is_person_labels = batch["is_person"]  # (B, 1)
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    outputs, pred_person_logits, iou_predictions = self.wrap_model.forward(images, click_points, point_labels)
                    # 计算Person分类损失
                    person_loss = self.criterion_for_person(pred_person_logits, is_person_labels)
                    person_indices = (is_person_labels.squeeze() == 1).nonzero(as_tuple=True)[0]
                    mask_loss = torch.tensor(0.0, device=self.device)
                    # 仅对is_person_labels == 1的样本计算mask损失
                    if person_indices.numel() > 0:
                        outputs_person = outputs[person_indices]
                        gt_masks_person = gt_masks[person_indices]
                        iou_predictions_person = iou_predictions[person_indices]
                        if len(outputs_person.size()) == 4 and outputs_person.size(1) == 1:
                            outputs_person = outputs_person.squeeze(1)
                        mask_loss = self.criterion(outputs_person, gt_masks_person.float(), iou_predictions_person)

                    loss = mask_loss + person_loss
                # Backward and optimize
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item()

                # Logging (only on main process)
                if batch_idx % self.log_interval_batch == 0 and self.rank == 0:
                    # 计算当前步骤数和已经消耗的时间
                    current_step = epoch * len(self.train_loader) + batch_idx + 1
                    elapsed_time = time.time() - start_time

                    # 估计总的训练时间和剩余时间
                    estimated_total_time = elapsed_time / current_step * total_steps
                    remaining_time = estimated_total_time - elapsed_time

                    # 将剩余时间转换为小时、分钟和秒
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)

                    logger.info(
                        f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{batch_idx+1}/{len(self.train_loader)}], "
                        f"Mask Loss: {mask_loss.item():.4f}, "
                        f"Person Classfication Loss: {person_loss.item():.4f}, "
                        f"The Full Remaining Time: {remaining_hours:02d} hours:{remaining_minutes:02d} minutes"
                    )
                    self.writer.add_scalar(
                        "Training/Loss",
                        loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )
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
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    checkpoint,
                    f"{self.save_path}/model_epoch_{epoch+1}.pth",
                )

                # Record epoch loss
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar("Epoch/Training Loss", avg_epoch_loss, epoch)
                logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {avg_epoch_loss:.4f}"
                )
        if self.config["train"]["validate_first"] == True and epoch % self.valid_epoch == 0:
            val_loss = self.validate(epoch)
        
    def validate(self, epoch):
        self.model.eval()
        val_mask_loss = 0.0
        val_person_loss = 0.0
        start_time = time.time()
        total_batches = len(self.val_loader)
        is_person_labels_num = 0
        not_person_labels_num = 0
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
                with torch.cuda.amp.autocast():
                    outputs, pred_person_logits, iou_predictions = self.wrap_model.forward(
                        images, click_points, point_labels
                    )
                    # 计算Person分类损失
                    person_loss = self.criterion_for_person(pred_person_logits, is_person_labels)
                    person_indices = (is_person_labels.squeeze() == 1).nonzero(as_tuple=True)[0]
                    mask_loss = torch.tensor(0.0, device=self.device)
                    # 仅对is_person_labels == 1的样本计算mask损失
                    if person_indices.numel() > 0:
                        outputs_person = outputs[person_indices]
                        gt_masks_person = gt_masks[person_indices]
                        iou_predictions_person = iou_predictions[person_indices]
                        if len(outputs_person.size()) == 4 and outputs_person.size(1) == 1:
                            outputs_person = outputs_person.squeeze(1)
                        mask_loss = self.criterion(outputs_person, gt_masks_person.float(), iou_predictions_person)

                # Accumulate losses
                val_mask_loss += mask_loss.item()
                val_person_loss += person_loss.item()
                # Compute predicted person labels
                pred_person_probs = torch.sigmoid(pred_person_logits)
                pred_person_labels = (pred_person_probs > 0.5).int()

                for i in range(b):
                    img_path = image_paths[i]
                    original_image = images[i]  #changed: Use augmented image
                    output_mask = (outputs[i] > 0.0).cpu().numpy().squeeze()
                    gt_mask = gt_masks[i].cpu().numpy().squeeze()
                    click_pts = click_points[i]
                    point_lbls = point_labels[i]
                    whether_person = is_person_labels[i]
                    track_id = track_ids[i]
                    pred_person_label = pred_person_labels[i]

                    # Call the helper function with the predicted person label
                    self.save_image_and_masks(
                        img_path, original_image, output_mask, gt_mask, click_pts, point_lbls, whether_person,
                            pred_person_label, epoch, track_id
                    )
                is_person_labels_num += is_person_labels.sum().item()
                not_person_labels_num += (1 - is_person_labels).sum().item()
                if (batch_idx + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    batches_remaining = total_batches - (batch_idx + 1)
                    estimated_time_per_batch = elapsed_time / (batch_idx + 1)
                    estimated_remaining_time = batches_remaining * estimated_time_per_batch

                    # Log progress
                    logger.info(
                        f"Processed {batch_idx + 1}/{total_batches} batches. "
                        f"Elapsed Time: {elapsed_time:.2f} seconds, "
                        f"Estimated Remaining Time: {(estimated_remaining_time // 60):.2f} minutes."
                        f"Person Labels Num: {is_person_labels_num}, Not Person Labels Num: {not_person_labels_num}"
                    )

        # Compute average losses
        val_mask_loss /= len(self.val_loader)
        val_person_loss /= len(self.val_loader)
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
                f"Person Loss: {val_person_loss:.4f}"
            )
            # Write losses to TensorBoard
            self.writer.add_scalar("Validation/Total Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/Mask Loss", val_mask_loss, epoch)
            self.writer.add_scalar("Validation/Person Loss", val_person_loss, epoch)

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
                             point_labels, person_class, pred_person_label, epoch, track_id):
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
        save_dir = os.path.join(self.validate_save_path, str(epoch+1), os.path.dirname(last_three_parts))
        os.makedirs(save_dir, exist_ok=True)

        # Save original image
        save_img_path = os.path.join(self.validate_save_path, str(epoch+1), last_three_parts)
        original_image.save(save_img_path)
        
        image_before_augment = Image.open(img_path)
        image_before_augment.save(os.path.join(
            self.validate_save_path,
            str(epoch+1),
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
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0]+ f'_{track_id}_pred_mask_only.jpg'
        )
        pred_mask.convert('L').save(save_pred_mask_only_path)

        # Save ground truth mask only image
        save_gt_mask_only_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
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
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + f'_{track_id}_pred_mask_overlay.jpg'
        )
        blended_pred.convert('RGB').save(save_pred_mask_img_path)

        # Create blended image of original image and ground truth mask
        blended_gt = self.blend_image_with_mask(original_image, gt_mask_img, mask_color=(0, 255, 0, 100))
        # Save blended image with ground truth mask
        save_gt_mask_img_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + f'_{track_id}_gt_mask_overlay.jpg'
        )
        blended_gt.convert('RGB').save(save_gt_mask_img_path)
        

def get_dataset(config, dataset_type="train"):
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
                    augment=True,
                    enable_negative_sample=True,
                    max_length_for_validate=val_length if dataset_type == "val" else None
                ))
            elif dataset_name in ["openimagev7", "coco", "samacoco"]:
                annotation_file = dataset_config.get("annotation_file", None)
                datasets.append(COCOPersonDataset(
                    images_dir=dataset_path,
                    annotation_file=annotation_file,
                    use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"], 
                    augment= True,
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
                augment=True,
                enable_negative_sample=True,
                max_length_for_validate=val_length if dataset_type == "val" else None
            )
        elif dataset_name in ["openimagev7", "coco", "samacoco"]:
            dataset = COCOPersonDataset(
                images_dir=dataset_path,
                annotation_file=annotation_file,
                use_SAM2_transform=config["train"]["transfer_image_in_dataset_use_sam2"],
                augment=True,
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
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        # DDP 模式
        print(f"!!!!!!!!!!!!!!!!!!!Choose DDP mode,world_size{os.environ['WORLD_SIZE']}!!!!!!!!!!!!!!!!!!!")
        # dist.init_process_group(backend="nccl", init_method="env://")
        dist.init_process_group(
            backend="nccl", 
            init_method="tcp://10.16.64.8:29500",  # 主节点的 IP 和端口
            world_size=int(os.environ['WORLD_SIZE']),
            rank=int(os.environ['RANK'])
        )
        dist.barrier()
        args.local_rank = dist.get_rank() # args.local_rank有问题,多机不能这样做
        torch.cuda.set_device(args.local_rank)#理论上torchrun 自动分配
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
    train_dataset = get_dataset(config, dataset_type="train")
    val_dataset = get_dataset(config, dataset_type="val")

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
