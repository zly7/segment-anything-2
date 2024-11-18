# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.person_classifier_mlp = MLP(transformer_dim, transformer_dim, 1, 3)
        self.person_classifier_token = nn.Embedding(1, transformer_dim)
        
        #res-iou
        self.res_iou_mlp = MLP(transformer_dim*2, transformer_dim, 1, 3)

        # HQ-SAM parameters #hq
        self.hq_token = nn.Embedding(1, transformer_dim)  # HQ-Output-Token #hq
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)  #hq
        self.num_mask_tokens = self.num_mask_tokens + 1  

        # Convolutional layers for obtaining HQ-Feature #hq 反卷积的步长就是扩大的倍数,还有一个是通道数
        self.compress_feature_2 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim //2),  
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 8, kernel_size=2, stride=2),  
        )  

        
        self.compress_feature_1 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim , transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim //2),  
            nn.GELU(),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 8, 3, 1, 1),
        )  

        self.compress_feature_0 = nn.Sequential( # 这里感受野就是3x3,这里是不是要大一点的感受野存疑，但是我感觉主要就是要针对比较小的物体的预测
            nn.Conv2d(transformer_dim, transformer_dim // 2, 3, 1, 1),
            LayerNorm2d(transformer_dim // 2),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 8, 3, 1, 1),
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        use_hq: bool = False,  #hq
        use_res_iou: bool = False,  
        direct_high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, person_classfication_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            use_hq = use_hq,
            use_res_iou = use_res_iou,
            direct_high_res_features= direct_high_res_features,
        )

        # HQ-SAM mask selection logic 
        if multimask_output:
            # Mask with highest score 
            mask_slice = slice(1, self.num_mask_tokens - 1)  
            iou_pred_slice = iou_pred[:, mask_slice]  
            iou_pred_max, max_iou_idx = torch.max(iou_pred_slice, dim=1)  
            iou_pred = iou_pred_max.unsqueeze(1)  
            masks_multi = masks[:, mask_slice, :, :]  
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)  
        else:
            mask_slice = slice(0, 1)  
            iou_pred = iou_pred[:, mask_slice]  
            masks_sam = masks[:, mask_slice]  
            
        if use_hq:
            masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]  #hq
            masks = masks_sam + masks_hq  #hq
        else:
            masks = masks_sam

        return masks, iou_pred, mask_tokens_out, object_score_logits, person_classfication_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, #[b,256,64,64]
        image_pe: torch.Tensor, #[1,256,64,64]
        sparse_prompt_embeddings: torch.Tensor,#[b,2,256]
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        use_hq = False,  #[b,256,64,64]
        use_res_iou = False,
        direct_high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details. 内部函数没有其它地方调用"""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            if use_hq is True:  
                output_tokens = torch.cat(
                    [
                        self.obj_score_token.weight,
                        self.iou_token.weight,
                        self.mask_tokens.weight,
                        self.hq_token.weight,  #hq
                        self.person_classifier_token.weight,
                    ],
                    dim=0,
                )  #hq
            else:
                output_tokens = torch.cat(
                    [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight, self.person_classifier_token.weight],
                    dim=0,
                )
            s = 1
        else:
            if use_hq is True:
                output_tokens = torch.cat(
                    [self.iou_token.weight, self.mask_tokens.weight, self.hq_token.weight, self.person_classifier_token.weight], dim=0
                )  #hq
            else:
                output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.person_classifier_token.weight], dim=0)

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :] # hq算是mask_token
        person_class_token = hs[:, s + 1 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features: # always true
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) # updascaled_embedding [b,32,256,256] 转置卷积,32=256/8,对应transformer_dim // 8

        # HQ-SAM upscaled embeddings #hq
        upscaled_embedding_sam = upscaled_embedding  #hq
        if use_hq is True:  #hq
            d_feat_s0, d_feat_s1 = direct_high_res_features
            upscaled_embedding_hq = self.compress_feature_2(src) + self.compress_feature_1(d_feat_s1) + self.compress_feature_0(d_feat_s0)  #hq

        # Generate mask predictions
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            else:  
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))  #hq
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape 

        masks_sam = (hyper_in[:, : self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c, h * w)).view(
            b, -1, h, w
        ) 
        if use_hq is True:  #hq
            masks_hq = (hyper_in[:, self.num_mask_tokens - 1 :] @ upscaled_embedding_hq.view(b, c, h * w)).view(
                b, -1, h, w
            )  
            masks = torch.cat([masks_sam, masks_hq], dim=1)
        else:
            masks = masks_sam 

        # Generate mask quality predictions

        iou_pred = self.iou_prediction_head(iou_token_out)
        if use_res_iou:
            res_iou_token = torch.cat([iou_token_out.unsqueeze(1).repeat(1,4,1), mask_tokens_out[:,:4]], dim=-1)
            res_iou = self.res_iou_mlp(res_iou_token)
            iou_pred = iou_pred + res_iou.squeeze(-1)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        person_classfication_logits = self.person_classifier_mlp(person_class_token)
        return masks, iou_pred, mask_tokens_out, object_score_logits, person_classfication_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
