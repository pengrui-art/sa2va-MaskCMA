import os.path

import torch

from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mmengine.model import BaseModule


from vlm.utils import load_checkpoint_with_prefix, load_state_dict_to_model

BASE_DIR = "/data1/pengrui/Model/facebook/sam2-hiera-large"


class SAM2TrainRunner(BaseModule):
    def __init__(
        self,
        cfg_path: str = "sam2_hiera_l.yaml",
        ckpt_path: str = "sam2_hiera_large.pt",
        hydra_overrides_extra=None,
        apply_postprocessing=True,
        enable_cma: bool = True,
        cma_num_heads: int = 8,
        # CMA advanced options (backward compatible by default)
        cma_dropout: float = 0.0,
        cma_use_multi_token: bool = False,
        cma_topk_tokens: int = 0,
        cma_use_film: bool = False,
        cma_film_dropout: float = 0.0,
        # Multi-scale CMA application
        cma_apply_on_highres: bool = False,
        cma_highres_levels: tuple = (-1,),
    ):
        super().__init__(init_cfg=None)

        import third_parts.sam2  # noqa: F401

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            "++model._target_=projects.llava_sam2.models.extension.SAM2Base",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)
        state_dict = load_checkpoint_with_prefix(os.path.join(BASE_DIR, ckpt_path))
        # If user disabled high-res features via hydra overrides, drop related weights to avoid clutter.
        filter_prefixes = []
        if not getattr(sam2_model, "use_high_res_features_in_sam", True):
            filter_prefixes.extend(
                [
                    "sam_mask_decoder.conv_s0",
                    "sam_mask_decoder.conv_s1",
                ]
            )
        load_state_dict_to_model(
            sam2_model,
            state_dict,
            allow_unexpected=True,
            allow_missing=False,
            filter_prefixes=filter_prefixes if filter_prefixes else None,
        )

        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)
        # internal flag to avoid spamming resize warning every iteration
        self._warned_feat_resize = False

        # Cross-modal attention (optional)
        self.enable_cma = enable_cma
        if self.enable_cma:
            try:
                from .cma import CrossModalAttention2D
            except Exception:
                # Fallback to relative import path if needed
                from projects.llava_sam2.models.cma import CrossModalAttention2D
            self.cma = CrossModalAttention2D(
                dim=self.hidden_dim,
                num_heads=cma_num_heads,
                dropout=cma_dropout,
                use_multi_token=cma_use_multi_token,
                topk_tokens=cma_topk_tokens,
                use_film=cma_use_film,
                film_dropout=cma_film_dropout,
            )
        else:
            self.cma = None
        # Store multi-scale flags
        self.cma_apply_on_highres = bool(cma_apply_on_highres)
        # normalize indices into a list
        if isinstance(cma_highres_levels, (list, tuple)):
            self.cma_highres_levels = list(cma_highres_levels)
        else:
            self.cma_highres_levels = [-1]

    def set_cma_warmup_scale(self, scale: float):
        """Externally control CMA residual strength (e.g., during LR warmup)."""
        if getattr(self, "cma", None) is not None and hasattr(
            self.cma, "set_warmup_scale"
        ):
            self.cma.set_warmup_scale(scale)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.0
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[
            :, None, None
        ]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[
            :, None, None
        ]
        image -= img_mean
        image /= img_std
        return image

    def inject_language_embd(
        self,
        sam_states,
        language_embd,
        nf_nobj=None,
        return_lowres_feat: bool = False,
        lowres_feat_stage: str = "post",
    ):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(
                sam_states["current_vision_feats"][:-1], sam_states["feat_sizes"][:-1]
            )
        ]

        B = sam_states["current_vision_feats"][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states["feat_sizes"][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = (
                sam_states["current_vision_feats"][-1] + self.sam2_model.no_mem_embed
            )
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError(
                "directly add no memory embedding is not implemented"
            )

        pre_cma_feat = pix_feat_with_mem

        # Optionally apply cross-modal attention to inject language guidance
        if self.enable_cma and self.cma is not None and language_embd is not None:
            # language_embd expected shapes:
            # - (nf, nobj, C) flattened later, or
            # - (B, N, C) or (B, C) for single frame
            # Here language_embd comes from upper pipeline as (B, N, C) or (B, C)
            if isinstance(language_embd, (list, tuple)):
                # Convert list of tensors [B x C] per object into (B, N, C)
                lang_feats = torch.stack(language_embd, dim=1)
            else:
                lang_feats = language_embd
            # Ensure dtype matches
            lang_feats = lang_feats.to(pix_feat_with_mem.dtype)
            # Low-res backbone feature (E x E)
            pix_feat_with_mem = self.cma(pix_feat_with_mem, lang_feats)
            # Optional: also inject on selected high-res feature maps
            if self.cma_apply_on_highres:
                for lvl in self.cma_highres_levels:
                    try:
                        idx = lvl if lvl >= 0 else (len(high_res_features) + lvl)
                        if 0 <= idx < len(high_res_features):
                            high_res_features[idx] = self.cma(
                                high_res_features[idx], lang_feats
                            )
                    except Exception:
                        # be robust: skip if shape/dtype mismatch occurs
                        pass

        # ---- Shape sanity check & adaptive fix ---------------------------------
        # SAM head expects (B, C, E, E) where E = self.sam2_model.sam_image_embedding_size
        expected_E = getattr(self.sam2_model, "sam_image_embedding_size", None)
        if expected_E is not None:
            h, w = pix_feat_with_mem.shape[-2:]
            if (h != expected_E) or (w != expected_E):
                rank0 = True
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    rank0 = torch.distributed.get_rank() == 0
                if rank0 and not self._warned_feat_resize:
                    print(
                        f"[SAM2TrainRunner][WARN] backbone feature spatial size {(h,w)} != expected ({expected_E},{expected_E}); interpolating (this warning shown once)."
                    )
                    self._warned_feat_resize = True
                pix_feat_with_mem = torch.nn.functional.interpolate(
                    pix_feat_with_mem,
                    size=(expected_E, expected_E),
                    mode="bilinear",
                    align_corners=False,
                )
                # If channel mismatch occurs it will still crash later with clearer msg
        # ------------------------------------------------------------------------
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                _,
            ) = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(
                    is_init_cond_frame=True, point_inputs=None
                ),
                # Inject language Embed if possible
                language_embd=language_embd,
            )
            # Expose the low-res feature that produced low_res_masks if requested.
            if return_lowres_feat:
                lowres_feat = (
                    pre_cma_feat if lowres_feat_stage == "pre" else pix_feat_with_mem
                )
            else:
                lowres_feat = None

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        if return_lowres_feat:
            return pred_masks, lowres_feat
        return pred_masks

    def get_sam2_embeddings(self, images, expand_size=1):
        # Step 1: inference the backbone with the images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)

        if expand_size > 1:
            # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = (
                    feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                )
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self.sam2_model._prepare_backbone_features(feats)
        )

        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }

    def forward(self, batch):
        raise NotImplementedError
