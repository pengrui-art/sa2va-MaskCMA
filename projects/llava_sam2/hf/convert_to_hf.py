import argparse
import copy
import os.path as osp
import torch
import pickle
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root (repository root) is on sys.path so that `projects.*`
# imports succeed no matter where this script is launched from.
# ---------------------------------------------------------------------------
_THIS_FILE = osp.abspath(__file__)
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_FILE, "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if os.environ.get("SA2VA_CONVERT_DEBUG") == "1":
    print("[DEBUG] Added project root to sys.path:", _PROJECT_ROOT)
    print("[DEBUG] sys.path head:", sys.path[:5])

from mmengine.dist import (
    collect_results,
    get_dist_info,
    get_rank,
    init_dist,
    master_only,
)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Sa2VA checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "config",
        help="Config file name or path (e.g. projects/llava_sam2/configs/sa2va_1b.py)",
    )
    # Keep backward compatibility with README that uses --pth-model
    parser.add_argument(
        "pth_model",
        nargs="?",
        help="Path to consolidated *.pth model file OR DeepSpeed folder (mp_rank_00_model_states.pt)",
    )
    parser.add_argument(
        "--pth-model",
        dest="pth_model_flag",
        help="(Alt) Path to consolidated *.pth model file OR DeepSpeed ZeRO folder",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./work_dirs/hf_model",
        help="Output directory to save HF model",
    )
    parser.add_argument(
        "--fill-base",
        action="store_true",
        help="If set, load base model from cfg.path and fill in missing vision/LLM weights (use when checkpoint only saved LoRA / partial deltas).",
    )
    args = parser.parse_args()

    # Prefer explicit flag if given
    if args.pth_model_flag:
        args.pth_model = args.pth_model_flag

    if not args.pth_model:
        parser.error(
            "You must supply a model path either as positional pth_model or via --pth-model"
        )
    return args


@master_only
def master_print(msg):
    print(msg)


def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    # ------------------------------------------------------------
    # Load checkpoint: support either (a) single consolidated *.pth
    # or (b) DeepSpeed ZeRO folder containing mp_rank_00_model_states.pt
    # ------------------------------------------------------------
    ckpt_path = args.pth_model
    state_dict = None

    if os.path.isdir(ckpt_path):
        mp_file = osp.join(ckpt_path, "mp_rank_00_model_states.pt")
        if not osp.exists(mp_file):
            raise FileNotFoundError(
                f"Detected a directory but mp_rank_00_model_states.pt not found in {ckpt_path}. "
                "If this is a DeepSpeed ZeRO checkpoint, please point to the folder containing mp_rank_00_model_states.pt."
            )
        master_print(f"Detected DeepSpeed ZeRO folder checkpoint: {ckpt_path}")
        # Try to read the mp_rank_00 file
        # PyTorch 2.6+ defaults weights_only=True which can break when extra objects are in the pickle
        try:
            mp_obj = torch.load(mp_file, map_location="cpu")
        except (pickle.UnpicklingError, RuntimeError) as e:
            master_print(
                "Primary torch.load failed due to safety guard, retrying with allowlisted globals / weights_only=False..."
            )
            # Try allowlisting mmengine ConfigDict first
            try:
                import torch.serialization as _ts
                from mmengine.config import ConfigDict as _MMCfgDict

                if hasattr(_ts, "add_safe_globals"):
                    _ts.add_safe_globals([_MMCfgDict])
                mp_obj = torch.load(mp_file, map_location="cpu")
            except Exception:
                # Final fallback: explicitly disable weights_only
                mp_obj = torch.load(mp_file, map_location="cpu", weights_only=False)
        # Common key name patterns in DeepSpeed checkpoints
        candidate_keys = ["module", "model", "state_dict"]
        for k in candidate_keys:
            if k in mp_obj and isinstance(mp_obj[k], dict):
                state_dict = mp_obj[k]
                master_print(
                    f'Loaded state dict from key "{k}" inside mp_rank_00_model_states.pt'
                )
                break
        if state_dict is None:
            # Fallback: maybe the whole object is already a state dict
            if all(isinstance(v, torch.Tensor) for v in mp_obj.values()):
                state_dict = mp_obj
            else:
                raise RuntimeError(
                    "Could not locate model state dict in mp_rank_00_model_states.pt. "
                    "Inspect the file structure manually."
                )
    else:
        backend = get_file_backend(ckpt_path)
        if isinstance(backend, PetrelBackend):
            from xtuner.utils.fileio import patch_fileio

            with patch_fileio():
                state_dict = guess_load_checkpoint(ckpt_path)
        else:
            state_dict = guess_load_checkpoint(ckpt_path)
        master_print(f"Loaded consolidated checkpoint file: {ckpt_path}")

    # Load into model (allow missing keys for safety; user can inspect warnings)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    master_print(
        f"Checkpoint load finished. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )
    if missing:
        master_print(f"  (First 10 missing) {missing[:10]}")
    if unexpected:
        master_print(f"  (First 10 unexpected) {unexpected[:10]}")
    if len(missing) > 500:
        master_print(
            "[WARNING] Large number of missing keys detected. This usually means you pointed to a single DeepSpeed rank shard (mp_rank_00) without first merging full weights.\n"
            "          Please run zero_to_fp32.py or enable stage3_gather_16bit_weights_on_model_save during training to create a consolidated .pth, then rerun conversion."
        )

    # Optional backfill from base pretrained model
    if args.fill_base or (len(missing) > 500):
        try:
            from transformers import AutoModel

            master_print(
                "Attempting to backfill missing weights from base model at cfg.path ..."
            )
            base_model = AutoModel.from_pretrained(cfg.path, trust_remote_code=True)
            base_state = base_model.state_dict()
            train_state = model.state_dict()
            missing_set = set(missing)
            fill_count = 0
            for bkey, bval in base_state.items():
                tgt_key = (
                    f"mllm.model.{bkey}"
                    if f"mllm.model.{bkey}" in train_state
                    else None
                )
                if tgt_key and tgt_key in missing_set:
                    with torch.no_grad():
                        param = train_state[tgt_key]
                        if param.shape == bval.shape:
                            param.copy_(bval.to(param.dtype))
                            fill_count += 1
            master_print(
                f"Backfill done. Filled {fill_count} / {len(missing)} missing keys from base model."
            )
            # Recompute missing after fill (not strictly necessary for export but informative)
            new_missing = [
                k
                for k in missing
                if k in missing_set
                and k in model.state_dict()
                and model.state_dict()[k].abs().sum() == 0
            ]
        except Exception as e:
            master_print(f"Backfill failed: {e}")

    # Merge LoRA weights into base modules (if present)
    if hasattr(model, "_merge_lora"):
        model._merge_lora()
    model.mllm.transfer_to_hf = True

    # Export FULL state dict (after merging) rather than model.all_state_dict().
    # Guard against custom overridden state_dict raising due to PEFT assumptions.
    try:
        full_state = model.state_dict()
    except AttributeError as e:
        master_print(
            f"model.state_dict() raised {e}; falling back to base nn.Module.state_dict()"
        )
        import torch.nn as nn

        full_state = nn.Module.state_dict(model)

    # Apply name mapping for HF compatibility
    name_map = {"mllm.model.": "", ".gamma": ".g_weight"}
    export_state = {}
    for k, v in full_state.items():
        nk = k
        for src, tgt in name_map.items():
            if src in nk:
                nk = nk.replace(src, tgt)
        export_state[nk] = v

    # build the hf format model
    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
    from transformers import AutoConfig

    def _manual_compose_config():
        """Compose a Sa2VAChatConfig from the already-built training model when
        direct from_pretrained() fails (e.g., base config missing 'architectures')."""
        vision_cfg = getattr(model.mllm.model.vision_model, "config", None)
        llm_cfg = getattr(model.mllm.model.language_model, "config", None)
        if vision_cfg is None or llm_cfg is None:
            raise RuntimeError(
                "Cannot extract vision / llm config from training model."
            )
        vision_dict = vision_cfg.to_dict()
        llm_dict = llm_cfg.to_dict()
        # Ensure architectures field exists for Sa2VAChatConfig selector
        if "architectures" not in llm_dict:
            llm_arch = model.mllm.model.language_model.__class__.__name__
            llm_dict["architectures"] = [llm_arch]
        composed = dict(
            vision_config=vision_dict,
            llm_config=llm_dict,
            template=getattr(cfg, "template", None),
            use_backbone_lora=0,
            use_llm_lora=0,
        )
        return composed

    try:
        internvl_config = Sa2VAChatConfig.from_pretrained(cfg.path)
        config_dict = internvl_config.to_dict()
    except Exception as e:
        master_print(
            f'Failed to load Sa2VAChatConfig.from_pretrained("{cfg.path}"): {e}\nFalling back to manual composition from current model components.'
        )
        config_dict = _manual_compose_config()

    # Insert / override fields
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_sa2va_chat.Sa2VAChatConfig",
        "AutoModel": "modeling_sa2va_chat.Sa2VAChatModel",
        "AutoModelForCausalLM": "modeling_sa2va_chat.Sa2VAChatModel",
    }
    # If nested llm_config present, set vocab_size; else treat top-level
    if "llm_config" in config_dict:
        config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    else:
        config_dict["vocab_size"] = len(model.tokenizer)
    # Ensure template present
    if "template" not in config_dict:
        config_dict["template"] = getattr(cfg, "template", None)
    # Validate template against PROMPT_TEMPLATE, fallback if needed
    try:
        from projects.llava_sam2.hf.models.templates import (
            PROMPT_TEMPLATE as _PROMPT_TBL,
        )

        tval = config_dict.get("template")
        if not tval or tval not in _PROMPT_TBL:
            master_print(
                f'Template "{tval}" invalid or missing, fallback to "qwen_chat"'
            )
            config_dict["template"] = "qwen_chat"
    except Exception as _e:
        master_print(f"Could not import PROMPT_TEMPLATE for validation: {_e}")

    sa2va_hf_config = Sa2VAChatConfig(**config_dict)
    hf_sa2va_model = Sa2VAChatModel(
        sa2va_hf_config,
        vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    missing_hf, unexpected_hf = hf_sa2va_model.load_state_dict(
        export_state, strict=False
    )
    master_print(
        f"HF load complete. Missing (HF expects but absent in export): {len(missing_hf)}, Unexpected (in export not used by HF): {len(unexpected_hf)}"
    )
    if missing_hf:
        master_print(f"  (First 10 missing) {missing_hf[:10]}")
    if unexpected_hf:
        master_print(f"  (First 10 unexpected) {unexpected_hf[:10]}")

    hf_sa2va_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    # copy the files
    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {args.save_path}")


if __name__ == "__main__":
    main()
