import logging

from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.logging.logger import print_log
from huggingface_hub import hf_hub_download

HF_HUB_PREFIX = "hf-hub:"


def load_checkpoint_with_prefix(
    filename, prefix=None, map_location="cpu", logger="current"
):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.
        logger: logger

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith("hf-hub:"):
        model_id = filename[len(HF_HUB_PREFIX) :]
        filename = hf_hub_download(model_id, "pytorch_model.bin")

    checkpoint = CheckpointLoader.load_checkpoint(
        filename, map_location=map_location, logger=logger
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not prefix:
        return state_dict
    if not prefix.endswith("."):
        prefix += "."
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f"{prefix} is not in the pretrained model"
    return state_dict


def load_state_dict_to_model(
    model,
    state_dict,
    logger="current",
    allow_unexpected: bool = True,
    allow_missing: bool = False,
    filter_prefixes=None,
):
    """Load a state_dict into model with relaxed options.

    Args:
        model: nn.Module to load weights into.
        state_dict (dict): checkpoint state dict.
        logger: logger name or object.
        allow_unexpected (bool): If True, ignore unexpected keys (only warn).
        allow_missing (bool): If True, ignore missing keys (only warn).
        filter_prefixes (list[str]|None): If provided, any checkpoint key that
            starts with one of these prefixes will be dropped before loading.
            Useful to drop obsolete layers (e.g. high-res decoder when disabled).
    """
    if filter_prefixes:
        filtered = {}
        dropped = []
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in filter_prefixes):
                dropped.append(k)
            else:
                filtered[k] = v
        if dropped:
            print_log(
                f"Filtered {len(dropped)} keys due to prefixes {filter_prefixes}: first 5 -> {dropped[:5]}",
                logger=logger,
                level=logging.WARNING,
            )
        state_dict = filtered

    # Use strict=False first so we can control error handling ourselves
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys

    if missing_keys and not allow_missing:
        print_log(
            f"Missing keys ({len(missing_keys)}): first 10 -> {missing_keys[:10]}",
            logger=logger,
            level=logging.ERROR,
        )
        raise RuntimeError(
            "Missing keys when loading checkpoint (set allow_missing=True to ignore)"
        )
    if unexpected_keys and not allow_unexpected:
        print_log(
            f"Unexpected keys ({len(unexpected_keys)}): first 10 -> {unexpected_keys[:10]}",
            logger=logger,
            level=logging.ERROR,
        )
        raise RuntimeError(
            "Unexpected keys when loading checkpoint (set allow_unexpected=True to ignore)"
        )

    # Log summaries
    if missing_keys:
        print_log(
            f"Ignored {len(missing_keys)} missing keys (model layers newly initialized).",
            logger=logger,
            level=logging.WARNING,
        )
    if unexpected_keys:
        print_log(
            f"Ignored {len(unexpected_keys)} unexpected keys (not used by current model).",
            logger=logger,
            level=logging.WARNING,
        )
    print_log("Loaded checkpoint successfully (relaxed mode)", logger=logger)
