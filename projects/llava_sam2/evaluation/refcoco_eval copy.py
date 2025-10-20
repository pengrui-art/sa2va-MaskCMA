import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    GenerationConfig,
)
from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset


def parse_args():
    parser = argparse.ArgumentParser(description="RefCocoSeg")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_ATTRIBUTES.keys(),
        default="refcoco",
        help="Specify a ref dataset",
    )
    parser.add_argument("--split", default="val", help="Specify a split")
    parser.add_argument(
        "--image-folder",
        default=os.environ.get("COCO2014_TRAIN", None),
        help="Path to COCO2014 train images folder. Can also set COCO2014_TRAIN env var.",
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("REFCOCO_DATA", None),
        help="Path to RefCOCO dataset (refer-style folder). Can also set REFCOCO_DATA env var.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("WORK_DIR", "./work_dirs"),
        help="Directory to save evaluation outputs/metrics.",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="Temporary directory for distributed result collection.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    # NOTE: downstream REFER api expects folder name "refcoco+" (with plus sign)
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco_plus"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
}

# Defaults retained for backward compatibility if not provided via flags/env.
DEFAULT_IMAGE_FOLDER = "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/glamm_data/images/coco2014/train2014/"
DEFAULT_DATA_PATH = "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/"


def main():
    args = parse_args()

    if args.launcher != "none":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
        # Prefer LOCAL_RANK for device mapping if available
        local_rank_env = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank_env)
    else:
        rank = 0
        world_size = 1

    # Avoid HF logging path that tries to default-construct custom configs (Sa2VA)
    def _safe_config_repr(self):
        try:
            name = self.__class__.__name__
            model_type = getattr(self, "model_type", None)
            return f"{name}(model_type={model_type})" if model_type else name
        except Exception:
            return self.__class__.__name__

    try:
        _HFPretrainedConfig.__repr__ = _safe_config_repr  # type: ignore[attr-defined]
    except Exception:
        pass

    # build model
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    image_folder = args.image_folder or DEFAULT_IMAGE_FOLDER
    data_path = args.data_path or DEFAULT_DATA_PATH

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Image folder not found: {image_folder}. Use --image-folder or set COCO2014_TRAIN env var."
        )
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data path not found: {data_path}. Use --data-path or set REFCOCO_DATA env var."
        )

    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info["dataset_name"],
        data_path=data_path,
        split=args.split,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {
            "img_id": data_batch["img_id"],
            "gt_masks": data_batch["gt_masks"],
        }
        prediction["gt_masks"] = mask_to_rle(prediction["gt_masks"].cpu().numpy())
        del data_batch["img_id"], data_batch["gt_masks"]

        texts = data_batch["text"]
        del data_batch["text"]
        pred_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch["text"] = text
            with torch.no_grad():
                pred_mask = model.predict_forward(**_data_batch, tokenizer=tokenizer)[
                    "prediction_masks"
                ]
            if len(pred_mask) == 0:
                # give a zero mask
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0]
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)

        prediction.update({"prediction_masks": pred_masks})
        results.append(prediction)

    tmpdir = args.tmpdir or (
        "./dist_test_temp_res_"
        + args.dataset
        + args.split
        + args.model_path.replace("/", "").replace(".", "")
    )
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results, args.work_dir)
        print(metric)

    # Clean distributed resources to avoid NCCL warnings on exit
    if (
        args.launcher != "none"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        try:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        except TypeError:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


if __name__ == "__main__":
    main()
