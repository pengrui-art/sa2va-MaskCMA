import argparse
import json
import os

import mmengine
import numpy as np
from PIL import Image

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig

from projects.llava_sam2.evaluation.dataset import RefVOSDataset
from projects.llava_sam2.evaluation.utils import (
    _init_dist_pytorch,
    _init_dist_slurm,
    get_dist_info,
    get_rank,
    collect_results_cpu,
)

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    vid_id = item["video_id"]
    exp_id = item["exp_id"]
    save_path = os.path.join(work_dir, "Annotations", vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert("L")
        file_name = item["frames"][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    "DAVIS": {
        "data_root": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/",
        "image_folder": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/valid/JPEGImages/",
        "expression_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/meta_expressions/valid/meta_expressions.json",
        "mask_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/valid/mask_dict.pkl",
    },
    "MEVIS": {
        "data_root": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid/",
        "image_folder": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid/JPEGImages",
        "expression_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid/meta_expressions.json",
        "mask_file": None,
    },
    "MEVIS_U": {
        "data_root": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/",
        "image_folder": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/JPEGImages",
        "expression_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/meta_expressions.json",
        "mask_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/mask_dict.json",
    },
    "REFYTVOS": {
        "data_root": "data/video_datas/rvos/",
        "image_folder": "data/video_datas/rvos/valid/JPEGImages/",
        "expression_file": "data/video_datas/rvos/meta_expressions/valid/meta_expressions.json",
        "mask_file": None,
    },
    "REVOS": {
        "data_root": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/",
        "image_folder": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/",
        "expression_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/meta_expressions_valid_.json",
        "mask_file": "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/mask_dict.json",
    },
}


# Hotfix: some custom configs (like Sa2VA) have non-trivial __init__ signatures.
# When Transformers logs the config, it calls __repr__, which by default builds a diff
# against a default-constructed instance (self.__class__()), potentially raising errors
# if the class cannot be instantiated with no args. We override __repr__ to be minimal
# and avoid constructing a default instance.
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


def parse_args():
    parser = argparse.ArgumentParser(description="RefVOS")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_INFO.keys(),
        default="MEVIS",
        help="Specify a dataset",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)  # dummy
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


if __name__ == "__main__":
    args = parse_args()

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = "work_dirs/foobar"

    if args.launcher == "none":
        rank = 0
        world_size = 1
    elif args.launcher == "pytorch":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
    elif args.launcher == "slurm":
        _init_dist_slurm("nccl")
        rank, world_size = get_dist_info()

    # Ensure each process uses the correct GPU to avoid NCCL device-id warnings
    local_rank_env = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank_env)

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
    dataset_info = DATASETS_INFO[args.dataset]

    dataset = RefVOSDataset(
        image_folder=dataset_info["image_folder"],
        expression_file=dataset_info["expression_file"],
        mask_file=dataset_info["mask_file"],
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    error_count = 0
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            try:
                result = model.predict_forward(
                    video=item["images"],
                    text=item["text_prompt"],
                    tokenizer=tokenizer,
                )
            except AssertionError as e:
                # Some model builds assert when no frame gets selected (e.g., `selected.sum()==0`).
                # Fall back to an empty prediction for this sample so evaluation can continue.
                if error_count < 10:
                    print(
                        f"[Rank {rank}] Warning: predict_forward assertion failed for video {item['video_id']} exp {item['exp_id']}: {e}. Fallback to empty masks."
                    )
                error_count += 1
                result = {
                    "prediction": "",
                    "prediction_masks": [],
                }
            except Exception as e:  # catch-all to avoid distributed crash
                if error_count < 10:
                    print(
                        f"[Rank {rank}] Warning: predict_forward error for video {item['video_id']} exp {item['exp_id']}: {repr(e)}. Fallback to empty masks."
                    )
                error_count += 1
                result = {
                    "prediction": "",
                    "prediction_masks": [],
                }

        text_idx = 0
        text_prediction = result["prediction"]
        if len(result["prediction_masks"]) > 0:
            mask_prediction = result["prediction_masks"][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros(
                (item["length"], item["ori_height"], item["ori_width"]), dtype=np.uint8
            )

        if args.submit:
            async_func(
                executor,
                mask_save,
                item=item,
                mask_prediction=mask_prediction,
                work_dir=work_dir,
            )
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            "index": item["index"],
            "video_id": item["video_id"],
            "exp_id": item["exp_id"],
            "text_prediction": text_prediction,
            "frames": item["frames"],
            "exp": item["text_prompt"],
            "prediction_masks": encoded_mask,
        }
        results.append(result)

    executor.shutdown(wait=True)
    print(f"[Rank {rank}] : Finished.")

    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item["video_id"]
                exp_id = item["exp_id"]
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f"{work_dir}/results.json", "w"))

    if (
        args.launcher != "none"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        # Synchronize and cleanly tear down the process group to avoid resource leak warnings
        try:
            torch.distributed.barrier(device_ids=[local_rank_env])
        except TypeError:
            # Older torch versions do not accept device_ids
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if rank == 0:
        print("Done")
