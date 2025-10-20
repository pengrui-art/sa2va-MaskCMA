import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image  # for type hints

# Ensure repo root on sys.path for 'projects.' absolute imports when running directly
_CUR_FILE = os.path.abspath(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR_FILE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Reuse utilities
from projects.llava_sam2.gradio.app_utils import (
    show_mask_pred,
    show_mask_pred_video,
    process_markdown,
    markdown_default,
    preprocess_video,
    image2video_and_save,
)
import tempfile


def make_description(title: str) -> str:
    return f"""
# {title}

支持功能:
- 图像描述 / VQA (VQA/Caption)
- 基于文本的图像目标分割 (Segment by Text)
- 多轮对话 (Chat) 与可选的分割输出

提示:
1. 如果是第一次对话或任务需要视觉信息, 会自动在输入前添加 `<image>` 标记。
2. 分割任务会尝试解析答案中的 `prediction_masks` 并叠加显示。
3. 多轮对话模式会保持上下文 (Chat)。切换任务或点击 "清空会话" 可重置。
"""


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)

# ----------------------------- Argument Parsing ----------------------------- #


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sa2VA Unified Gradio App")
    parser.add_argument(
        "--hf_path",
        default="/data1/pengrui/work_space/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568",
        help="Path to HF exported Sa2VA model (default: Sa2VA 1B HF at work_dirs)",
    )
    parser.add_argument(
        "--title",
        default="Sa2VA Unified Demo",
        help="Title shown at the top of the Gradio app",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run the model on, e.g. 'cuda:0', 'cuda:1', or 'cpu'",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(TORCH_DTYPE_MAP.keys()),
        help="Torch dtype for loading",
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable public sharing (Gradio)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the Gradio server on (e.g., 9000).",
    )
    return parser.parse_args(argv)


# ----------------------------- Model Loading ----------------------------- #


def _resolve_device(device_str: str) -> str:
    """Resolve device string to a valid torch device string.

    Accepts 'cpu', 'cuda', 'cuda:idx', or a bare index like '1' (treated as 'cuda:1').
    Falls back to 'cpu' if CUDA not available or index is invalid.
    """
    dev = device_str.strip().lower()
    if dev.isdigit():
        dev = f"cuda:{dev}"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available. Falling back to CPU.")
            return "cpu"
        # If specific index provided, validate bounds
        if ":" in dev:
            try:
                idx = int(dev.split(":", 1)[1])
                if idx < 0 or idx >= torch.cuda.device_count():
                    print(
                        f"[WARN] Requested device {dev} out of range. Falling back to cuda:0."
                    )
                    return "cuda:0"
            except ValueError:
                print(f"[WARN] Unrecognized CUDA device '{device_str}'. Using cuda:0.")
                return "cuda:0"
        return dev
    return "cpu"


def load_model(model_path: str, dtype: str, device: str):
    device = _resolve_device(device)
    torch_dtype = TORCH_DTYPE_MAP[dtype]
    # Detect FlashAttention availability without importing the module
    try:
        import importlib.util

        use_flash_attn = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        use_flash_attn = False

    # Set the active CUDA device so that any internal .cuda() uses the intended GPU
    if device.startswith("cuda"):
        try:
            idx = int(device.split(":", 1)[1])
            torch.cuda.set_device(idx)
        except Exception as e:
            print(f"[WARN] Unable to set CUDA device {device}: {e}")

    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    # Prefer default device routing for future tensor creations (PyTorch >= 2.1)
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device(device)
    except Exception as e:
        print(f"[WARN] set_default_device failed on {device}: {e}")
    # Force slow tokenizer to avoid json incompatibility with fast tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


# ----------------------------- Networking Helpers ------------------------- #


def _disable_external_calls_for_offline():
    """Best-effort switches to prevent Gradio/HF from making outbound requests.

    Useful in offline or proxy-restricted environments where outbound HTTP(s)
    may hang or time out during demo.launch().
    """
    try:
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # Ensure localhost bypasses proxies
        existing_no_proxy = os.environ.get("NO_PROXY", "")
        additions = ["127.0.0.1", "localhost"]
        parts = set([p.strip() for p in existing_no_proxy.split(",") if p.strip()])
        parts.update(additions)
        os.environ["NO_PROXY"] = ",".join(parts)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
    except Exception as _:
        pass


def _safe_launch(demo: gr.Blocks, share: bool, server_port: int | None = None, server_name: str = "127.0.0.1"):
    """Launch Gradio with a fallback that avoids external calls on failure."""
    try:
        # Bind to localhost explicitly; avoid DNS lookups
        demo.launch(share=share, server_name=server_name, server_port=server_port)
    except Exception as e:
        print(f"[WARN] Gradio launch failed: {e}. Retrying without share/analytics…")
        _disable_external_calls_for_offline()
        demo.launch(share=False, server_name=server_name, server_port=server_port)


# ----------------------------- Core Inference ------------------------------ #


def build_input_dict(
    task: str,
    image: Optional[Image.Image],
    video_frames: Optional[List[Image.Image]],
    text: str,
    past_text: str,
    tokenizer,
) -> Dict[str, Any]:
    # auto prepend <image> if first turn
    if past_text == "" and "<image>" not in text:
        text = "<image>" + text
    input_dict = {
        "text": text,
        "past_text": past_text,
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    if image is not None:
        input_dict["image"] = image
    if video_frames is not None:
        input_dict["video"] = video_frames
    return input_dict


def overlay_masks(base_image, prediction_masks):
    if not prediction_masks:
        return base_image, []
    try:
        overlay_img, colors = show_mask_pred(base_image, prediction_masks)
        return overlay_img, colors
    except Exception as e:
        print("[WARN] Mask overlay failed:", e)
        return base_image, []


def clean_generation_text(text: str) -> str:
    """Remove undesired special tokens from generation outputs."""
    if not isinstance(text, str):
        return text
    return text.replace("<|im_end|>", "").strip()


# ----------------------------- Gradio Handlers ----------------------------- #


def infer_handler(
    media_file,
    user_text,
    state: Dict[str, Any],
    chatbot: List[Tuple[str, str]],
):
    """Unified inference handler.

    state keys:
      past_text: internal model conversation serialization
      turns: int conversation turns
    """
    if state is None:
        state = {"past_text": "", "turns": 0}

    # Determine media type (image or video) based on uploaded file
    image = None
    video_path = None
    if media_file is not None:
        vp = media_file
        if isinstance(vp, dict):
            vp = vp.get("name") or vp.get("path") or vp.get("data")
        if isinstance(vp, str) and vp:
            lower = vp.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                try:
                    from PIL import Image as PILImage

                    image = PILImage.open(vp).convert("RGB")
                except Exception as e:
                    err = f"读取图像失败: {e}"
                    print(err)
                    return None, None, chatbot, process_markdown(err, []), state
            elif lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_path = vp
            else:
                return (
                    None,
                    None,
                    chatbot,
                    process_markdown("不支持的文件类型，请上传图像或视频。", []),
                    state,
                )
        else:
            return (
                None,
                None,
                chatbot,
                process_markdown("无效的文件，请重新上传图像或视频。", []),
                state,
            )
    else:
        # No media uploaded; if first turn, require media
        if state.get("past_text", "") == "":
            return (
                None,
                None,
                chatbot,
                process_markdown("请先上传图像或视频。", []),
                state,
            )

    text = user_text.strip()
    if not text:
        # Always return 5 outputs in the expected order
        return (
            (image if image is not None else None),
            (None if image is not None else None),
            chatbot,
            process_markdown("请输入指令。", []),
            state,
        )

    # Optionally encourage segmentation phrasing if user hints segmentation
    # (Removed explicit task selection; rely on model behavior.)

    video_frames = None
    if video_path is not None:
        try:
            # Normalize gradio video input (can be str path or dict with 'name'/'path')
            vp = video_path
            if not isinstance(vp, str) or not vp:
                raise ValueError("无效的视频路径")
            video_frames = preprocess_video(vp, text)
        except Exception as e:
            err = f"视频预处理失败: {e}"
            print(err)
            return None, None, chatbot, process_markdown(err, []), state

    input_dict = build_input_dict(
        task="Chat",  # default task context; model may infer internally
        image=image if image is not None else None,
        video_frames=video_frames,
        text=text,
        past_text=state["past_text"],
        tokenizer=tokenizer,
    )

    # Ensure CUDA context matches the model's device to avoid cross-device tensors
    try:
        model_device = str(next(sa2va_model.parameters()).device)
        if model_device.startswith("cuda"):
            try:
                torch.cuda.set_device(int(model_device.split(":", 1)[1]))
            except Exception:
                pass
        ret = sa2va_model.predict_forward(**input_dict)
    except Exception as e:
        err = f"推理失败: {e}"
        print(err)
        return (
            (image if image is not None else None),
            (None if image is not None else None),
            chatbot,
            process_markdown(err, []),
            state,
        )

    # Update internal past_text for multi-turn
    state["past_text"] = ret.get("past_text", state["past_text"])  # safe
    state["turns"] += 1

    prediction = ret.get("prediction", "(No prediction text)").strip()
    prediction = clean_generation_text(prediction)
    masks = ret.get("prediction_masks", [])

    # Only overlay for segmentation-related tasks or if masks exist
    # Default to original media when not overlaying
    final_image = image if image is not None else None
    final_video_path = video_path if video_path is not None else None
    color_list = []
    if masks:
        if image is not None:
            final_image, color_list = overlay_masks(image, masks)
        else:
            try:
                overlaid_frames, color_list = show_mask_pred_video(video_frames, masks)
                # Save to temp video
                tmp_dir = tempfile.mkdtemp(prefix="sa2va_")
                final_video_path = os.path.join(tmp_dir, "result.mp4")
                image2video_and_save(overlaid_frames, final_video_path)
            except Exception as e:
                print("[WARN] Video overlay failed:", e)

    md_answer = process_markdown(prediction, color_list)
    chatbot.append((user_text, prediction))

    return (
        (final_image if image is not None else None),
        (final_video_path if video_path is not None else None),
        chatbot,
        md_answer,
        state,
    )


def preview_media(media_file):
    """Preview uploaded image or video in the output area before inference."""
    if media_file is None:
        return None, None
    vp = media_file
    if isinstance(vp, dict):
        vp = vp.get("name") or vp.get("path") or vp.get("data")
    if not isinstance(vp, str) or not vp:
        return None, None
    lower = vp.lower()
    try:
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            from PIL import Image as PILImage

            img = PILImage.open(vp).convert("RGB")
            return img, None
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return None, vp
    except Exception as e:
        print(f"[WARN] 预览失败: {e}")
    return None, None


def clear_history(state, chatbot):
    state = {"past_text": "", "turns": 0}
    return state, []


# ----------------------------- Launch UI ---------------------------------- #


def build_interface(title: str):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(make_description(title))

        with gr.Row():
            with gr.Column(scale=1):
                media_file = gr.File(
                    label="上传图像或视频 (Image or Video)", file_count="single"
                )
                user_text = gr.Textbox(
                    lines=2,
                    label="输入 / 指令 (Prompt)",
                    placeholder="例如：'Describe the image' 或者 'Please segment the cat'",
                )
                run_btn = gr.Button("提交 / Submit", variant="primary")
                clear_btn = gr.Button("清空会话 / Clear")
            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="输出图像 (Result)")
                output_video = gr.Video(label="输出视频 (Result)")
                answer_md = gr.Markdown()
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="对话 (Chat)")

        state = gr.State({"past_text": "", "turns": 0})

        # Show media preview as soon as a file is uploaded
        media_file.change(
            preview_media,
            inputs=[media_file],
            outputs=[output_image, output_video],
        )

        run_btn.click(
            infer_handler,
            inputs=[media_file, user_text, state, chatbot],
            outputs=[output_image, output_video, chatbot, answer_md, state],
        )
        clear_btn.click(
            clear_history, inputs=[state, chatbot], outputs=[state, chatbot]
        )

    return demo


# ----------------------------- Main --------------------------------------- #
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sa2va_model, tokenizer = load_model(args.hf_path, args.dtype, args.device)
    demo = build_interface(args.title)
    demo.queue()
    _safe_launch(demo, share=args.share, server_port=args.port)
