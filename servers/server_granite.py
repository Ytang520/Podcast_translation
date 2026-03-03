"""
Granite Speech 3 Transcription Server (FastAPI).

Supports two backends:
  - transformers : loads the model directly with HuggingFace transformers
  - vllm         : forwards requests to a running vLLM instance

Usage (transformers backend — default):
    python servers/server_granite.py --backend transformers \
        --model_name /root/autodl-tmp/model/granite_speech_3 --port 8001

Usage (vLLM backend — launch vLLM first, then this server):
    # Terminal 1:
    vllm serve /root/autodl-tmp/model/granite_speech_3 \
        --api-key token-abc123 --max-model-len 2048 \
        --enable-lora \
        --lora-modules speech=/root/autodl-tmp/model/granite_speech_3 \
        --max-lora-rank 64

    # Terminal 2:
    python servers/server_granite.py --backend vllm \
        --vllm_url http://localhost:8000/v1 --port 8001
"""
import os

# These MUST be set BEFORE importing transformers / huggingface_hub
# so the libraries never attempt any network calls.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import base64
import argparse
import shutil
import json as _json

from fastapi import FastAPI, UploadFile, File, HTTPException

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Granite Speech Transcription Server")
parser.add_argument(
    "--backend",
    type=str,
    choices=["transformers", "vllm"],
    default="transformers",
    help="Inference backend: 'transformers' (direct) or 'vllm' (via vLLM server) "
         "(default: transformers)",
)
# --- transformers backend options ---
parser.add_argument(
    "--model_name",
    type=str,
    default="/root/autodl-tmp/model/granite_speech_3",
    help="[transformers] Local path to Granite Speech model "
         "(default: /root/autodl-tmp/model/granite_speech_3)",
)
# --- vLLM backend options ---
parser.add_argument(
    "--vllm_url",
    type=str,
    default="http://localhost:8000/v1",
    help="[vllm] vLLM OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
)
parser.add_argument(
    "--vllm_api_key",
    type=str,
    default="token-abc123",
    help="[vllm] API key for vLLM server (default: token-abc123)",
)
parser.add_argument(
    "--lora_model",
    type=str,
    default="speech",
    help="[vllm] LoRA model name registered in vLLM (default: speech)",
)
# --- shared ---
parser.add_argument(
    "--port",
    type=int,
    default=8001,
    help="Port to listen on (default: 8001)",
)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Granite Speech Transcription Server")

# ---------------------------------------------------------------------------
# Backend initialisation
# ---------------------------------------------------------------------------
BACKEND = args.backend

if BACKEND == "transformers":
    import torch
    import torchaudio
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    TARGET_SAMPLE_RATE = 16000
    SYSTEM_PROMPT = (
        "Knowledge Cutoff Date: April 2024.\n"
        "Today's Date: April 9, 2025.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )
    USER_PROMPT = "<|audio|>can you transcribe the speech into a written format?"

    # Patch adapter_config.json so base_model_name_or_path points locally
    adapter_cfg_path = os.path.join(args.model_name, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as _f:
            adapter_cfg = _json.load(_f)
        old_base = adapter_cfg.get("base_model_name_or_path", "")
        local_abs = os.path.abspath(args.model_name)
        if old_base != local_abs:
            print(f"Patching adapter_config.json: base_model_name_or_path")
            print(f"  {old_base!r}  ->  {local_abs!r}")
            adapter_cfg["base_model_name_or_path"] = local_abs
            with open(adapter_cfg_path, "w", encoding="utf-8") as _f:
                _json.dump(adapter_cfg, _f, indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[transformers] Initializing Granite Speech from '{args.model_name}' on {device} ...")

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True
        )
        tokenizer = processor.tokenizer
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print(f"[transformers] Model loaded successfully on {device}.")
    except Exception as e:
        print(f"[transformers] Error loading model: {e}")
        raise e

elif BACKEND == "vllm":
    from openai import OpenAI

    client = OpenAI(api_key=args.vllm_api_key, base_url=args.vllm_url)
    print(f"[vllm] Using vLLM at {args.vllm_url}, LoRA model: {args.lora_model}")

    # MIME type mapping for base64 audio
    _MIME_MAP = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
    }

    def _audio_mime(filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        return _MIME_MAP.get(ext, "audio/mpeg")


# =====================================================================
# Transcription logic per backend
# =====================================================================

def _transcribe_transformers(temp_filename: str) -> str:
    """Transcribe using the locally loaded transformers model."""
    import subprocess

    temp_wav = "temp_converted.wav"
    try:
        # Convert to 16kHz mono WAV if not already WAV
        if not temp_filename.lower().endswith(".wav"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_filename,
                 "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", temp_wav],
                check=True, capture_output=True,
            )
            audio_path = temp_wav
        else:
            audio_path = temp_filename

        wav, sr = torchaudio.load(audio_path, normalize=True)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample
        if sr != TARGET_SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(wav)

        # Chat prompt
        chat = [
            dict(role="system", content=SYSTEM_PROMPT),
            dict(role="user", content=USER_PROMPT),
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
        model_outputs = model.generate(
            **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
        )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return output_text[0].upper()
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def _transcribe_vllm(temp_filename: str, original_filename: str) -> str:
    """Transcribe by forwarding base64-encoded audio to vLLM."""
    with open(temp_filename, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = _audio_mime(original_filename)

    completion = client.chat.completions.create(
        model=args.lora_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "can you transcribe the speech into a written format?",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:{mime_type};base64,{audio_base64}",
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=200,
    )

    text = completion.choices[0].message.content
    return text.upper() if text else ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health-check endpoint."""
    info = {"status": "ok", "backend": BACKEND}
    if BACKEND == "transformers":
        info["model"] = args.model_name
        info["device"] = device
    else:
        info["vllm_url"] = args.vllm_url
        info["lora_model"] = args.lora_model
    return info


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file using Granite Speech.

    Returns:
        JSON with filename and transcribed text.
    """
    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if BACKEND == "transformers":
            text = _transcribe_transformers(temp_filename)
        else:
            text = _transcribe_vllm(temp_filename, file.filename)

        return {
            "filename": file.filename,
            "text": text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
