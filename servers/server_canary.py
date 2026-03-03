"""
NVIDIA Canary 1B v2 Transcription Server (FastAPI).

Loads NVIDIA Canary ASR model via NeMo and exposes a /transcribe API endpoint.

Usage:
    python servers/server_canary.py [--model_name /root/autodl-tmp/model/canary-1b-v2] [--port 8002]
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from nemo.collections.asr.models import ASRModel
import tempfile
import argparse
import glob
import os
import shutil
import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Canary ASR Transcription Server")
parser.add_argument(
    "--model_name",
    type=str,
    default="/root/autodl-tmp/model/canary_1b_v2",
    help="Local directory/path to the model, or a .nemo file path "
         "(default: /root/autodl-tmp/model/canary-1b-v2)",
)
parser.add_argument(
    "--port",
    type=int,
    default=8002,
    help="Port to listen on (default: 8002)",
)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Canary ASR Transcription Server")


# ---------------------------------------------------------------------------
# Helper: resolve model path
# ---------------------------------------------------------------------------
def resolve_model_path(model_name: str) -> tuple[str, str]:
    """
    Determine how to load the model.

    Returns
    -------
    (method, path)
        method : "restore"  -> call ASRModel.restore_from(restore_path=path)
                 "pretrained" -> call ASRModel.from_pretrained(model_name=path)
        path   : the resolved path or model identifier
    """
    # Case 1: explicit .nemo file
    if model_name.endswith(".nemo") and os.path.isfile(model_name):
        return "restore", model_name

    # Case 2: local directory – look for a .nemo checkpoint inside
    if os.path.isdir(model_name):
        nemo_files = sorted(glob.glob(os.path.join(model_name, "*.nemo")))
        if nemo_files:
            # Prefer the first .nemo file found
            print(f"Found .nemo file in directory: {nemo_files[0]}")
            return "restore", nemo_files[0]

        # Also check one level deeper (e.g. <dir>/checkpoints/*.nemo)
        nemo_files_deep = sorted(glob.glob(os.path.join(model_name, "**", "*.nemo"), recursive=True))
        if nemo_files_deep:
            print(f"Found .nemo file in subdirectory: {nemo_files_deep[0]}")
            return "restore", nemo_files_deep[0]

        # If the directory contains a HuggingFace-style model (config.json / model card),
        # NeMo's from_pretrained can sometimes handle it – but it must NOT look like
        # an absolute path to HF validation.  We still try restore_from with the dir.
        raise FileNotFoundError(
            f"Local directory '{model_name}' exists but no .nemo file was found inside. "
            f"Contents: {os.listdir(model_name)}"
        )

    # Case 3: treat as a HuggingFace / NGC model identifier
    return "pretrained", model_name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print(f"Initializing Canary ASR model '{args.model_name}' ...")

try:
    # Dynamically select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    method, resolved_path = resolve_model_path(args.model_name)
    print(f"Loading method: {method}, resolved path: {resolved_path}")

    if method == "restore":
        asr_model = ASRModel.restore_from(restore_path=resolved_path, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(model_name=resolved_path, map_location=device)

    print(f"Success: Canary ASR model loaded from '{resolved_path}'.")
except Exception as e:
    print(f"Error loading Canary ASR model: {e}")
    raise e

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model": args.model_name,
        "device": device
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("en"),
):
    """
    Transcribe an uploaded audio file using Canary ASR.
    """
    # Create a secure, collision-free temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_filename = temp_file.name

    try:
        # Save uploaded file to disk securely
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe using Canary inside a threadpool to prevent blocking the async event loop
        # EncDecMultiTaskModel.transcribe() expects the audio file list as the
        # first positional argument ('audio'), not 'paths2audio_files'.
        output = await run_in_threadpool(
            lambda: asr_model.transcribe(
                [temp_filename],
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )

        # NeMo transcribe returns a list of results
        if isinstance(output, list) and len(output) > 0:
            first_result = output[0]
            if isinstance(first_result, str):
                text = first_result
            elif hasattr(first_result, "text"):
                text = first_result.text
            else:
                text = str(first_result)
        else:
            text = str(output)

        return {
            "filename": file.filename,
            "text": text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Close the temp file descriptor and delete the file
        temp_file.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)