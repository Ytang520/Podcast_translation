"""
Whisper Transcription Server (FastAPI).

Loads OpenAI Whisper model and exposes a /transcribe API endpoint.

Usage:
    python servers/server.py [--model_name large-v3] [--model_dir /path/to/models] [--port 8000]
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import argparse
import os
import shutil

# ---------------------------------------------------------------------------
# Argument parsing (runs at import-time so uvicorn picks up the values)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Whisper Transcription Server")
parser.add_argument(
    "--model_name",
    type=str,
    default="large-v3",
    help="Whisper model name, e.g. large-v3, medium, small (default: large-v3)",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="/root/autodl-tmp/model/whisper-large",
    help="Local directory for model weights (default: /root/autodl-tmp/model/whisper-large)",
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port to listen on (default: 8000)",
)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper Transcription Server")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print(f"Initializing Whisper Model '{args.model_name}' ...")
os.makedirs(args.model_dir, exist_ok=True)

try:
    model = whisper.load_model(
        args.model_name, download_root=args.model_dir, device="cuda"
    )
    print(f"Success: Whisper '{args.model_name}' loaded on CUDA.")
except RuntimeError as e:
    print(f"Error loading model on CUDA: {e}")
    print("Fallback to CPU is possible but not recommended for large models.")
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
        "model_dir": args.model_dir,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file using Whisper.

    Returns:
        JSON with filename, detected language, and transcribed text.
    """
    temp_filename = f"temp_{file.filename}"

    try:
        # Save uploaded file to disk
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe
        result = model.transcribe(temp_filename)

        return {
            "filename": file.filename,
            "language": result.get("language"),
            "text": result.get("text"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)