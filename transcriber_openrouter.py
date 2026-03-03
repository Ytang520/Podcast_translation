"""
OpenRouter-based audio transcription client.

Uses OpenRouter's chat completions API with audio input to transcribe
audio files via closed-source models (e.g. openai/gpt-audio-mini).

The audio file is base64-encoded and sent as an `input_audio` content block.
"""
import os
import base64
import requests
from pathlib import Path
from typing import Optional

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_TRANSCRIPTION_MODEL,
    SUPPORTED_AUDIO_FORMATS,
)


class OpenRouterTranscriptionError(Exception):
    """Exception raised when OpenRouter transcription fails."""
    pass


# Mapping from file extension to the format string expected by the API
_FORMAT_MAP = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
    ".webm": "webm",
}


def _encode_audio_to_base64(audio_path: str) -> str:
    """Read an audio file and return its base64-encoded string."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def _get_audio_format(audio_path: str) -> str:
    """
    Determine the audio format string from the file extension.

    Returns:
        str: format identifier (e.g. 'mp3', 'wav')

    Raises:
        OpenRouterTranscriptionError: if the extension is not recognised
    """
    ext = Path(audio_path).suffix.lower()
    fmt = _FORMAT_MAP.get(ext)
    if fmt is None:
        raise OpenRouterTranscriptionError(
            f"Unsupported audio format: {ext}. "
            f"Supported: {list(_FORMAT_MAP.keys())}"
        )
    return fmt


def transcribe_audio_openrouter(
    audio_path: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """
    Transcribe audio using OpenRouter's chat completions API.

    The audio file is base64-encoded and sent as an ``input_audio`` block
    inside a user message.  The model is expected to return the transcription
    as plain text in the assistant response.

    Args:
        audio_path: Path to the audio file.
        model: OpenRouter model identifier
               (default from config: openai/gpt-audio-mini).
        temperature: Sampling temperature (default 0.2).
        api_key: OpenRouter API key (default from .env / config).
        base_url: OpenRouter base URL (default from .env / config).

    Returns:
        dict: {
            "text": str,       # Transcription text
            "model": str,      # Model used
            "temperature": float
        }

    Raises:
        OpenRouterTranscriptionError: on any failure
        FileNotFoundError: if audio file does not exist
    """
    audio_path = Path(audio_path)

    # --- Validate inputs ---
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise OpenRouterTranscriptionError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported: {SUPPORTED_AUDIO_FORMATS}"
        )

    api_key = api_key or OPENROUTER_API_KEY
    base_url = base_url or OPENROUTER_BASE_URL
    model = model or OPENROUTER_TRANSCRIPTION_MODEL

    if not api_key:
        raise OpenRouterTranscriptionError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or pass it explicitly."
        )

    # --- Encode audio ---
    audio_format = _get_audio_format(str(audio_path))
    base64_audio = _encode_audio_to_base64(str(audio_path))

    # --- Build request ---
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please transcribe this audio file.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": audio_format,
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
    }

    # --- Send request ---
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OpenRouterTranscriptionError(
            f"Cannot connect to OpenRouter API at {base_url}. "
            "Check your network connection."
        )
    except requests.exceptions.Timeout:
        raise OpenRouterTranscriptionError(
            "OpenRouter API request timed out. The audio file may be too large."
        )
    except requests.exceptions.HTTPError as e:
        # Try to extract a helpful error message from the response body
        detail = ""
        try:
            detail = response.json().get("error", {}).get("message", "")
        except Exception:
            detail = response.text[:500]
        raise OpenRouterTranscriptionError(
            f"OpenRouter API error ({response.status_code}): {detail or e}"
        )

    # --- Parse response ---
    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise OpenRouterTranscriptionError(
            f"Unexpected API response structure: {e}. "
            f"Response: {response.text[:500]}"
        )

    return {
        "text": text,
        "model": model,
        "temperature": temperature,
    }


def dry_run_transcribe_openrouter(
    audio_path: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> dict:
    """
    Dry run: validate OpenRouter transcription without making API calls.

    Args:
        audio_path: Path to the audio file.
        model: Model identifier (default from config).
        temperature: Temperature value to display.

    Returns:
        dict: Validation results.
    """
    audio_path = Path(audio_path)
    model = model or OPENROUTER_TRANSCRIPTION_MODEL

    checks = {
        "file_exists": audio_path.exists(),
        "format_valid": (
            audio_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
            if audio_path.exists()
            else False
        ),
        "api_key_set": bool(OPENROUTER_API_KEY),
        "base_url": OPENROUTER_BASE_URL,
        "model": model,
        "temperature": temperature,
        "endpoint": f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
    }

    if audio_path.exists():
        checks["file_size_mb"] = audio_path.stat().st_size / (1024 * 1024)

    checks["logic_valid"] = checks["file_exists"] and checks["format_valid"] and checks["api_key_set"]

    return checks


if __name__ == "__main__":
    import sys

    print("=== OpenRouter Transcriber Dry Run ===")

    test_path = sys.argv[1] if len(sys.argv) > 1 else "test_audio.mp3"
    result = dry_run_transcribe_openrouter(test_path)

    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nLogic check:", "PASSED" if result["logic_valid"] else "FAILED")
