"""
Configuration loader for the audio translation pipeline.
Loads API keys and settings from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = PROJECT_ROOT / "input"

# Transcription Server API Configuration (Remote Servers)
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://localhost:8000")
WHISPER_MODEL = "large-v3"
GRANITE_API_URL = os.getenv("GRANITE_API_URL", "http://localhost:8001")
CANARY_API_URL = os.getenv("CANARY_API_URL", "http://localhost:8002")

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TRANSCRIPTION_MODEL = os.getenv("OPENROUTER_TRANSCRIPTION_MODEL", "openai/gpt-audio-mini")
TRANSLATION_MODEL = "deepseek/deepseek-v3.2"
CROSS_VALIDATION_MODEL = TRANSLATION_MODEL  # Model for cross-validating Whisper transcriptions

# MiniMax API Configuration
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_UPLOAD_URL = f"{MINIMAX_BASE_URL}/files/upload"
MINIMAX_CLONE_URL = f"{MINIMAX_BASE_URL}/voice_clone"
MINIMAX_T2A_URL = f"{MINIMAX_BASE_URL}/t2a/v2"
MINIMAX_TTS_MODEL = "speech-2.8-hd"

# Terminology database paths
PHILOSOPHY_TERMS_PATH = DATA_DIR / "philosophy_terms.json"
EPISODE_TERMS_PATH = DATA_DIR / "episode_terms.json"
CHINESE_ONLY_TERMS_PATH = DATA_DIR / "chinese_only_terms.json"

# Audio settings
MAX_AUDIO_CHUNK_SECONDS = 30  # For long audio processing
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac"]


def validate_config() -> dict:
    """
    Validate configuration and return status.
    
    Returns:
        dict: Configuration validation results
    """
    issues = []
    
    if not OPENROUTER_API_KEY:
        issues.append("OPENROUTER_API_KEY is not set")
    
    if not MINIMAX_API_KEY:
        issues.append("MINIMAX_API_KEY is not set")
    
    if WHISPER_API_URL == "http://localhost:8000":
        issues.append("WHISPER_API_URL is using default localhost - update to your server")
    
    if GRANITE_API_URL == "http://localhost:8001":
        issues.append("GRANITE_API_URL is using default localhost - update to your server")
    
    if CANARY_API_URL == "http://localhost:8002":
        issues.append("CANARY_API_URL is using default localhost - update to your server")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "config": {
            "whisper_url": WHISPER_API_URL,
            "granite_url": GRANITE_API_URL,
            "canary_url": CANARY_API_URL,
            "translation_model": TRANSLATION_MODEL,
            "tts_model": MINIMAX_TTS_MODEL
        }
    }


if __name__ == "__main__":
    # Dry run: validate configuration
    result = validate_config()
    print("Configuration Validation:")
    print(f"  Valid: {result['valid']}")
    if result['issues']:
        print("  Issues:")
        for issue in result['issues']:
            print(f"    - {issue}")
    print(f"  Config: {result['config']}")
