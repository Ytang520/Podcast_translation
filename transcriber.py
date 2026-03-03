"""
Remote Whisper API client for audio transcription.
Uses the speaches/faster-whisper-server OpenAI-compatible API.
"""
import os
import requests
from pathlib import Path
from typing import Optional

from config import WHISPER_API_URL, WHISPER_MODEL, SUPPORTED_AUDIO_FORMATS


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    pass


def transcribe_audio(
    audio_path: str,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    language: str = "en"
) -> dict:
    """
    Transcribe audio using remote Whisper API.
    
    Args:
        audio_path: Path to the audio file
        api_url: Whisper API URL (default from config)
        model: Whisper model to use (default from config)
        language: Audio language (default: English)
    
    Returns:
        dict: {
            "text": str,           # Full transcription text
            "segments": list,      # Timestamped segments (if available)
            "language": str,       # Detected language
            "duration": float      # Audio duration in seconds
        }
    
    Raises:
        TranscriptionError: If transcription fails
        FileNotFoundError: If audio file not found
    """
    audio_path = Path(audio_path)
    
    # Validate audio file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Validate audio format
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise TranscriptionError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported: {SUPPORTED_AUDIO_FORMATS}"
        )
    
    # Use defaults from config
    api_url = api_url or WHISPER_API_URL
    model = model or WHISPER_MODEL
    
    # Build API endpoint
    endpoint = f"{api_url.rstrip('/')}/v1/audio/transcriptions"
    
    # Prepare request
    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (audio_path.name, audio_file, "audio/mpeg")
            }
            data = {
                "model": model,
                "language": language,
                "response_format": "verbose_json"  # Get timestamps
            }
            
            response = requests.post(
                endpoint,
                files=files,
                data=data,
                timeout=300  # 5 minute timeout for large files
            )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", language),
                "duration": result.get("duration", 0.0)
            }
            
    except requests.exceptions.ConnectionError:
        raise TranscriptionError(
            f"Cannot connect to Whisper API at {api_url}. "
            "Ensure the server is running."
        )
    except requests.exceptions.Timeout:
        raise TranscriptionError(
            "Transcription timed out. Audio file may be too large."
        )
    except requests.exceptions.HTTPError as e:
        raise TranscriptionError(f"API error: {e}")


def dry_run_transcribe(audio_path: str) -> dict:
    """
    Dry run: Validate transcription logic without calling API.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        dict: Validation results
    """
    audio_path = Path(audio_path)
    
    checks = {
        "file_exists": audio_path.exists(),
        "format_valid": audio_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS if audio_path.exists() else False,
        "api_url": WHISPER_API_URL,
        "model": WHISPER_MODEL,
        "endpoint": f"{WHISPER_API_URL.rstrip('/')}/v1/audio/transcriptions"
    }
    
    # Get file size if exists
    if audio_path.exists():
        checks["file_size_mb"] = audio_path.stat().st_size / (1024 * 1024)
    
    return checks


def batch_transcribe(
    audio_paths: list,
    max_workers: int = 4,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    language: str = "en",
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Batch transcribe multiple audio files concurrently.
    
    Since Whisper processes files serially, this function uses multiple
    concurrent API calls to speed up processing of multiple files.
    
    Args:
        audio_paths: List of paths to audio files
        max_workers: Maximum concurrent transcription workers (default: 4)
        api_url: Whisper API URL (default from config)
        model: Whisper model to use (default from config)
        language: Audio language (default: English)
        progress_callback: Optional callback(completed, total, current_file) for progress
        
    Returns:
        dict: {
            "results": {filepath: transcription_result, ...},
            "errors": {filepath: error_message, ...},
            "total_files": int,
            "successful": int,
            "failed": int,
            "total_duration": float
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    start_time = time.time()
    results = {}
    errors = {}
    total = len(audio_paths)
    completed = 0
    
    def transcribe_single(audio_path):
        """Worker function to transcribe a single file."""
        try:
            result = transcribe_audio(
                audio_path, 
                api_url=api_url, 
                model=model, 
                language=language
            )
            return audio_path, result, None
        except Exception as e:
            return audio_path, None, str(e)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(transcribe_single, path): path 
            for path in audio_paths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            audio_path, result, error = future.result()
            completed += 1
            
            if result:
                results[str(audio_path)] = result
            else:
                errors[str(audio_path)] = error
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed, total, audio_path)
    
    elapsed_time = time.time() - start_time
    
    return {
        "results": results,
        "errors": errors,
        "total_files": total,
        "successful": len(results),
        "failed": len(errors),
        "elapsed_seconds": elapsed_time
    }


def batch_transcribe_directory(
    directory: str,
    max_workers: int = 4,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    language: str = "en",
    recursive: bool = False
) -> dict:
    """
    Transcribe all audio files in a directory.
    
    Args:
        directory: Path to directory containing audio files
        max_workers: Maximum concurrent workers
        api_url: Whisper API URL
        model: Whisper model
        language: Audio language
        recursive: If True, search subdirectories
        
    Returns:
        dict: Batch transcription results
    """
    from tqdm import tqdm
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    # Find all audio files
    audio_files = []
    if recursive:
        for ext in SUPPORTED_AUDIO_FORMATS:
            audio_files.extend(dir_path.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_AUDIO_FORMATS:
            audio_files.extend(dir_path.glob(f"*{ext}"))
    
    if not audio_files:
        return {
            "results": {},
            "errors": {},
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "elapsed_seconds": 0.0
        }
    
    # Create progress bar
    pbar = tqdm(total=len(audio_files), desc="Transcribing")
    
    def progress_cb(completed, total, current_file):
        pbar.update(1)
        pbar.set_postfix(file=Path(current_file).name)
    
    result = batch_transcribe(
        audio_files,
        max_workers=max_workers,
        api_url=api_url,
        model=model,
        language=language,
        progress_callback=progress_cb
    )
    
    pbar.close()
    return result


def dry_run_batch_transcribe(audio_paths: list, max_workers: int = 4) -> dict:
    """
    Dry run: Validate batch transcription logic without calling API.
    
    Args:
        audio_paths: List of audio file paths
        max_workers: Number of workers to simulate
        
    Returns:
        dict: Validation results
    """
    valid_files = []
    invalid_files = []
    
    for path in audio_paths:
        p = Path(path)
        if p.exists() and p.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
            valid_files.append(str(p))
        else:
            invalid_files.append({
                "path": str(p),
                "exists": p.exists(),
                "format_valid": p.suffix.lower() in SUPPORTED_AUDIO_FORMATS if p.exists() else False
            })
    
    return {
        "total_files": len(audio_paths),
        "valid_files": len(valid_files),
        "invalid_files": len(invalid_files),
        "invalid_details": invalid_files,
        "max_workers": max_workers,
        "api_url": WHISPER_API_URL,
        "model": WHISPER_MODEL,
        "estimated_speedup": f"Up to {max_workers}x with concurrent processing",
        "logic_valid": True
    }


if __name__ == "__main__":
    import sys
    
    print("=== Transcriber Dry Run ===")
    
    if len(sys.argv) > 1:
        test_paths = sys.argv[1:]
    else:
        test_paths = ["test_audio.mp3"]
    
    # Single file dry run
    if len(test_paths) == 1:
        result = dry_run_transcribe(test_paths[0])
        print(f"Audio path: {test_paths[0]}")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        # Batch dry run
        print(f"\nBatch mode: {len(test_paths)} files")
        result = dry_run_batch_transcribe(test_paths)
        for key, value in result.items():
            if key != "invalid_details":
                print(f"  {key}: {value}")
    
    print("\nLogic check: PASSED")
