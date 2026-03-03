"""
MiniMax TTS voice cloning module.
Uses the Chinese MiniMax API for voice cloning and speech synthesis.
"""
import os
import requests
import base64
from pathlib import Path
from typing import Optional

from config import (
    MINIMAX_API_KEY,
    MINIMAX_UPLOAD_URL,
    MINIMAX_CLONE_URL,
    MINIMAX_T2A_URL,
    MINIMAX_TTS_MODEL
)


class TTSError(Exception):
    """Exception raised when TTS operation fails."""
    pass


def upload_audio(
    audio_path: str,
    purpose: str = "voice_clone",
    api_key: Optional[str] = None
) -> str:
    """
    Upload audio file to MiniMax for voice cloning.
    
    Args:
        audio_path: Path to audio file (mp3, m4a, wav)
        purpose: "voice_clone" for source audio, "prompt_audio" for example
        api_key: MiniMax API key (default from config)
        
    Returns:
        file_id: ID of uploaded file
        
    Raises:
        TTSError: If upload fails
        FileNotFoundError: If audio file not found
    """
    audio_path = Path(audio_path)
    api_key = api_key or MINIMAX_API_KEY
    
    if not api_key:
        raise TTSError("MINIMAX_API_KEY is not set")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Validate format
    valid_formats = [".mp3", ".m4a", ".wav"]
    if audio_path.suffix.lower() not in valid_formats:
        raise TTSError(f"Unsupported format. Must be: {valid_formats}")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f)}
        data = {"purpose": purpose}
        
        try:
            response = requests.post(
                MINIMAX_UPLOAD_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            file_id = result.get("file", {}).get("file_id")
            
            if not file_id:
                raise TTSError(f"No file_id in response: {result}")
            
            return file_id
            
        except requests.exceptions.HTTPError as e:
            raise TTSError(f"Upload failed: {e}")


def clone_voice(
    source_file_id: str,
    voice_id: str,
    prompt_file_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
    preview_text: str = "这是一段测试语音合成的文本。",
    api_key: Optional[str] = None
) -> dict:
    """
    Clone a voice from uploaded audio.
    
    Args:
        source_file_id: file_id of source audio (10s-5min)
        voice_id: Custom voice ID you define
        prompt_file_id: Optional file_id of prompt audio (<8s)
        prompt_text: Optional text describing the prompt audio
        preview_text: Text for preview synthesis
        api_key: MiniMax API key
        
    Returns:
        dict: Clone result with preview audio (base64)
        
    Raises:
        TTSError: If cloning fails
    """
    api_key = api_key or MINIMAX_API_KEY
    
    if not api_key:
        raise TTSError("MINIMAX_API_KEY is not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "file_id": source_file_id,
        "voice_id": voice_id,
        "text": preview_text,
        "model": MINIMAX_TTS_MODEL
    }
    
    # Add optional clone prompt
    if prompt_file_id:
        payload["clone_prompt"] = {
            "prompt_audio": prompt_file_id
        }
        if prompt_text:
            payload["clone_prompt"]["prompt_text"] = prompt_text
    
    try:
        response = requests.post(
            MINIMAX_CLONE_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        raise TTSError(f"Voice cloning failed: {e}")


def synthesize_speech(
    text: str,
    voice_id: str,
    output_path: str,
    api_key: Optional[str] = None
) -> str:
    """
    Synthesize speech using a cloned voice.
    
    Args:
        text: Chinese text to synthesize
        voice_id: Cloned voice ID
        output_path: Path to save audio file
        api_key: MiniMax API key
        
    Returns:
        Path to saved audio file
        
    Raises:
        TTSError: If synthesis fails
    """
    api_key = api_key or MINIMAX_API_KEY
    
    if not api_key:
        raise TTSError("MINIMAX_API_KEY is not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_id": voice_id,
        "model": MINIMAX_TTS_MODEL,
        "audio_setting": {
            "format": "mp3",
            "sample_rate": 24000
        }
    }
    
    try:
        response = requests.post(
            MINIMAX_T2A_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract audio data (base64 encoded)
        audio_data = result.get("audio", {}).get("data")
        
        if not audio_data:
            raise TTSError(f"No audio data in response: {result}")
        
        # Decode and save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio_bytes = base64.b64decode(audio_data)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        return str(output_path)
        
    except requests.exceptions.HTTPError as e:
        raise TTSError(f"Speech synthesis failed: {e}")


def synthesize_long_text(
    text: str,
    voice_id: str,
    output_path: str,
    max_chars_per_chunk: int = 500,
    api_key: Optional[str] = None
) -> str:
    """
    Synthesize long text by splitting into chunks.
    
    Args:
        text: Long Chinese text to synthesize
        voice_id: Cloned voice ID
        output_path: Path to save final audio file
        max_chars_per_chunk: Maximum characters per API call
        api_key: MiniMax API key
        
    Returns:
        Path to saved audio file
    """
    from pydub import AudioSegment
    
    # Split text into chunks (at sentence boundaries)
    chunks = split_text_into_chunks(text, max_chars_per_chunk)
    
    # Synthesize each chunk
    chunk_audio_paths = []
    output_path = Path(output_path)
    
    for i, chunk in enumerate(chunks):
        chunk_path = output_path.parent / f"_chunk_{i}.mp3"
        synthesize_speech(chunk, voice_id, str(chunk_path), api_key)
        chunk_audio_paths.append(chunk_path)
    
    # Combine all chunks
    combined = AudioSegment.empty()
    for chunk_path in chunk_audio_paths:
        audio = AudioSegment.from_mp3(chunk_path)
        combined += audio
        # Clean up chunk file
        chunk_path.unlink()
    
    # Export final audio
    combined.export(output_path, format="mp3")
    
    return str(output_path)


def split_text_into_chunks(text: str, max_chars: int = 500) -> list:
    """
    Split text into chunks at sentence boundaries.
    
    Args:
        text: Chinese text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # Chinese sentence delimiters
    sentence_pattern = r'([。！？；\n])'
    sentences = re.split(sentence_pattern, text)
    
    # Recombine sentences with their delimiters
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    
    # Handle last piece if odd number
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])
    
    # Group into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in combined_sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            # Warn if a single sentence exceeds the chunk limit
            if len(sentence) > max_chars:
                print(f"[Warning] Single sentence ({len(sentence)} chars) exceeds max_chars ({max_chars}). "
                      "MiniMax API may reject this chunk.")
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def dry_run_tts(voice_sample_path: str = "voice_sample.mp3") -> dict:
    """
    Dry run: Test TTS logic without calling API.
    
    Args:
        voice_sample_path: Path to test voice sample
        
    Returns:
        dict: Validation results
    """
    voice_sample_path = Path(voice_sample_path)
    
    # Test text splitting
    test_text = "这是第一句话。这是第二句话！这是第三句话？" * 10
    chunks = split_text_into_chunks(test_text, max_chars=50)
    
    return {
        "voice_sample_exists": voice_sample_path.exists(),
        "upload_url": MINIMAX_UPLOAD_URL,
        "clone_url": MINIMAX_CLONE_URL,
        "t2a_url": MINIMAX_T2A_URL,
        "model": MINIMAX_TTS_MODEL,
        "api_key_set": bool(MINIMAX_API_KEY),
        "test_text_length": len(test_text),
        "chunks_created": len(chunks),
        "chunk_sizes": [len(c) for c in chunks],
        "expected_workflow": [
            "1. upload_audio(voice_sample) → file_id",
            "2. clone_voice(file_id, custom_voice_id) → voice_id",
            "3. synthesize_speech(chinese_text, voice_id) → audio_file"
        ],
        "logic_valid": all(len(c) <= 50 for c in chunks)
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=== TTS MiniMax Dry Run ===\n")
    
    sample_path = sys.argv[1] if len(sys.argv) > 1 else "voice_sample.mp3"
    result = dry_run_tts(sample_path)
    
    print(f"Voice sample exists: {result['voice_sample_exists']}")
    print(f"Upload URL: {result['upload_url']}")
    print(f"Clone URL: {result['clone_url']}")
    print(f"Model: {result['model']}")
    print(f"API key set: {result['api_key_set']}")
    
    print(f"\nText chunking test:")
    print(f"  Test text length: {result['test_text_length']}")
    print(f"  Chunks created: {result['chunks_created']}")
    print(f"  Chunk sizes: {result['chunk_sizes'][:5]}...")
    
    print(f"\nExpected workflow:")
    for step in result['expected_workflow']:
        print(f"  {step}")
    
    print(f"\n✓ Logic check: {'PASSED' if result['logic_valid'] else 'FAILED'}")
