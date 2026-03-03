"""
Step 3: Audio Generation with Voice Cloning

Synthesizes Chinese audio from Step 2 translation output.

Usage:
    python step3_audio.py --input ep001_translation.json --voice_sample voice.mp3 --output ep001_chinese.mp3
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from config import validate_config
from tts_minimax import (
    upload_audio,
    clone_voice,
    synthesize_long_text,
    dry_run_tts,
    TTSError
)


class AudioGenerationError(Exception):
    """Exception raised when audio generation fails."""
    pass


def run_step3(
    input_json: str,
    voice_sample: str,
    output_audio: str,
    dry_run: bool = False
) -> dict:
    """
    Run Step 3: Voice cloning and TTS synthesis.
    
    Args:
        input_json: Path to Step 2 translation JSON
        voice_sample: Path to voice sample for cloning (10s-5min)
        output_audio: Path for output audio file
        dry_run: If True, validate logic without API calls
    """
    input_path = Path(input_json)
    voice_path = Path(voice_sample)
    output_path = Path(output_audio)
    
    # Validate inputs
    if not input_path.exists():
        raise FileNotFoundError(f"Translation JSON not found: {input_path}")
    
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice sample not found: {voice_path}")
    
    # Load translation data
    with open(input_path, "r", encoding="utf-8") as f:
        translation_data = json.load(f)
    
    episode_id = translation_data.get("episode_id", "unknown")
    chinese_text = translation_data.get("chinese_text", "")
    
    if not chinese_text:
        raise AudioGenerationError("No Chinese text found in translation JSON")
    
    if dry_run:
        return run_dry_run(input_path, voice_path, output_path, translation_data)
    
    results = {
        "status": "pending",
        "episode_id": episode_id,
        "started_at": datetime.now().isoformat(),
    }
    
    try:
        # Step 3a: Upload voice sample
        print("\n[1/3] Uploading voice sample...")
        file_id = upload_audio(str(voice_path), purpose="voice_clone")
        print(f"  Uploaded: {file_id}")
        
        # Step 3b: Clone voice
        print("\n[2/3] Cloning voice...")
        voice_id = f"philosophy_voice_{episode_id}"
        clone_result = clone_voice(
            source_file_id=file_id,
            voice_id=voice_id,
            preview_text="这是一段哲学讲座的中文翻译测试。"
        )
        print(f"  Voice cloned: {voice_id}")
        
        # Step 3c: Synthesize speech
        print("\n[3/3] Synthesizing Chinese speech...")
        print(f"  Text length: {len(chinese_text)} characters")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = synthesize_long_text(
            chinese_text,
            voice_id=voice_id,
            output_path=str(output_path)
        )
        print(f"  Output saved: {output_file}")
        
        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()
        results["synthesis"] = {
            "voice_id": voice_id,
            "text_length": len(chinese_text),
            "output_file": str(output_path)
        }
        results["output_file"] = str(output_path)
        
    except TTSError as e:
        results["status"] = "failed"
        results["error"] = str(e)
        raise AudioGenerationError(str(e))
    
    return results


def run_dry_run(
    input_path: Path,
    voice_path: Path,
    output_path: Path,
    translation_data: dict
) -> dict:
    """Run Step 3 in dry-run mode."""
    print("\n" + "=" * 50)
    print("DRY RUN MODE - Step 3: Audio Generation")
    print("=" * 50)
    
    results = {"mode": "dry_run", "status": "validating", "checks": {}}
    
    # Check 1: Translation data
    print("\n[1/3] Checking translation data...")
    episode_id = translation_data.get("episode_id", "unknown")
    chinese_text = translation_data.get("chinese_text", "")
    summary = translation_data.get("summary", "")
    
    results["checks"]["translation_data"] = {
        "episode_id": episode_id,
        "chinese_text_length": len(chinese_text),
        "summary_length": len(summary),
        "valid": bool(chinese_text)
    }
    print(f"  Episode: {episode_id}")
    print(f"  Chinese text: {len(chinese_text)} chars")
    print(f"  Summary: {len(summary)} chars")
    
    # Check 2: Voice sample
    print("\n[2/3] Checking voice sample...")
    results["checks"]["voice_sample"] = {"exists": voice_path.exists()}
    print(f"  Voice sample exists: {voice_path.exists()}")
    
    # Check 3: TTS
    print("\n[3/3] Checking TTS...")
    tts_check = dry_run_tts(str(voice_path))
    results["checks"]["tts"] = tts_check
    print(f"  Model: {tts_check['model']}")
    print(f"  Chunking test: {tts_check['chunks_created']} chunks")
    print(f"  Logic valid: {tts_check['logic_valid']}")
    
    # Summary
    print("\n" + "=" * 50)
    all_valid = bool(chinese_text) and voice_path.exists() and tts_check['logic_valid']
    results["status"] = "valid" if all_valid else "issues_found"
    
    if all_valid:
        print("[OK] All Step 3 checks PASSED")
        print("\nRemove --dry_run to generate the audio")
    else:
        print("[FAIL] Some checks FAILED")
    
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Audio Generation with Voice Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run
    python step3_audio.py --input ep001_translation.json --voice_sample voice.mp3 --output ep001.mp3 --dry_run
    
    # Full processing
    python step3_audio.py --input ep001_translation.json --voice_sample voice.mp3 --output ep001_chinese.mp3
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to Step 2 JSON output")
    parser.add_argument("--voice_sample", "-v", required=True, help="Path to voice sample (10s-5min)")
    parser.add_argument("--output", "-o", required=True, help="Path for output audio file")
    parser.add_argument("--dry_run", "-d", action="store_true", help="Validate without API calls")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Step 3: Audio Generation with Voice Cloning")
    print("=" * 50)
    print(f"Input JSON:   {args.input}")
    print(f"Voice Sample: {args.voice_sample}")
    print(f"Output:       {args.output}")
    print(f"Dry Run:      {args.dry_run}")
    
    try:
        results = run_step3(
            input_json=args.input,
            voice_sample=args.voice_sample,
            output_audio=args.output,
            dry_run=args.dry_run
        )
        
        print(f"\nStep 3 completed: {results['status']}")
        
        if results['status'] == 'success':
            print(f"\nFinal audio: {results['output_file']}")
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Step 3 failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
