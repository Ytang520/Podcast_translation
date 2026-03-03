"""
Step 1: Audio Text Extraction

Extracts English text from audio using Whisper API or OpenRouter (closed-source models).
Alternatively, cross-validates 3 existing Whisper text files via --from_texts.

Usage:
    # Direct Whisper transcription (default)
    python step1_transcribe.py --input audio.mp3 --episode_id ep001 [--output_dir output/]

    # OpenRouter transcription (e.g. gpt-audio-mini)
    python step1_transcribe.py --input audio.mp3 --episode_id ep001 --method openrouter [--temperature 0.2]

    # Transcribe N times for cross-validation
    python step1_transcribe.py --input audio.mp3 --episode_id ep001 --method openrouter --runs 3

    # Cross-validate existing text files (delegates to cross_validator.py)
    python step1_transcribe.py --from_texts --episode_id ep001 [--input_dir input/] [--output_dir output/]
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from config import INPUT_DIR, validate_config, WHISPER_API_URL, WHISPER_MODEL, OPENROUTER_TRANSCRIPTION_MODEL, OPENROUTER_API_KEY
from transcriber import (
    transcribe_audio, 
    batch_transcribe,
    dry_run_transcribe, 
    TranscriptionError
)
from transcriber_openrouter import (
    transcribe_audio_openrouter,
    dry_run_transcribe_openrouter,
    OpenRouterTranscriptionError,
)
from cross_validator import run_cross_validation


def run_step1(
    input_audio: str,
    episode_id: str,
    output_dir: str = None,
    dry_run: bool = False,
    method: str = "whisper",
    temperature: float = 0.6,
    transcription_model: str = None,
) -> dict:
    """
    Run Step 1: Transcribe audio to text.
    
    Args:
        input_audio: Path to English audio file
        episode_id: Unique identifier for this episode
        output_dir: Directory to save output files
        dry_run: If True, validate logic without API calls
        
    Returns:
        dict: Results including transcription and file paths
    """
    input_path = Path(input_audio)
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    
    if dry_run:
        return run_dry_run(input_path, episode_id, method=method, temperature=temperature, transcription_model=transcription_model)
    
    results = {
        "status": "pending",
        "episode_id": episode_id,
        "started_at": datetime.now().isoformat(),
    }
    
    try:
        # Transcribe audio
        print("\n[1/2] Transcribing audio...")
        print(f"  Method: {method}")

        if method == "openrouter":
            model_name = transcription_model or OPENROUTER_TRANSCRIPTION_MODEL
            print(f"  Model: {model_name}")
            print(f"  Temperature: {temperature}")

            transcription = transcribe_audio_openrouter(
                str(input_path),
                model=model_name,
                temperature=temperature,
            )
            english_text = transcription["text"]
            duration = 0
            segments = []

            print(f"  Text length: {len(english_text)} characters")
        else:
            # Default: Whisper API
            print(f"  API: {WHISPER_API_URL}")
            print(f"  Model: {WHISPER_MODEL}")

            transcription = transcribe_audio(str(input_path))
            english_text = transcription["text"]
            duration = transcription.get("duration", 0)
            segments = transcription.get("segments", [])

            print(f"  Duration: {duration:.1f}s")
            print(f"  Text length: {len(english_text)} characters")
            print(f"  Segments: {len(segments)}")
        
        # Save output files
        print("\n[2/2] Saving output files...")
        
        # JSON output for Step 2
        output_json_path = output_dir / f"{episode_id}_transcription.json"
        output_data = {
            "episode_id": episode_id,
            "created_at": datetime.now().isoformat(),
            "audio_file": str(input_path.name),
            "duration_seconds": duration,
            "english_text": english_text,
            "segments": segments
        }
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Plain text output
        txt_path = output_dir / f"{episode_id}_english.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Episode: {episode_id}\n")
            f.write(f"# Duration: {duration:.1f}s\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            f.write(english_text)
        
        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()
        results["transcription"] = {
            "duration": duration,
            "text_length": len(english_text),
            "segments_count": len(segments)
        }
        results["output_files"] = {
            "json": str(output_json_path),
            "text": str(txt_path)
        }
        
        print(f"\n  Output saved:")
        print(f"    JSON: {output_json_path}")
        print(f"    Text: {txt_path}")
        
    except (TranscriptionError, OpenRouterTranscriptionError) as e:
        results["status"] = "failed"
        results["error"] = str(e)
        raise
    
    return results


def run_multi_transcribe(
    input_audio: str,
    episode_id: str,
    runs: int = 3,
    method: str = "whisper",
    temperature: float = 0.6,
    transcription_model: str = None,
    input_dir: str = None,
    dry_run: bool = False,
) -> dict:
    """
    Transcribe audio N times and save each result as a numbered .txt file
    in input/<episode_id>/ for cross-validation.

    Args:
        input_audio: Path to english audio file
        episode_id: Episode identifier
        runs: Number of transcription runs (default: 3)
        method: 'whisper' or 'openrouter'
        temperature: Base sampling temperature (OpenRouter only)
        transcription_model: Override model name (OpenRouter only)
        input_dir: Root input directory (default: config INPUT_DIR)
        dry_run: If True, validate logic without API calls

    Returns:
        dict: Results including all transcription file paths
    """
    input_path = Path(input_audio)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    target_dir = Path(input_dir) if input_dir else INPUT_DIR
    episode_dir = target_dir / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return _run_multi_dry_run(
            input_path=input_path,
            episode_id=episode_id,
            runs=runs,
            method=method,
            temperature=temperature,
            transcription_model=transcription_model,
            episode_dir=episode_dir,
        )

    print(f"\n{'=' * 50}")
    print(f"Multi-Transcription Mode: {runs} run(s)")
    print(f"{'=' * 50}")
    print(f"  Input:      {input_path}")
    print(f"  Episode ID: {episode_id}")
    print(f"  Method:     {method}")
    print(f"  Target dir: {episode_dir}")
    print(f"  Runs:       {runs}")

    results = {
        "status": "pending",
        "episode_id": episode_id,
        "runs": runs,
        "started_at": datetime.now().isoformat(),
        "output_files": [],
    }

    for i in range(runs):
        # Slightly perturb temperature for each run (OpenRouter only)
        # so the model produces diverse outputs for cross-validation.
        run_temp = temperature + 0.02 * i if method == "openrouter" else temperature

        print(f"\n--- Run {i + 1}/{runs} ---")
        if method == "openrouter":
            print(f"  Temperature: {run_temp:.2f}")

        try:
            if method == "openrouter":
                model_name = transcription_model or OPENROUTER_TRANSCRIPTION_MODEL
                transcription = transcribe_audio_openrouter(
                    str(input_path),
                    model=model_name,
                    temperature=run_temp,
                )
                english_text = transcription["text"]
            else:
                transcription = transcribe_audio(str(input_path))
                english_text = transcription["text"]

            # Save as numbered .txt
            txt_path = episode_dir / f"{i}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(english_text)

            print(f"  Saved: {txt_path} ({len(english_text)} chars)")
            results["output_files"].append(str(txt_path))

        except (TranscriptionError, OpenRouterTranscriptionError) as e:
            print(f"  [FAIL] Run {i + 1} failed: {e}")
            results["status"] = "partial_failure"
            results.setdefault("errors", []).append({"run": i, "error": str(e)})
            continue

    if not results.get("errors"):
        results["status"] = "success"
    results["completed_at"] = datetime.now().isoformat()

    print(f"\n{'=' * 50}")
    print(f"Multi-Transcription Complete: {len(results['output_files'])}/{runs} files saved")
    print(f"  Target dir: {episode_dir}")
    print(f"\nNext: Run cross-validation:")
    print(f"  python step1_transcribe.py --from_texts -e {episode_id} -o output/{episode_id}/")
    print(f"{'=' * 50}")

    return results


def _run_multi_dry_run(
    input_path: Path,
    episode_id: str,
    runs: int,
    method: str,
    temperature: float,
    transcription_model: str,
    episode_dir: Path,
) -> dict:
    """Dry-run mode for multi-transcription."""
    print(f"\n{'=' * 50}")
    print("DRY RUN MODE - Multi-Transcription")
    print(f"{'=' * 50}")

    results = {"mode": "dry_run", "status": "validating", "checks": {}}

    # Check 1: Input file
    print("\n[1/3] Checking input file...")
    file_exists = input_path.exists()
    format_valid = input_path.suffix.lower() in [".mp3", ".wav", ".m4a", ".flac"]
    print(f"  File exists: {file_exists}")
    print(f"  Format valid: {format_valid}")
    if file_exists:
        size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    results["checks"]["input"] = {"file_exists": file_exists, "format_valid": format_valid}

    # Check 2: Configuration
    print("\n[2/3] Checking configuration...")
    print(f"  Method: {method}")
    if method == "openrouter":
        model_name = transcription_model or OPENROUTER_TRANSCRIPTION_MODEL
        print(f"  Model: {model_name}")
        print(f"  API key set: {bool(OPENROUTER_API_KEY)}")
        results["checks"]["config"] = {"method": "openrouter", "model": model_name, "api_key_set": bool(OPENROUTER_API_KEY)}
    else:
        print(f"  Whisper URL: {WHISPER_API_URL}")
        print(f"  Model: {WHISPER_MODEL}")
        results["checks"]["config"] = {"method": "whisper", "whisper_url": WHISPER_API_URL, "model": WHISPER_MODEL}

    # Check 3: Multi-run plan
    print("\n[3/3] Multi-run plan...")
    print(f"  Runs:       {runs}")
    print(f"  Target dir: {episode_dir}")
    print(f"  Files:")
    for i in range(runs):
        run_temp = temperature + 0.02 * i if method == "openrouter" else temperature
        temp_str = f" (temperature={run_temp:.2f})" if method == "openrouter" else ""
        print(f"    {i}.txt{temp_str}")
    results["checks"]["plan"] = {"runs": runs, "target_dir": str(episode_dir)}

    # Summary
    print(f"\n{'=' * 50}")
    if method == "openrouter":
        all_valid = file_exists and format_valid and bool(OPENROUTER_API_KEY)
    else:
        all_valid = file_exists and format_valid
    results["status"] = "valid" if all_valid else "issues_found"
    if all_valid:
        print("[OK] All multi-transcription checks PASSED")
        print("\nRemove --dry_run to run multi-transcription")
    else:
        print("[FAIL] Some checks FAILED")
    print(f"{'=' * 50}")

    return results


def run_dry_run(
    input_path: Path,
    episode_id: str,
    method: str = "whisper",
    temperature: float = 0.6,
    transcription_model: str = None,
) -> dict:
    """Run Step 1 in dry-run mode."""
    print("\n" + "=" * 50)
    print("DRY RUN MODE - Step 1: Audio Transcription")
    print("=" * 50)
    
    results = {"mode": "dry_run", "status": "validating", "checks": {}}
    
    # Check 1: Configuration
    print("\n[1/2] Checking configuration...")
    config_check = validate_config()
    print(f"  Method: {method}")
    
    if method == "openrouter":
        transcribe_check = dry_run_transcribe_openrouter(
            str(input_path), model=transcription_model, temperature=temperature
        )
        results["checks"]["config"] = {
            "method": "openrouter",
            "model": transcribe_check["model"],
            "temperature": transcribe_check["temperature"],
            "api_key_set": transcribe_check["api_key_set"],
        }
        print(f"  Model: {transcribe_check['model']}")
        print(f"  Temperature: {transcribe_check['temperature']}")
        print(f"  API key set: {transcribe_check['api_key_set']}")
    else:
        transcribe_check = dry_run_transcribe(str(input_path))
        results["checks"]["config"] = {
            "method": "whisper",
            "whisper_url": WHISPER_API_URL,
            "model": WHISPER_MODEL,
        }
        print(f"  Whisper URL: {WHISPER_API_URL}")
        print(f"  Model: {WHISPER_MODEL}")
    
    # Check 2: Input file
    print("\n[2/2] Checking input file...")
    results["checks"]["transcriber"] = transcribe_check
    print(f"  File exists: {transcribe_check['file_exists']}")
    print(f"  Format valid: {transcribe_check['format_valid']}")
    if transcribe_check.get('file_size_mb'):
        print(f"  File size: {transcribe_check['file_size_mb']:.1f} MB")
    if transcribe_check.get('endpoint'):
        print(f"  API endpoint: {transcribe_check['endpoint']}")
    
    # Summary
    print("\n" + "=" * 50)
    all_valid = transcribe_check['file_exists'] and transcribe_check['format_valid']
    results["status"] = "valid" if all_valid else "issues_found"
    
    if all_valid:
        print("[OK] All Step 1 checks PASSED")
        print("\nRemove --dry_run to run transcription")
    else:
        print("[FAIL] Some checks FAILED")
    
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Audio Text Extraction (Whisper / OpenRouter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (Whisper mode)
    python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --dry_run
    
    # Full processing (Whisper mode)
    python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --output_dir output/

    # OpenRouter transcription (e.g. gpt-audio-mini)
    python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --method openrouter --temperature 0.2

    # Cross-validate existing text files
    python step1_transcribe.py --from_texts --episode_id ep001 --output_dir output/ep001/
        """
    )
    
    parser.add_argument("--input", "-i", help="Path to input audio file (required in Whisper mode)")
    parser.add_argument("--episode_id", "-e", required=True, help="Episode identifier")
    parser.add_argument("--output_dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--dry_run", "-d", action="store_true", help="Validate without API calls")
    parser.add_argument("--from_texts", "-t", action="store_true",
                        help="Cross-validate 3 existing Whisper text files instead of running Whisper")
    parser.add_argument("--input_dir", help="Input directory for --from_texts mode (default: input/)")
    parser.add_argument("--model", "-m", help="LLM model for cross-validation (only with --from_texts)")
    parser.add_argument("--adaptive", "-a", action="store_true",
                        help="Escalate to 4th/5th file on high mismatch (only with --from_texts)")
    parser.add_argument("--method", choices=["whisper", "openrouter"], default="whisper",
                        help="Transcription method: whisper (default) or openrouter (closed-source model)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for OpenRouter model (default: 0.2, only with --method openrouter)")
    parser.add_argument("--transcription_model",
                        help="Override the default OpenRouter transcription model (only with --method openrouter)")
    parser.add_argument("--runs", "-n", type=int, default=1,
                        help="Transcribe N times and save to input/<episode_id>/ for cross-validation (default: 1, recommended: 3)")
    
    args = parser.parse_args()
    
    # --from_texts mode: delegate to cross_validator
    if args.from_texts:
        print("=" * 50)
        print("Step 1: Cross-Validate Whisper Transcriptions")
        print("=" * 50)
        print(f"Episode ID: {args.episode_id}")
        print(f"Input Dir:  {args.input_dir or 'input/'}")
        print(f"Output Dir: {args.output_dir or f'output/{args.episode_id}/'}")
        print(f"Model:      {args.model or 'Default'}")
        print(f"Dry Run:    {args.dry_run}")
        print(f"Adaptive:   {args.adaptive}")
        
        try:
            results = run_cross_validation(
                episode_id=args.episode_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model=args.model,
                dry_run=args.dry_run,
                adaptive=args.adaptive,
            )
            print(f"\nStep 1 completed: {results['status']}")
            if results.get('status') == 'success':
                print(f"\nNext: Run step2_translate.py with the JSON output")
            return 0
        except Exception as e:
            print(f"\n[FAIL] Cross-validation failed: {e}", file=sys.stderr)
            return 1
    
    # Whisper / OpenRouter mode: --input is required
    if not args.input:
        parser.error("--input is required in Whisper/OpenRouter mode (or use --from_texts)")
    
    # Multi-transcription mode (--runs > 1)
    if args.runs > 1:
        print("=" * 50)
        print(f"Step 1: Multi-Transcription ({args.runs} runs)")
        print("=" * 50)
        print(f"Input:      {args.input}")
        print(f"Episode ID: {args.episode_id}")
        print(f"Method:     {args.method}")
        print(f"Runs:       {args.runs}")
        print(f"Dry Run:    {args.dry_run}")
        if args.method == "openrouter":
            print(f"Temperature: {args.temperature} (base)")
            if args.transcription_model:
                print(f"Model:       {args.transcription_model}")
        
        try:
            results = run_multi_transcribe(
                input_audio=args.input,
                episode_id=args.episode_id,
                runs=args.runs,
                method=args.method,
                temperature=args.temperature,
                transcription_model=args.transcription_model,
                input_dir=args.input_dir,
                dry_run=args.dry_run,
            )
            
            print(f"\nMulti-transcription completed: {results['status']}")
            return 0
            
        except Exception as e:
            print(f"\n[FAIL] Multi-transcription failed: {e}", file=sys.stderr)
            return 1

    # Single transcription mode (default)
    print("=" * 50)
    print("Step 1: Audio Text Extraction")
    print("=" * 50)
    print(f"Input:      {args.input}")
    print(f"Episode ID: {args.episode_id}")
    print(f"Method:     {args.method}")
    print(f"Output Dir: {args.output_dir or 'same as input'}")
    print(f"Dry Run:    {args.dry_run}")
    if args.method == "openrouter":
        print(f"Temperature: {args.temperature}")
        if args.transcription_model:
            print(f"Model:       {args.transcription_model}")
    
    try:
        results = run_step1(
            input_audio=args.input,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            method=args.method,
            temperature=args.temperature,
            transcription_model=args.transcription_model,
        )
        
        print(f"\nStep 1 completed: {results['status']}")
        
        if results['status'] == 'success':
            print(f"\nNext: Run step2_translate.py with the JSON output")
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Step 1 failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
