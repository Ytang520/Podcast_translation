"""
Main entry point for the philosophy audio translation pipeline.
Translates English philosophy podcasts to Chinese with voice cloning.

Three-step translation process:
    1. Extract terms with AI search for authoritative explanations
    2. Translate using extracted terminology
    3. Polish Chinese draft for podcast narration

Usage:
    python main.py --input audio.mp3 --voice_sample voice.mp3 --output translated.mp3 --episode_id ep001
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from config import validate_config, CHINESE_ONLY_TERMS_PATH, DATA_DIR
from transcriber import transcribe_audio, dry_run_transcribe, TranscriptionError
from translator import translate_with_extraction, dry_run_translate, dry_run_extract_terms, TranslationError
from terminology_db import TerminologyDB, create_default_philosophy_terms
from tts_minimax import (
    upload_audio, 
    clone_voice, 
    synthesize_long_text,
    dry_run_tts,
    TTSError
)


class PipelineError(Exception):
    """Exception raised when pipeline fails."""
    pass


def initialize_terminology_db() -> TerminologyDB:
    """
    Initialize terminology database with default philosophy terms.
    
    Returns:
        TerminologyDB: Initialized database
    """
    db = TerminologyDB()
    
    # Load default terms if database is empty
    if len(db.terms) == 0:
        print("Initializing terminology database with default philosophy terms...")
        default_terms = create_default_philosophy_terms()
        for eng, chi in default_terms.items():
            db.add_term(eng, chi)
        db.save()
        print(f"  Added {len(default_terms)} terms")
    
    return db


def _load_chinese_only_terms(path: str = None) -> list:
    """Load chinese-only terms from JSON file."""
    import json as _json
    terms_path = Path(path) if path else CHINESE_ONLY_TERMS_PATH
    if not terms_path.exists():
        return []
    try:
        with open(terms_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        terms = data.get("terms", [])
        print(f"  Loaded {len(terms)} chinese-only terms from {terms_path.name}")
        return terms
    except Exception as e:
        print(f"  [Warning] Failed to load chinese-only terms: {e}")
        return []


def run_pipeline(
    input_audio: str,
    voice_sample: str,
    output_audio: str,
    episode_id: str,
    dry_run: bool = False,
    model: str = None,
    enable_search: bool = True,
    enable_reasoning: bool = False,
    polish_segment_chars: int = None,
    chinese_only_terms_path: str = None
) -> dict:
    """
    Run the full translation pipeline.
    
    Args:
        input_audio: Path to English audio file
        voice_sample: Path to voice sample for cloning
        output_audio: Path for output Chinese audio
        episode_id: Unique identifier for this episode
        dry_run: If True, validate logic without API calls
        model: Specific model to use (e.g., deepseek/deepseek-r1)
        enable_search: Enable web search for term explanations
        enable_reasoning: Enable reasoning mode (for reasoning models)
        polish_segment_chars: Soft target length for polishing segments (paragraph-aligned)
        
    Returns:
        dict: Pipeline results
    """
    results = {
        "status": "pending",
        "episode_id": episode_id,
        "started_at": datetime.now().isoformat(),
        "steps": {}
    }
    
    input_path = Path(input_audio)
    voice_path = Path(voice_sample)
    output_path = Path(output_audio)
    
    # Validate inputs
    if not input_path.exists():
        raise PipelineError(f"Input audio not found: {input_path}")
    
    if not voice_path.exists():
        raise PipelineError(f"Voice sample not found: {voice_path}")
    
    # Initialize terminology database
    term_db = initialize_terminology_db()
    term_db.set_episode(episode_id)
    
    if dry_run:
        return run_dry_run(input_path, voice_path, output_path, episode_id, term_db, model=model, enable_search=enable_search, enable_reasoning=enable_reasoning)
    
    try:
        # Step 1: Transcribe audio
        print("\n[1/4] Transcribing audio...")
        transcription = transcribe_audio(str(input_path))
        english_text = transcription["text"]
        results["steps"]["transcription"] = {
            "status": "success",
            "duration": transcription.get("duration", 0),
            "text_length": len(english_text)
        }
        print(f"  Transcribed {len(english_text)} characters")
        
        # Step 2: Extract terms and translate (two-step process)
        print("\n[2/4] Extracting terms and translating to Chinese...")
        print("  Step 2a: Extracting philosophy terms with AI search...")
        chinese_only_terms = _load_chinese_only_terms(chinese_only_terms_path)
        result = translate_with_extraction(
            english_text,
            term_db=term_db,
            episode_id=episode_id,
            enable_search=enable_search,
            save_terms=True,
            model=model,
            enable_reasoning=enable_reasoning,
            polish_segment_chars=polish_segment_chars,
            chinese_only_terms=chinese_only_terms
        )
        chinese_text = result["translation"]
        draft_chinese_text = result.get("draft_translation", "")
        new_terms_count = result["new_terms_added"]
        
        results["steps"]["term_extraction"] = {
            "status": "success",
            "total_terms": len(result["extracted_terms"]["terms"]),
            "new_terms": new_terms_count
        }
        results["steps"]["translation"] = {
            "status": "success",
            "text_length": len(chinese_text),
            "draft_text_length": len(draft_chinese_text) if draft_chinese_text else None
        }
        print(f"  Step 2b: Translated to {len(chinese_text)} characters")
        print(f"  New terms added: {new_terms_count}")
        term_db.save()
        
        # Step 3: Clone voice
        print("\n[3/4] Cloning voice...")
        file_id = upload_audio(str(voice_path), purpose="voice_clone")
        voice_id = f"philosophy_voice_{episode_id}"
        clone_result = clone_voice(
            source_file_id=file_id,
            voice_id=voice_id,
            preview_text="这是一段哲学讲座的中文翻译测试。"
        )
        results["steps"]["voice_clone"] = {
            "status": "success",
            "voice_id": voice_id
        }
        print(f"  Voice cloned: {voice_id}")
        
        # Step 4: Synthesize speech
        print("\n[4/4] Synthesizing Chinese speech...")
        output_file = synthesize_long_text(
            chinese_text,
            voice_id=voice_id,
            output_path=str(output_path)
        )
        results["steps"]["synthesis"] = {
            "status": "success",
            "output_file": output_file
        }
        print(f"  Output saved: {output_file}")
        
        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()
        
    except (TranscriptionError, TranslationError, TTSError) as e:
        results["status"] = "failed"
        results["error"] = str(e)
        raise PipelineError(str(e))
    
    return results


def run_dry_run(
    input_path: Path,
    voice_path: Path,
    output_path: Path,
    episode_id: str,
    term_db: TerminologyDB,
    model: str = None,
    enable_search: bool = True,
    enable_reasoning: bool = False
) -> dict:
    """
    Run pipeline in dry-run mode (validate logic without API calls).
    
    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 50)
    print("DRY RUN MODE - Validating pipeline logic")
    print("=" * 50)
    
    results = {
        "mode": "dry_run",
        "status": "validating",
        "checks": {}
    }
    
    # Check 1: Configuration
    print("\n[1/5] Checking configuration...")
    config_check = validate_config()
    results["checks"]["config"] = config_check
    print(f"  Config valid: {config_check['valid']}")
    if config_check['issues']:
        for issue in config_check['issues']:
            print(f"    ⚠ {issue}")
    
    # Check 2: Transcriber
    print("\n[2/5] Checking transcriber...")
    transcribe_check = dry_run_transcribe(str(input_path))
    results["checks"]["transcriber"] = transcribe_check
    print(f"  File exists: {transcribe_check['file_exists']}")
    print(f"  Format valid: {transcribe_check['format_valid']}")
    print(f"  API endpoint: {transcribe_check['endpoint']}")
    
    # Check 3: Terminology database
    print("\n[3/5] Checking terminology database...")
    results["checks"]["terminology"] = {
        "terms_loaded": len(term_db.terms),
        "episode_set": episode_id,
        "stats": term_db.get_stats()
    }
    print(f"  Terms loaded: {len(term_db.terms)}")
    print(f"  Episode: {episode_id}")
    
    # Check 4: Translator (Two-Step Process)
    print("\n[4/5] Checking translator...")
    sample_text = "This lecture explores epistemology and the categorical imperative."
    
    # Check term extraction
    extract_check = dry_run_extract_terms(sample_text, model=model, enable_search=enable_search)
    results["checks"]["term_extraction"] = extract_check
    print(f"  Step 1 - Term Extraction:")
    print(f"    Model: {extract_check['model']}")
    print(f"    Web search: {extract_check['web_search_enabled']}")
    print(f"    Logic valid: {extract_check['logic_valid']}")
    
    # Check translation
    translate_check = dry_run_translate(sample_text, model=model)
    results["checks"]["translator"] = translate_check
    print(f"  Step 2 - Translation:")
    print(f"    Model: {translate_check['model']}")
    print(f"    Logic valid: {translate_check['logic_valid']}")
    print(f"  Reasoning mode: {enable_reasoning}")
    
    # Check 5: TTS
    print("\n[5/5] Checking TTS...")
    tts_check = dry_run_tts(str(voice_path))
    results["checks"]["tts"] = tts_check
    print(f"  Voice sample exists: {tts_check['voice_sample_exists']}")
    print(f"  Model: {tts_check['model']}")
    print(f"  Chunking test: {tts_check['chunks_created']} chunks")
    
    # Summary
    print("\n" + "=" * 50)
    all_valid = (
        transcribe_check['file_exists'] and
        transcribe_check['format_valid'] and
        tts_check['voice_sample_exists'] and
        translate_check['logic_valid'] and
        tts_check['logic_valid']
    )
    
    results["status"] = "valid" if all_valid else "issues_found"
    
    if all_valid:
        print("✓ All logic checks PASSED")
        print("\nTo run the actual pipeline, remove --dry_run flag")
    else:
        print("✗ Some checks FAILED - review issues above")
    
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Philosophy Audio Translation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (validate without API calls)
    python main.py --input lecture.mp3 --voice_sample voice.mp3 --output output.mp3 --episode_id ep001 --dry_run
    
    # Full pipeline
    python main.py --input lecture.mp3 --voice_sample voice.mp3 --output output.mp3 --episode_id ep001
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input English audio file"
    )
    parser.add_argument(
        "--voice_sample", "-v",
        required=True,
        help="Path to voice sample for cloning (10s-5min)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for output Chinese audio file"
    )
    parser.add_argument(
        "--episode_id", "-e",
        required=True,
        help="Unique identifier for this episode (for term tracking)"
    )
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Validate pipeline logic without making API calls"
    )
    parser.add_argument(
        "--model", "-m",
        help="Specific model to use (e.g. deepseek/deepseek-r1)"
    )
    parser.add_argument(
        "--enable_search", "-s",
        action="store_true",
        default=True,
        help="Enable web search for term explanations (default: True)"
    )
    parser.add_argument(
        "--no_search",
        action="store_true",
        help="Disable web search for term explanations"
    )
    parser.add_argument(
        "--enable_reasoning", "-r",
        action="store_true",
        help="Enable reasoning mode (for reasoning models)"
    )
    parser.add_argument(
        "--polish_segment_chars",
        type=int,
        help="Soft target length (chars) for polishing segments; splits by paragraphs"
    )
    parser.add_argument(
        "--chinese_only_terms",
        help="Path to JSON file listing terms that should never have English annotations (default: data/chinese_only_terms.json)"
    )
    
    args = parser.parse_args()
    
    # Handle search flag logic
    enable_search = not args.no_search
    
    print("=" * 50)
    print("Philosophy Audio Translation Pipeline")
    print("=" * 50)
    print(f"Input:        {args.input}")
    print(f"Voice Sample: {args.voice_sample}")
    print(f"Output:       {args.output}")
    print(f"Episode ID:   {args.episode_id}")
    print(f"Dry Run:      {args.dry_run}")
    print(f"Model:        {args.model or 'Default'}")
    print(f"Web Search:   {enable_search}")
    print(f"Reasoning:    {args.enable_reasoning}")
    if args.polish_segment_chars:
        print(f"Polish Segs:  {args.polish_segment_chars} chars")
    
    try:
        results = run_pipeline(
            input_audio=args.input,
            voice_sample=args.voice_sample,
            output_audio=args.output,
            episode_id=args.episode_id,
            dry_run=args.dry_run,
            model=args.model,
            enable_search=enable_search,
            enable_reasoning=args.enable_reasoning,
            polish_segment_chars=args.polish_segment_chars,
            chinese_only_terms_path=args.chinese_only_terms
        )
        
        print(f"\nPipeline completed: {results['status']}")
        return 0
        
    except PipelineError as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
