"""
Step 2: Terminology Extraction & Translation

Extracts philosophy terms and translates text from Step 1 output.

Usage:
    python step2_translate.py --input ep001_transcription.json --episode_id ep001 [--output_dir output/]
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from config import validate_config, OPENROUTER_BASE_URL, TRANSLATION_MODEL, CHINESE_ONLY_TERMS_PATH
from translator import (
    translate_with_extraction,
    call_openrouter_api,
    dry_run_translate,
    dry_run_extract_terms,
    TranslationError
)
from terminology_db import TerminologyDB, create_default_philosophy_terms


# Summary generation prompt
SUMMARY_SYSTEM_PROMPT = """你是一位哲学内容摘要专家。请为以下中文哲学翻译文本生成简明摘要。

要求：
1. 摘要长度约200字（中文字符）
2. 突出主要哲学概念和论点
3. 保持学术性但易于理解
4. 不要添加原文没有的观点

请直接输出摘要内容，不要加任何前缀："""


def generate_summary(chinese_text: str, max_words: int = 200, model: str = None) -> str:
    """Generate a ~200 word summary of the translated text."""
    user_content = f"请为以下文本生成约{max_words}字的摘要：\n\n{chinese_text}"
    
    summary = call_openrouter_api(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.3,
        model=model
    )
    
    return summary.strip()


def initialize_terminology_db() -> TerminologyDB:
    """Initialize terminology database with default philosophy terms."""
    db = TerminologyDB()
    
    if len(db.terms) == 0:
        print("Initializing terminology database...")
        default_terms = create_default_philosophy_terms()
        for eng, chi in default_terms.items():
            db.add_term(eng, chi)
        db.save()
        print(f"  Added {len(default_terms)} terms")
    
    return db


def load_chinese_only_terms(path: str = None) -> list:
    """Load chinese-only terms from JSON file.
    
    These terms will never have English annotations (even on first occurrence).
    """
    terms_path = Path(path) if path else CHINESE_ONLY_TERMS_PATH
    if not terms_path.exists():
        return []
    
    try:
        with open(terms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        terms = data.get("terms", [])
        print(f"  Loaded {len(terms)} chinese-only terms from {terms_path.name}")
        return terms
    except Exception as e:
        print(f"  [Warning] Failed to load chinese-only terms: {e}")
        return []

def get_checkpoint_path(output_dir: Path, episode_id: str) -> Path:
    """Get the checkpoint file path for an episode."""
    return output_dir / f"{episode_id}_checkpoint.json"


def load_checkpoint(output_dir: Path, episode_id: str) -> dict:
    """Load checkpoint data if it exists."""
    checkpoint_path = get_checkpoint_path(output_dir, episode_id)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_checkpoint(output_dir: Path, episode_id: str, checkpoint_data: dict):
    """Save checkpoint data to file."""
    checkpoint_path = get_checkpoint_path(output_dir, episode_id)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def clear_checkpoint(output_dir: Path, episode_id: str):
    """Clear checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(output_dir, episode_id)
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def run_step2(
    input_json: str,
    episode_id: str,
    output_dir: str = None,
    dry_run: bool = False,
    model: str = None,
    enable_search: bool = True,
    enable_reasoning: bool = False,
    force: bool = False,
    polish_segment_chars: int = None,
    chinese_only_terms_path: str = None
) -> dict:
    """
    Run Step 2: Term extraction and translation.
    
    Args:
        input_json: Path to Step 1 transcription JSON
        episode_id: Episode identifier (for term tracking)
        output_dir: Directory to save output files
        dry_run: If True, validate logic without API calls
        model: Specific model to use (e.g., deepseek/deepseek-r1)
        enable_search: Enable web search for authoritative term explanations
        enable_reasoning: Enable reasoning mode (for reasoning models like deepseek-r1)
        force: If True, ignore checkpoints and start from scratch
        polish_segment_chars: Soft target length for polishing segments (paragraph-aligned)
    """
    input_path = Path(input_json)
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Transcription JSON not found: {input_path}")
    
    # Load transcription
    with open(input_path, "r", encoding="utf-8") as f:
        transcription_data = json.load(f)
    
    english_text = transcription_data.get("english_text", "")
    if not english_text:
        raise ValueError("No English text found in transcription JSON")
    
    # Initialize terminology database
    term_db = initialize_terminology_db()
    term_db.set_episode(episode_id)
    
    if dry_run:
        return run_dry_run(
            input_path, 
            english_text, 
            term_db,
            model=model,
            enable_search=enable_search,
            enable_reasoning=enable_reasoning
        )
    
    results = {
        "status": "pending",
        "episode_id": episode_id,
        "started_at": datetime.now().isoformat(),
    }
    
    # Load existing checkpoint if any (unless force is set)
    if force:
        clear_checkpoint(output_dir, episode_id)
        print("\n[INFO] Force mode: starting fresh (checkpoint cleared)")
        checkpoint = {}
    else:
        checkpoint = load_checkpoint(output_dir, episode_id)
        if checkpoint:
            print(f"\n[INFO] Found checkpoint, resuming from last successful step...")
    
    try:
        # Step 1/3: Extract terms and translate
        if checkpoint.get("step1_complete"):
            print("\n[1/3] Extracting terms and translating... [SKIPPED - already complete]")
            chinese_text = checkpoint["chinese_text"]
            draft_chinese_text = checkpoint.get("draft_chinese_text", chinese_text)
            extracted_terms = checkpoint["extracted_terms"]
            print(f"  Loaded from checkpoint: {len(chinese_text)} characters, {len(extracted_terms['terms'])} terms")
        else:
            print("\n[1/3] Extracting terms and translating...")
            print("  Extracting philosophy terms with AI search...")
            
            chinese_only_terms = load_chinese_only_terms(chinese_only_terms_path)
            
            translation_result = translate_with_extraction(
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
            chinese_text = translation_result["translation"]
            draft_chinese_text = translation_result.get("draft_translation", chinese_text)
            extracted_terms = translation_result["extracted_terms"]
            
            print(f"  Terms extracted: {len(extracted_terms['terms'])}")
            print(f"  New terms: {extracted_terms['new_terms_count']}")
            print(f"  Translated: {len(chinese_text)} characters")
            
            # Save checkpoint after Step 1
            checkpoint["step1_complete"] = True
            checkpoint["chinese_text"] = chinese_text
            checkpoint["draft_chinese_text"] = draft_chinese_text
            checkpoint["extracted_terms"] = extracted_terms
            save_checkpoint(output_dir, episode_id, checkpoint)
            print("  [Checkpoint saved]")
        
        # Step 2/3: Generate summary
        if checkpoint.get("step2_complete"):
            print("\n[2/3] Generating summary... [SKIPPED - already complete]")
            summary = checkpoint["summary"]
            print(f"  Loaded from checkpoint: {len(summary)} characters")
        else:
            print("\n[2/3] Generating summary (~200 words)...")
            summary = generate_summary(chinese_text, model=model)
            print(f"  Summary: {len(summary)} characters")
            
            # Save checkpoint after Step 2
            checkpoint["step2_complete"] = True
            checkpoint["summary"] = summary
            save_checkpoint(output_dir, episode_id, checkpoint)
            print("  [Checkpoint saved]")
        
        # Step 3/3: Save output files
        print("\n[3/3] Saving output files...")
        
        # JSON output for Step 3
        output_json_path = output_dir / f"{episode_id}_translation.json"
        output_data = {
            "episode_id": episode_id,
            "created_at": datetime.now().isoformat(),
            "english_text": english_text,
            "chinese_text": chinese_text,
            "draft_chinese_text": draft_chinese_text,
            "summary": summary,
            "extracted_terms": extracted_terms["terms"],
            "audio_duration": transcription_data.get("duration_seconds", 0)
        }
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Chinese text file
        txt_path = output_dir / f"{episode_id}_chinese.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(chinese_text)
        
        # Summary with terms
        summary_path = output_dir / f"{episode_id}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Episode: {episode_id}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Summary\n\n")
            f.write(summary)
            f.write("\n\n## Extracted Terms\n\n")
            for term in extracted_terms["terms"]:
                f.write(f"- {term['english']} ({term['chinese']})\n")
                if term.get("explanation"):
                    f.write(f"  解释: {term['explanation']}\n")
        
        # Save terminology database
        term_db.save()
        
        # Clear checkpoint on successful completion
        clear_checkpoint(output_dir, episode_id)
        print("  [Checkpoint cleared - all steps complete]")
        
        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()
        results["translation"] = {
            "text_length": len(chinese_text),
            "terms_count": len(extracted_terms["terms"]),
            "new_terms": extracted_terms.get("new_terms_count", 0)
        }
        results["output_files"] = {
            "json": str(output_json_path),
            "chinese_text": str(txt_path),
            "summary": str(summary_path)
        }
        
        print(f"\n  Output saved:")
        print(f"    JSON: {output_json_path}")
        print(f"    Text: {txt_path}")
        print(f"    Summary: {summary_path}")
        
    except TranslationError as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"\n[ERROR] Step failed: {e}")
        print(f"[INFO] Checkpoint saved. Re-run to resume from last successful step.")
        raise
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"\n[ERROR] Unexpected error: {e}")
        print(f"[INFO] Checkpoint saved. Re-run to resume from last successful step.")
        raise
    
    return results


def run_dry_run(
    input_path: Path, 
    english_text: str, 
    term_db: TerminologyDB,
    model: str = None,
    enable_search: bool = True,
    enable_reasoning: bool = False
) -> dict:
    """Run Step 2 in dry-run mode."""
    print("\n" + "=" * 50)
    print("DRY RUN MODE - Step 2: Terminology & Translation")
    print("=" * 50)
    
    results = {"mode": "dry_run", "status": "validating", "checks": {}}
    
    # Check 1: Input data
    print("\n[1/4] Checking input data...")
    results["checks"]["input"] = {
        "file_exists": input_path.exists(),
        "text_length": len(english_text)
    }
    print(f"  File exists: {input_path.exists()}")
    print(f"  Text length: {len(english_text)} chars")
    
    # Check 2: Terminology database
    print("\n[2/4] Checking terminology database...")
    results["checks"]["terminology"] = {"terms_loaded": len(term_db.terms)}
    print(f"  Terms loaded: {len(term_db.terms)}")
    
    # Check 3: Term extraction
    print("\n[3/4] Checking term extraction...")
    sample_text = english_text[:500] if len(english_text) > 500 else english_text
    extract_check = dry_run_extract_terms(sample_text, model=model, enable_search=enable_search)
    results["checks"]["term_extraction"] = extract_check
    results["checks"]["term_extraction"]["enable_reasoning"] = enable_reasoning
    print(f"  Web search: {extract_check['web_search_enabled']}")
    print(f"  Model: {extract_check['model']}")
    
    # Check 4: Translation
    print("\n[4/4] Checking translation...")
    translate_check = dry_run_translate(sample_text, model=model)
    results["checks"]["translator"] = translate_check
    print(f"  API endpoint: {OPENROUTER_BASE_URL}")
    print(f"  Model: {translate_check['model']}")
    
    # Summary
    print("\n" + "=" * 50)
    all_valid = input_path.exists() and len(english_text) > 0
    results["status"] = "valid" if all_valid else "issues_found"
    
    if all_valid:
        print("[OK] All Step 2 checks PASSED")
        print("\nRemove --dry_run to run translation")
    else:
        print("[FAIL] Some checks FAILED")
    
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Terminology Extraction & Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run
    python step2_translate.py --input ep001_transcription.json --episode_id ep001 --dry_run
    
    # Full processing
    python step2_translate.py --input ep001_transcription.json --episode_id ep001 --output_dir output/
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to Step 1 JSON output")
    parser.add_argument("--episode_id", "-e", required=True, help="Episode identifier")
    parser.add_argument("--output_dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--dry_run", "-d", action="store_true", help="Validate without API calls")
    parser.add_argument("--model", "-m", help="Specific model to use (e.g. deepseek/deepseek-r1)")
    parser.add_argument("--enable_search", "-s", action="store_true", default=True, help="Enable web search for term explanations (default: True)")
    parser.add_argument("--no_search", action="store_true", help="Disable web search for term explanations")
    parser.add_argument("--enable_reasoning", "-r", action="store_true", help="Enable reasoning mode (for reasoning models)")
    parser.add_argument("--force", "-f", action="store_true", help="Ignore checkpoints and start from scratch")
    parser.add_argument("--polish_segment_chars", type=int, help="Soft target length (chars) for polishing segments; splits by paragraphs")
    parser.add_argument("--chinese_only_terms", help="Path to JSON file listing terms that should never have English annotations (default: data/chinese_only_terms.json)")
    
    args = parser.parse_args()
    
    # Handle search flag logic
    enable_search = not args.no_search
    
    print("=" * 50)
    print("Step 2: Terminology Extraction & Translation")
    print("=" * 50)
    print(f"Input:      {args.input}")
    print(f"Episode ID: {args.episode_id}")
    print(f"Output Dir: {args.output_dir or 'same as input'}")
    print(f"Dry Run:    {args.dry_run}")
    print(f"Model:      {args.model or 'Default'}")
    print(f"Web Search: {enable_search}")
    print(f"Reasoning:  {args.enable_reasoning}")
    print(f"Force:      {args.force}")
    
    try:
        results = run_step2(
            input_json=args.input,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            model=args.model,
            enable_search=enable_search,
            enable_reasoning=args.enable_reasoning,
            force=args.force,
            polish_segment_chars=args.polish_segment_chars,
            chinese_only_terms_path=args.chinese_only_terms
        )
        
        print(f"\nStep 2 completed: {results['status']}")
        
        if results['status'] == 'success':
            print(f"\nNext: Run step3_audio.py with the translation JSON")
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Step 2 failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
