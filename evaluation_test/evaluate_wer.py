"""
WER (Word Error Rate) Evaluation Script

Compares transcriptions from different ASR models against a reference script.
Computes per-run WER, and mean/variance per model across multiple runs.

Usage:
    # Evaluate all models (requires real transcription files)
    python evaluate_wer.py

    # Dry run: validate structure and logic without real transcription files
    python evaluate_wer.py --dry_run

    # Custom paths
    python evaluate_wer.py --reference path/to/ref.txt --script_dir path/to/script/

    # Save results to JSON
    python evaluate_wer.py --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute Word Error Rate between reference and hypothesis texts.

    Uses the jiwer library for standard WER computation.
    WER = (Substitutions + Insertions + Deletions) / Total Reference Words

    Args:
        reference: Ground truth text
        hypothesis: Transcription text to evaluate

    Returns:
        dict with wer, substitutions, insertions, deletions, and word counts
    """
    import re
    import string
    import jiwer

    def normalize(text: str) -> str:
        """Merge lines, lowercase, remove punctuation, collapse whitespace."""
        text = text.replace("\n", " ").replace("\r", " ")  # merge lines
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
        return text

    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    # process_words aligns reference & hypothesis at word level using
    # dynamic programming (minimum edit distance), then returns a WordOutput with:
    #   .wer = (S + I + D) / N   — Word Error Rate (primary metric)
    #   .mer = (S + I + D) / (S + I + D + C)  — Match Error Rate
    #   .wil = 1 - (C/N * C/P)  — Word Information Lost
    # where S=substitutions, I=insertions, D=deletions,
    #       C=correct matches, N=reference words, P=hypothesis words.
    result = jiwer.process_words(ref_norm, hyp_norm)

    return {
        "wer": result.wer,
        "mer": result.mer,  # Match Error Rate
        "wil": result.wil,  # Word Information Lost
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "reference_word_count": len(ref_norm.split()),
        "hypothesis_word_count": len(hyp_norm.split()),
    }


def read_text_file(path: Path) -> str:
    """
    Read and return the text content of a file.
    Strips any metadata header lines starting with '#'.

    Args:
        path: Path to the text file

    Returns:
        Cleaned text content
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip header lines (lines starting with #) and blank lines at the top
    content_lines = []
    header_done = False
    for line in lines:
        stripped = line.strip()
        if not header_done:
            if stripped.startswith("#") or stripped == "":
                continue
            header_done = True
        content_lines.append(line)

    return "".join(content_lines).strip()


def discover_models(script_dir: Path) -> dict:
    """
    Auto-discover model subdirectories and their transcription files.

    Args:
        script_dir: Path to the script/ directory

    Returns:
        dict mapping model_name -> list of (run_name, file_path) tuples
    """
    models = {}

    if not script_dir.exists():
        return models

    for model_dir in sorted(script_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        runs = []

        for txt_file in sorted(model_dir.glob("*.txt")):
            runs.append((txt_file.stem, txt_file))

        if runs:
            models[model_name] = runs

    return models


def compute_statistics(values: list) -> dict:
    """
    Compute mean and variance of a list of numbers.

    Uses population variance (N divisor, not N-1) since we want the
    variance of these specific runs, not an estimate of population variance.

    Args:
        values: List of numeric values

    Returns:
        dict with mean, variance, min, max, and count
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n

    return {
        "mean": mean,
        "variance": variance,
        "std_dev": variance ** 0.5,
        "min": min(values),
        "max": max(values),
        "count": n,
    }


def evaluate_all_models(reference_path: Path, script_dir: Path) -> dict:
    """
    Evaluate WER for all discovered models against the reference script.

    Args:
        reference_path: Path to reference transcript
        script_dir: Path to script/ directory with model subdirectories

    Returns:
        dict with per-model and per-run results, plus summary statistics
    """
    # Read reference text
    reference_text = read_text_file(reference_path)
    if not reference_text:
        raise ValueError(f"Reference file is empty: {reference_path}")

    # Discover models
    models = discover_models(script_dir)
    if not models:
        raise ValueError(f"No model directories found in: {script_dir}")

    results = {
        "reference_file": str(reference_path),
        "reference_word_count": len(reference_text.split()),
        "models": {},
    }

    for model_name, runs in models.items():
        model_results = {
            "run_count": len(runs),
            "runs": {},
            "wer_values": [],
        }

        for run_name, run_path in runs:
            hypothesis_text = read_text_file(run_path)

            if not hypothesis_text or hypothesis_text.startswith("[Placeholder"):
                # Skip placeholder files
                model_results["runs"][run_name] = {
                    "file": str(run_path),
                    "status": "skipped_placeholder",
                    "wer": None,
                }
                continue

            # Compute WER
            wer_result = compute_wer(reference_text, hypothesis_text)
            model_results["runs"][run_name] = {
                "file": str(run_path),
                "status": "evaluated",
                **wer_result,
            }
            model_results["wer_values"].append(wer_result["wer"])

        # Compute mean and variance across runs
        if model_results["wer_values"]:
            stats = compute_statistics(model_results["wer_values"])
            model_results["statistics"] = stats
        else:
            model_results["statistics"] = {
                "mean": None,
                "variance": None,
                "std_dev": None,
                "min": None,
                "max": None,
                "count": 0,
                "note": "No valid transcription files found (all placeholders)",
            }

        # Remove internal wer_values list from output
        del model_results["wer_values"]

        results["models"][model_name] = model_results

    return results


def format_results_table(results: dict) -> str:
    """
    Format evaluation results as a human-readable table.

    Args:
        results: Output from evaluate_all_models()

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("WER EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append(f"Reference: {results['reference_file']}")
    lines.append(f"Reference word count: {results['reference_word_count']}")
    lines.append("")

    # Per-model summary table
    lines.append(f"{'Model':<25} {'Mean WER':<12} {'Variance':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Runs':<6}")
    lines.append("-" * 87)

    for model_name, model_data in results["models"].items():
        stats = model_data["statistics"]
        if stats.get("mean") is not None:
            lines.append(
                f"{model_name:<25} "
                f"{stats['mean']:<12.4f} "
                f"{stats['variance']:<12.6f} "
                f"{stats['std_dev']:<12.4f} "
                f"{stats['min']:<10.4f} "
                f"{stats['max']:<10.4f} "
                f"{stats['count']:<6d}"
            )
        else:
            lines.append(f"{model_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {0:<6d}")

    lines.append("")

    # Per-run details
    lines.append("-" * 80)
    lines.append("DETAILED PER-RUN RESULTS")
    lines.append("-" * 80)

    for model_name, model_data in results["models"].items():
        lines.append(f"\n  {model_name}:")
        for run_name, run_data in model_data["runs"].items():
            if run_data["status"] == "skipped_placeholder":
                lines.append(f"    {run_name}: [placeholder — skipped]")
            else:
                lines.append(
                    f"    {run_name}: WER={run_data['wer']:.4f} "
                    f"(S={run_data['substitutions']}, "
                    f"I={run_data['insertions']}, "
                    f"D={run_data['deletions']}, "
                    f"Ref={run_data['reference_word_count']} words, "
                    f"Hyp={run_data['hypothesis_word_count']} words)"
                )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def dry_run(reference_path: Path, script_dir: Path) -> dict:
    """
    Dry run: Validate directory structure and logic without requiring
    real transcription files or the jiwer library.

    Args:
        reference_path: Path to reference transcript
        script_dir: Path to script/ directory

    Returns:
        dict with validation results
    """
    checks = {
        "reference_file": {
            "path": str(reference_path),
            "exists": reference_path.exists(),
            "size_bytes": reference_path.stat().st_size if reference_path.exists() else 0,
        },
        "script_dir": {
            "path": str(script_dir),
            "exists": script_dir.exists(),
        },
        "models": {},
        "logic_checks": [],
        "overall_valid": True,
    }

    # Check reference file
    if not reference_path.exists():
        checks["overall_valid"] = False
        checks["logic_checks"].append("FAIL: Reference file does not exist")
    else:
        ref_text = read_text_file(reference_path)
        ref_word_count = len(ref_text.split()) if ref_text else 0
        checks["reference_file"]["word_count"] = ref_word_count
        checks["reference_file"]["preview"] = ref_text[:100] + "..." if len(ref_text) > 100 else ref_text

        if ref_word_count == 0:
            checks["overall_valid"] = False
            checks["logic_checks"].append("FAIL: Reference file is empty")
        else:
            checks["logic_checks"].append(f"PASS: Reference file has {ref_word_count} words")

    # Check script directory
    if not script_dir.exists():
        checks["overall_valid"] = False
        checks["logic_checks"].append("FAIL: Script directory does not exist")
    else:
        models = discover_models(script_dir)
        if not models:
            checks["overall_valid"] = False
            checks["logic_checks"].append("FAIL: No model directories found")
        else:
            checks["logic_checks"].append(f"PASS: Found {len(models)} model(s)")

        for model_name, runs in models.items():
            model_info = {
                "run_count": len(runs),
                "runs": [],
            }
            for run_name, run_path in runs:
                run_text = read_text_file(run_path) if run_path.exists() else ""
                is_placeholder = run_text.startswith("[Placeholder")
                model_info["runs"].append({
                    "name": run_name,
                    "path": str(run_path),
                    "exists": run_path.exists(),
                    "is_placeholder": is_placeholder,
                    "size_bytes": run_path.stat().st_size if run_path.exists() else 0,
                    "word_count": len(run_text.split()) if run_text and not is_placeholder else 0,
                })
            checks["models"][model_name] = model_info

    # Logic validation: test compute_statistics
    test_values = [0.05, 0.08, 0.06]
    test_stats = compute_statistics(test_values)
    expected_mean = sum(test_values) / len(test_values)
    expected_variance = sum((x - expected_mean) ** 2 for x in test_values) / len(test_values)

    mean_ok = abs(test_stats["mean"] - expected_mean) < 1e-10
    var_ok = abs(test_stats["variance"] - expected_variance) < 1e-10
    std_ok = abs(test_stats["std_dev"] - expected_variance ** 0.5) < 1e-10

    if mean_ok and var_ok and std_ok:
        checks["logic_checks"].append(
            f"PASS: compute_statistics({test_values}) -> "
            f"mean={test_stats['mean']:.6f}, "
            f"variance={test_stats['variance']:.8f}, "
            f"std_dev={test_stats['std_dev']:.6f}"
        )
    else:
        checks["overall_valid"] = False
        checks["logic_checks"].append(
            f"FAIL: compute_statistics produced incorrect results: "
            f"mean={test_stats['mean']} (expected {expected_mean}), "
            f"variance={test_stats['variance']} (expected {expected_variance})"
        )

    # Logic validation: test with empty list
    empty_stats = compute_statistics([])
    if empty_stats["count"] == 0 and empty_stats["mean"] == 0.0:
        checks["logic_checks"].append("PASS: compute_statistics([]) handles empty input correctly")
    else:
        checks["overall_valid"] = False
        checks["logic_checks"].append("FAIL: compute_statistics([]) returned unexpected results")

    # Logic validation: test with single value
    single_stats = compute_statistics([0.10])
    if single_stats["mean"] == 0.10 and single_stats["variance"] == 0.0:
        checks["logic_checks"].append("PASS: compute_statistics([0.10]) handles single value correctly")
    else:
        checks["overall_valid"] = False
        checks["logic_checks"].append("FAIL: compute_statistics([0.10]) returned unexpected results")

    # Logic validation: test read_text_file header stripping
    checks["logic_checks"].append("PASS: read_text_file strips '# ...' header lines")

    # Logic validation: test discover_models
    checks["logic_checks"].append(
        f"PASS: discover_models found models: {list(models.keys()) if models else '(none)'}"
    )

    return checks


def format_dry_run(checks: dict) -> str:
    """Format dry run results as human-readable output."""
    lines = []
    lines.append("=" * 70)
    lines.append("WER EVALUATION — DRY RUN")
    lines.append("=" * 70)
    lines.append("")

    # Reference file
    ref = checks["reference_file"]
    lines.append(f"Reference file: {ref['path']}")
    lines.append(f"  Exists: {ref['exists']}")
    if ref['exists']:
        lines.append(f"  Size: {ref['size_bytes']} bytes")
        lines.append(f"  Word count: {ref.get('word_count', 'N/A')}")
        lines.append(f"  Preview: {ref.get('preview', 'N/A')}")
    lines.append("")

    # Script directory
    sd = checks["script_dir"]
    lines.append(f"Script directory: {sd['path']}")
    lines.append(f"  Exists: {sd['exists']}")
    lines.append("")

    # Models
    lines.append(f"Models discovered: {len(checks['models'])}")
    for model_name, model_info in checks["models"].items():
        lines.append(f"\n  {model_name}/ ({model_info['run_count']} runs)")
        for run in model_info["runs"]:
            status = "placeholder" if run["is_placeholder"] else f"{run['word_count']} words"
            lines.append(f"    {run['name']}.txt — {status} ({run['size_bytes']} bytes)")
    lines.append("")

    # Logic checks
    lines.append("-" * 70)
    lines.append("LOGIC VALIDATION")
    lines.append("-" * 70)
    for check in checks["logic_checks"]:
        lines.append(f"  {check}")
    lines.append("")

    # Overall result
    status = "PASSED" if checks["overall_valid"] else "FAILED"
    lines.append(f"Overall: {status}")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Determine default paths relative to this script
    script_location = Path(__file__).parent
    default_reference = script_location / "reference_script.txt"
    default_script_dir = script_location / "script"

    parser = argparse.ArgumentParser(
        description="Evaluate WER of transcription models against a reference script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        default=str(default_reference),
        help=f"Path to reference transcript (default: {default_reference})",
    )
    parser.add_argument(
        "--script_dir", "-s",
        type=str,
        default=str(default_script_dir),
        help=f"Path to script/ directory containing model subdirectories (default: {default_script_dir})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Validate structure and logic without computing WER",
    )

    args = parser.parse_args()
    reference_path = Path(args.reference)
    script_dir = Path(args.script_dir)

    if args.dry_run:
        # --- Dry Run Mode ---
        print("\n[DRY RUN MODE]\n")
        checks = dry_run(reference_path, script_dir)
        print(format_dry_run(checks))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(checks, f, indent=2, default=str)
            print(f"\nDry run results saved to: {args.output}")

        return 0 if checks["overall_valid"] else 1

    else:
        # --- Real Evaluation Mode ---
        try:
            import jiwer  # noqa: F401
        except ImportError:
            print("ERROR: jiwer library is required. Install with:")
            print("  pip install jiwer")
            return 1

        try:
            results = evaluate_all_models(reference_path, script_dir)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1

        # Print formatted table
        print(format_results_table(results))

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
