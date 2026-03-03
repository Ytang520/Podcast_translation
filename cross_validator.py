"""
Cross-Validator: LLM-based cross-validation of multiple Whisper transcriptions.

Reads 3 Whisper transcription text files (0.txt, 1.txt, 2.txt) from input/epxxx/,
uses an LLM to cross-validate them by majority agreement and contextual reasoning,
and produces the final epxxx_transcription.json for Step 2.

Adaptive mode (--adaptive): start with 3 files; if too many mismatches are
detected (≥ 3 warnings), escalate to 4th then 5th file if they exist.

This is a standalone module — it can be run independently without the Whisper model.

Usage:
    python cross_validator.py --episode_id ep001 [--input_dir input/] [--output_dir output/ep001/] [--model ...] [--adaptive]
"""
import argparse
import sys
import json
import re
from pathlib import Path
from datetime import datetime

from config import (
    INPUT_DIR,
    CROSS_VALIDATION_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    validate_config,
)
from translator import call_openrouter_api


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Adaptive escalation threshold: if warnings >= this, try more files
ADAPTIVE_WARNING_THRESHOLD = 3

CROSS_VALIDATION_SYSTEM_PROMPT = """你是一位专业的英文音频转录校对专家。现在有同一段音频的三个不同的 Whisper 转录版本。
由于语音识别模型的局限性，每个版本可能存在不同的错误。

请你仔细对比这三个版本，按照以下规则生成最终的正确文稿：
1. 保留三个版本中共同正确的部分（即三个版本一致的内容直接保留）
2. 当某部分有两个版本相同、一个不同时，采用两个相同版本的内容
3. 当某部分三个版本都不同时，根据上下文语义、语法正确性和逻辑连贯性判断最可能正确的内容，并在 warnings 中记录该处的信息
4. 修正明显的语音识别错误（如专有名词拼写、标点符号等）
5. 保持原文的段落结构和格式

请以JSON格式输出，格式如下：
{
  "final_text": "最终校对后的完整文稿",
  "warnings": [
    {
      "location": "相关文本位置的简短描述或引用",
      "original_texts": ["版本1的该部分内容", "版本2的该部分内容", "版本3的该部分内容"],
      "chosen_text": "最终选择的内容",
      "reason": "选择该版本的原因"
    }
  ]
}

如果没有三个版本都不同的情况，则 warnings 为 null。
请只输出JSON，不要添加任何其他说明或markdown代码块标记。"""


CROSS_VALIDATION_SYSTEM_PROMPT_4 = """你是一位专业的英文音频转录校对专家。现在有同一段音频的四个不同的 Whisper 转录版本。
由于语音识别模型的局限性，每个版本可能存在不同的错误。

请你仔细对比这四个版本，按照以下规则生成最终的正确文稿：
1. 保留四个版本中共同正确的部分（即四个版本一致的内容直接保留）
2. 当某部分有三个或以上版本相同时，采用多数版本的内容
3. 当某部分有两个版本相同、另外两个各不同时，优先采用两个相同版本的内容
4. 当某部分四个版本都不同时，根据上下文语义、语法正确性和逻辑连贯性判断最可能正确的内容，并在 warnings 中记录该处的信息
5. 修正明显的语音识别错误（如专有名词拼写、标点符号等）
6. 保持原文的段落结构和格式

请以JSON格式输出，格式如下：
{
  "final_text": "最终校对后的完整文稿",
  "warnings": [
    {
      "location": "相关文本位置的简短描述或引用",
      "original_texts": ["版本1的该部分内容", "版本2的该部分内容", "版本3的该部分内容", "版本4的该部分内容"],
      "chosen_text": "最终选择的内容",
      "reason": "选择该版本的原因"
    }
  ]
}

如果没有无法通过多数决定的情况，则 warnings 为 null。
请只输出JSON，不要添加任何其他说明或markdown代码块标记。"""


CROSS_VALIDATION_SYSTEM_PROMPT_5 = """你是一位专业的英文音频转录校对专家。现在有同一段音频的五个不同的 Whisper 转录版本。
由于语音识别模型的局限性，每个版本可能存在不同的错误。

请你仔细对比这五个版本，按照以下规则生成最终的正确文稿：
1. 保留五个版本中共同正确的部分（即五个版本一致的内容直接保留）
2. 当某部分有三个或以上版本相同时，采用多数版本的内容
3. 当某部分只有两个版本相同，其余各不同时，优先采用两个相同版本的内容，并结合上下文判断
4. 当某部分五个版本都不同或无法通过多数决定时，根据上下文语义、语法正确性和逻辑连贯性判断最可能正确的内容，并在 warnings 中记录该处的信息
5. 修正明显的语音识别错误（如专有名词拼写、标点符号等）
6. 保持原文的段落结构和格式

请以JSON格式输出，格式如下：
{
  "final_text": "最终校对后的完整文稿",
  "warnings": [
    {
      "location": "相关文本位置的简短描述或引用",
      "original_texts": ["版本1的该部分内容", "版本2的该部分内容", "版本3的该部分内容", "版本4的该部分内容", "版本5的该部分内容"],
      "chosen_text": "最终选择的内容",
      "reason": "选择该版本的原因"
    }
  ]
}

如果没有无法通过多数决定的情况，则 warnings 为 null。
请只输出JSON，不要添加任何其他说明或markdown代码块标记。"""


# Map number of texts to the corresponding prompt
_PROMPT_BY_COUNT = {
    3: CROSS_VALIDATION_SYSTEM_PROMPT,
    4: CROSS_VALIDATION_SYSTEM_PROMPT_4,
    5: CROSS_VALIDATION_SYSTEM_PROMPT_5,
}

# Chinese number words for building dynamic user messages
_CN_NUMBERS = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

class CrossValidationError(Exception):
    """Exception raised when cross-validation fails."""
    pass


def load_whisper_texts(input_dir: Path, episode_id: str) -> list:
    """
    Load 3 Whisper transcription text files from input_dir/episode_id/.

    Args:
        input_dir: Root input directory (e.g. input/)
        episode_id: Episode identifier (e.g. ep001)

    Returns:
        list of 3 transcription strings [text_0, text_1, text_2]

    Raises:
        FileNotFoundError: If the episode directory or any text file is missing
        ValueError: If any text file is empty
    """
    episode_dir = input_dir / episode_id
    if not episode_dir.exists():
        raise FileNotFoundError(
            f"Episode directory not found: {episode_dir}"
        )

    texts = []
    for i in range(3):
        txt_path = episode_dir / f"{i}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Transcription file not found: {txt_path}"
            )

        content = txt_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(
                f"Transcription file is empty: {txt_path}"
            )

        texts.append(content)

    return texts


def load_available_whisper_texts(input_dir: Path, episode_id: str, max_count: int = 5) -> list:
    """
    Load up to *max_count* Whisper transcription text files (0.txt … 4.txt).

    The first 3 files are mandatory.  Files 3.txt and 4.txt are optional;
    they are loaded only when they exist and are non-empty.

    Args:
        input_dir: Root input directory (e.g. input/)
        episode_id: Episode identifier (e.g. ep001)
        max_count: Maximum number of files to attempt (default 5)

    Returns:
        list of transcription strings (length 3–5)

    Raises:
        FileNotFoundError: If the episode directory or any of the first 3 files is missing
        ValueError: If any of the first 3 files is empty
    """
    episode_dir = input_dir / episode_id
    if not episode_dir.exists():
        raise FileNotFoundError(
            f"Episode directory not found: {episode_dir}"
        )

    texts = []
    for i in range(max_count):
        txt_path = episode_dir / f"{i}.txt"
        if i < 3:
            # First 3 are mandatory
            if not txt_path.exists():
                raise FileNotFoundError(
                    f"Transcription file not found: {txt_path}"
                )
            content = txt_path.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError(
                    f"Transcription file is empty: {txt_path}"
                )
            texts.append(content)
        else:
            # 4th and 5th are optional
            if not txt_path.exists():
                break
            content = txt_path.read_text(encoding="utf-8").strip()
            if not content:
                break  # skip empty optional files
            texts.append(content)

    return texts


def _extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response, handling cases where the model
    wraps the output in markdown code fences or adds preamble text.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict with 'final_text' and 'warnings' keys

    Raises:
        CrossValidationError: If JSON cannot be extracted or is invalid
    """
    text = response.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences: ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block (greedy from first { to last })
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise CrossValidationError(
        f"Failed to extract valid JSON from LLM response. "
        f"Response preview: {text[:300]}..."
    )


def _has_too_many_mismatches(warnings) -> bool:
    """
    Decide whether the warnings from a cross-validation round indicate
    too many unresolvable mismatches — meaning we should escalate to
    more transcription files (if available).

    Returns True when the number of warnings >= ADAPTIVE_WARNING_THRESHOLD.
    """
    if not warnings:
        return False
    return len(warnings) >= ADAPTIVE_WARNING_THRESHOLD


def cross_validate_transcriptions(
    texts: list,
    model: str = None
) -> dict:
    """
    Cross-validate 3–5 Whisper transcriptions using an LLM.

    The LLM compares the versions, keeps common correct parts,
    adopts majority agreement, and uses contextual reasoning for
    ambiguous parts (emitting warnings for those).

    Args:
        texts: List of 3, 4, or 5 transcription strings
        model: LLM model to use (default: CROSS_VALIDATION_MODEL from config)

    Returns:
        dict with keys:
            - 'final_text' (str): The corrected transcription
            - 'warnings' (list or None): Warnings for ambiguous parts

    Raises:
        CrossValidationError: If the LLM call or response parsing fails
    """
    n = len(texts)
    if n not in (3, 4, 5):
        raise CrossValidationError(
            f"Expected 3, 4, or 5 transcription texts, got {n}"
        )

    model = model or CROSS_VALIDATION_MODEL
    system_prompt = _PROMPT_BY_COUNT[n]

    # Build user message dynamically for N versions
    cn_total = _CN_NUMBERS.get(n, str(n))
    parts = [f"以下是同一段音频的{cn_total}个 Whisper 转录版本：\n"]
    for i, text in enumerate(texts):
        parts.append(f"\n=== 版本 {i + 1} ===\n{text}\n")
    parts.append(f"\n请根据系统提示中的规则，对比这{cn_total}个版本并输出最终校对结果的JSON。")
    user_content = "\n".join(parts)

    print(f"  Model: {model}")
    print(f"  Sending {n} transcriptions for cross-validation...")

    try:
        response = call_openrouter_api(
            system_prompt=system_prompt,
            user_content=user_content,
            model=model,
            temperature=0.1,  # Low temperature for accuracy
            max_tokens=53600
        )
    except Exception as e:
        raise CrossValidationError(f"LLM API call failed: {e}")

    # Parse the JSON response
    result = _extract_json_from_response(response)

    # Validate required fields
    if "final_text" not in result:
        raise CrossValidationError(
            "LLM response JSON missing 'final_text' field"
        )

    final_text = result["final_text"]
    warnings = result.get("warnings", None)

    # Normalize: treat empty list as None
    if isinstance(warnings, list) and len(warnings) == 0:
        warnings = None

    print(f"  Cross-validation complete: {len(final_text)} characters")
    if warnings:
        print(f"  Warnings: {len(warnings)} ambiguous parts detected")
    else:
        print(f"  Warnings: None (all parts resolved by majority agreement)")

    return {
        "final_text": final_text,
        "warnings": warnings,
    }


def save_transcription_json(
    final_text: str,
    episode_id: str,
    warnings,
    output_dir: Path,
    source_files: list = None,
) -> Path:
    """
    Save the cross-validated transcription as JSON for Step 2.

    The output schema is compatible with step2_translate.py, which
    reads 'english_text' from the JSON.

    Args:
        final_text: The corrected transcription text
        episode_id: Episode identifier
        warnings: List of warning dicts or None
        output_dir: Directory to save the output JSON
        source_files: Optional list of source file paths for metadata

    Returns:
        Path to the saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json_path = output_dir / f"{episode_id}_transcription.json"

    output_data = {
        "episode_id": episode_id,
        "created_at": datetime.now().isoformat(),
        "source": "cross_validation",
        "english_text": final_text,
        "cross_validation_warnings": warnings,
    }

    if source_files:
        output_data["source_files"] = [str(p) for p in source_files]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Also save plain text for reference
    txt_path = output_dir / f"{episode_id}_english.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Episode: {episode_id}\n")
        f.write(f"# Source: Cross-validated from 3 Whisper transcriptions\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write(final_text)

    return output_json_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cross_validation(
    episode_id: str,
    input_dir: str = None,
    output_dir: str = None,
    model: str = None,
    dry_run: bool = False,
    adaptive: bool = False,
) -> dict:
    """
    Run the full cross-validation workflow.

    Args:
        episode_id: Episode identifier (e.g. ep001)
        input_dir: Root input directory (default: config INPUT_DIR)
        output_dir: Output directory (default: output/episode_id/)
        model: LLM model override
        dry_run: If True, validate inputs without calling API
        adaptive: If True, escalate to 4th/5th file on high mismatch

    Returns:
        dict with status, output paths, and warnings
    """
    input_dir = Path(input_dir) if input_dir else INPUT_DIR
    output_dir = Path(output_dir) if output_dir else (
        Path("output") / episode_id
    )

    if dry_run:
        return _run_dry_run(input_dir, episode_id, output_dir, model, adaptive=adaptive)

    results = {
        "status": "pending",
        "episode_id": episode_id,
        "started_at": datetime.now().isoformat(),
        "adaptive": adaptive,
    }

    try:
        # Step 1/3: Load transcription texts
        print("\n[1/3] Loading Whisper transcription files...")
        print(f"  Input: {input_dir / episode_id}")

        if adaptive:
            all_texts = load_available_whisper_texts(input_dir, episode_id)
            print(f"  Adaptive mode: {len(all_texts)} file(s) available")
        else:
            all_texts = load_whisper_texts(input_dir, episode_id)

        for i, t in enumerate(all_texts):
            print(f"  {i}.txt: {len(t)} characters")

        # Step 2/3: Cross-validate with LLM (with adaptive escalation)
        print("\n[2/3] Cross-validating transcriptions with LLM...")

        # Start with 3 files
        use_count = 3
        texts_to_use = all_texts[:use_count]
        cv_result = cross_validate_transcriptions(texts_to_use, model=model)

        # Adaptive escalation loop
        if adaptive:
            while (
                _has_too_many_mismatches(cv_result["warnings"])
                and use_count < len(all_texts)
            ):
                use_count += 1
                texts_to_use = all_texts[:use_count]
                warn_count = len(cv_result["warnings"])
                print(
                    f"\n  ⚠ {warn_count} warnings (≥ {ADAPTIVE_WARNING_THRESHOLD}) "
                    f"— escalating to {use_count} files..."
                )
                cv_result = cross_validate_transcriptions(
                    texts_to_use, model=model
                )

            results["files_used"] = use_count

        final_text = cv_result["final_text"]
        warnings = cv_result["warnings"]

        # Step 3/3: Save output
        print("\n[3/3] Saving output files...")
        source_files = [
            input_dir / episode_id / f"{i}.txt" for i in range(use_count)
        ]
        output_json_path = save_transcription_json(
            final_text=final_text,
            episode_id=episode_id,
            warnings=warnings,
            output_dir=output_dir,
            source_files=source_files,
        )

        txt_path = output_dir / f"{episode_id}_english.txt"

        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()
        results["final_text_length"] = len(final_text)
        results["warnings_count"] = len(warnings) if warnings else 0
        results["output_files"] = {
            "json": str(output_json_path),
            "text": str(txt_path),
        }

        print(f"\n  Output saved:")
        print(f"    JSON: {output_json_path}")
        print(f"    Text: {txt_path}")

        if warnings:
            print(f"\n  ⚠ {len(warnings)} warning(s) — review cross_validation_warnings in the JSON")

    except (FileNotFoundError, ValueError, CrossValidationError) as e:
        results["status"] = "failed"
        results["error"] = str(e)
        raise

    return results


def _run_dry_run(
    input_dir: Path,
    episode_id: str,
    output_dir: Path,
    model: str = None,
    adaptive: bool = False,
) -> dict:
    """Run cross-validation in dry-run mode (validate inputs only)."""
    print("\n" + "=" * 50)
    print("DRY RUN MODE - Cross-Validation")
    if adaptive:
        print("  (Adaptive escalation ENABLED)")
    print("=" * 50)

    results = {"mode": "dry_run", "status": "validating", "adaptive": adaptive, "checks": {}}

    # Check 1: Configuration
    print("\n[1/3] Checking configuration...")
    validate_config()
    model = model or CROSS_VALIDATION_MODEL
    results["checks"]["config"] = {
        "api_url": OPENROUTER_BASE_URL,
        "model": model,
        "api_key_set": bool(OPENROUTER_API_KEY),
    }
    print(f"  API URL: {OPENROUTER_BASE_URL}")
    print(f"  Model: {model}")
    print(f"  API key set: {bool(OPENROUTER_API_KEY)}")

    # Check 2: Input files
    print("\n[2/3] Checking input files...")
    episode_dir = input_dir / episode_id
    dir_exists = episode_dir.exists()
    results["checks"]["input_dir"] = {
        "path": str(episode_dir),
        "exists": dir_exists,
    }
    print(f"  Episode dir: {episode_dir}")
    print(f"  Dir exists: {dir_exists}")

    # Check mandatory files (0–2) and optional files (3–4)
    max_files = 5 if adaptive else 3
    files_info = []
    all_files_ok = True
    available_count = 0
    for i in range(max_files):
        txt_path = episode_dir / f"{i}.txt"
        exists = txt_path.exists()
        size = txt_path.stat().st_size if exists else 0
        mandatory = i < 3
        info = {
            "file": f"{i}.txt",
            "exists": exists,
            "size_bytes": size,
            "mandatory": mandatory,
        }

        if exists and size > 0:
            content = txt_path.read_text(encoding="utf-8").strip()
            info["preview"] = content[:100] + ("..." if len(content) > 100 else "")
            info["char_count"] = len(content)
            available_count += 1
        elif mandatory:
            all_files_ok = False

        files_info.append(info)
        tag = "(mandatory)" if mandatory else "(optional)"
        status = "OK" if exists and size > 0 else "MISSING/EMPTY"
        print(f"  {i}.txt: {status} ({size} bytes) {tag}")

    results["checks"]["files"] = files_info
    results["checks"]["available_count"] = available_count

    if adaptive:
        print(f"\n  Adaptive: {available_count} file(s) available for escalation")
        print(f"  Escalation threshold: ≥ {ADAPTIVE_WARNING_THRESHOLD} warnings")

    # Check 3: Output directory
    print("\n[3/3] Checking output directory...")
    results["checks"]["output"] = {
        "path": str(output_dir),
        "exists": output_dir.exists(),
        "target_file": f"{episode_id}_transcription.json",
    }
    print(f"  Output dir: {output_dir}")
    print(f"  Target: {episode_id}_transcription.json")

    # Summary
    print("\n" + "=" * 50)
    all_valid = dir_exists and all_files_ok and bool(OPENROUTER_API_KEY)
    results["status"] = "valid" if all_valid else "issues_found"

    if all_valid:
        print("[OK] All cross-validation checks PASSED")
        print("\nRemove --dry_run to run cross-validation")
    else:
        print("[FAIL] Some checks FAILED")

    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Validator: LLM-based cross-validation of Whisper transcriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run — validate inputs
    python cross_validator.py --episode_id ep001 --dry_run

    # Full cross-validation
    python cross_validator.py --episode_id ep001

    # Adaptive: escalate to 4th/5th file on high mismatch
    python cross_validator.py --episode_id ep001 --adaptive

    # Custom directories
    python cross_validator.py --episode_id ep001 --input_dir my_input/ --output_dir my_output/ep001/

    # Use a specific model
    python cross_validator.py --episode_id ep001 --model deepseek/deepseek-r1
        """
    )

    parser.add_argument(
        "--episode_id", "-e", required=True,
        help="Episode identifier (e.g. ep001)"
    )
    parser.add_argument(
        "--input_dir", "-i",
        help=f"Root input directory containing episode subfolders (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Output directory (default: output/<episode_id>/)"
    )
    parser.add_argument(
        "--model", "-m",
        help=f"LLM model for cross-validation (default: {CROSS_VALIDATION_MODEL})"
    )
    parser.add_argument(
        "--dry_run", "-d", action="store_true",
        help="Validate inputs without calling API"
    )
    parser.add_argument(
        "--adaptive", "-a", action="store_true",
        help="Escalate to 4th/5th transcription file when too many mismatches are detected"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Cross-Validator: Whisper Transcription Cross-Validation")
    print("=" * 50)
    print(f"Episode ID: {args.episode_id}")
    print(f"Input Dir:  {args.input_dir or str(INPUT_DIR)}")
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

        print(f"\nCross-validation completed: {results['status']}")

        if results["status"] == "success":
            print(f"\nNext: Run step2_translate.py with the JSON output")

        return 0

    except Exception as e:
        print(f"\n[FAIL] Cross-validation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
