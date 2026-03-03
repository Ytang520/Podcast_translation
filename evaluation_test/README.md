# WER Evaluation

Evaluate the Word Error Rate (WER) of different ASR transcription models against a human-verified reference script.

## Directory Structure

```
evaluation_test/
├── audio.mp3                          # Raw podcast audio (you add this)
├── reference_script.txt               # Human-verified ground truth transcript
├── evaluate_wer.py                    # Evaluation script
├── README.md                          # This file
└── script/
    ├── whisper_large_v3/              # Whisper Large V3 transcriptions
    │   ├── run_1.txt                  # Transcription attempt 1
    │   ├── run_2.txt                  # Transcription attempt 2
    │   └── run_3.txt                  # Transcription attempt 3
    └── canary_1b_v2/                  # Canary 1B V2 transcriptions
        ├── run_1.txt
        ├── run_2.txt
        └── run_3.txt
```

## Setup

```bash
pip install jiwer
```

## Usage

### 1. Prepare your transcription files

Replace each placeholder `run_X.txt` with the actual transcription output from the corresponding model. Each file should contain plain text — the script will automatically:

- Strip any `# ...` header lines at the top
- Merge multiple lines into a single sentence
- Lowercase, remove punctuation, and collapse whitespace before comparison

### 2. Dry run (validate structure)

```bash
python evaluate_wer.py --dry_run
```

This checks the directory structure, verifies the reference file, and runs internal logic tests — **no `jiwer` needed**.

### 3. Run evaluation

```bash
python evaluate_wer.py
```

Outputs a summary table with per-model **mean WER**, **variance**, **std dev**, and detailed per-run breakdown (substitutions, insertions, deletions).

### 4. Save results to JSON

```bash
python evaluate_wer.py --output results.json
```

### 5. Custom paths

```bash
python evaluate_wer.py --reference path/to/ref.txt --script_dir path/to/script/
```

## Output Example

```
================================================================================
WER EVALUATION RESULTS
================================================================================
Reference: reference_script.txt
Reference word count: 3093

Model                     Mean WER     Variance     Std Dev      Min        Max        Runs
---------------------------------------------------------------------------------------
whisper_large_v3          0.0520       0.000036     0.0060       0.0450     0.0580     3
canary_1b_v2              0.0780       0.000100     0.0100       0.0680     0.0900     3
```

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **WER** | `(S + I + D) / N` | Word Error Rate — primary metric |
| **MER** | `(S + I + D) / (S + I + D + C)` | Match Error Rate |
| **WIL** | `1 - (C/N × C/P)` | Word Information Lost |

Where: **S** = substitutions, **I** = insertions, **D** = deletions, **C** = correct matches, **N** = reference words, **P** = hypothesis words

## Adding More Models

Simply create a new subdirectory under `script/` with `.txt` files inside:

```bash
mkdir script/my_new_model
# Add run_1.txt, run_2.txt, run_3.txt with transcription text
```

The script auto-discovers all model subdirectories.
