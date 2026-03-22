# Philosophy Audio Translation Pipeline

Automated three-step pipeline to translate English philosophy podcasts to Chinese. Currently for the podcast ([History of Philosophy Without Any Gaps](https://historyofphilosophy.net/)), which is my favorite philosophy podcast! Super thanks to Peter!!!

You are also welcome to subscribe to my [Xiaoyuzhoufm](https://www.xiaoyuzhoufm.com/podcast/6993470f11391268fd6847a7) to listen to the translated podcasts!

![workflow](resources/presentation_flowchart.png)


## Features

- **Whisper (or close-source model) Transcription** — Remote API via [speaches](https://github.com/speaches-ai/speaches)
- **LLM Cross-Validation** — Cross-validate 3 Whisper outputs for accuracy
- **AI Term Extraction** — Searches Wikipedia, Stanford Encyclopedia
- **~200 Word Summary** — Quick overview of content
- **Voice Cloning** — MiniMax API with speaker voice preservation

---

## Design Differences and Contributions

Compared to existing work and typical pipelines, this project focuses on the following aspects:

- **Single-audio processing**
  - Includes an explicit polishing stage to improve fluency and readability of the translated text
  - Supports domain-specific terminology translation:
    - Can integrate external knowledge bases (for example, Wikipedia or specialized databases)
    - Provides the original term in parentheses alongside the translation to aid understanding
    - Allows optional human-in-the-loop refinement

- **Multi-audio consistency**
  - Aligns translations across multiple audio sources to ensure consistent terminology
  - Prevents the same term from being translated into different meanings across episodes or segments

- **Non-real-time translation pipeline**
  - Focuses on quality rather than latency
  - Supports stronger TTS models (for example, MiniMax)
  - Allows downstream fine-tuning and customization of TTS models

---

## To-Do List

### Technical

- [ ] **Evaluation and analysis**
  - [ ] Assess whether the architecture and design choices are sufficiently effective
  - [ ] Translation quality evaluation
    - [ ] Metrics
      - [ ] COMET-Kiwi
      - [ ] LLM-as-a-judge (review official judge prompts from different sources)
      - [ ] BLEU
    - [ ] Model comparison
      - [ ] Analyze translation differences between GPT-4o and DeepSeek models
- [ ] **MiniMax API enhancements**
  - [ ] Multi-speaker conversation support
    - [ ] 2-speaker dialogue
    - [ ] Multi-speaker (3–5 speakers)
  - [ ] Incorporate tone, pauses, and other prosodic features — prompt the model to actively inject these during transcription

- [ ] **Fine-tune Qwen TTS model (and others)**
  - [ ] Single-speaker
    - [ ] Targeted fine-tuning on MiniMax-based philosophy podcast audio
    - [ ] Expand to more domains (LoRA fine-tuning for easy adapter swapping → leverage open-source audio datasets)
  - [ ] Multi-speaker

- [ ] **End-to-end translation**
  - [ ] Synchronize tone, voice, and intonation with the original audio

### Presentation

- [ ] **Desktop GUI** — click-and-drag interface to run the full pipeline (PC only)
  - [ ] LLM sidebar for interactive Q&A assistance
  - [ ] Core workflow integration

- [ ] **UI polish** *(TBD)*

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** [ffmpeg](https://ffmpeg.org/) must be installed on the system (required by `pydub`).

### 2. Configure `.env`

Fill in the `.env` file. You can ignore whisper_api if you use close source model (e.g. gpt-audio-mini), and it will use your Openrouter API. 

```env
# Whisper API (Remote Server)
# Deploy speaches on your GPU server:
#   docker run -d -p 8000:8000 --gpus all ghcr.io/speaches-ai/speaches:latest
WHISPER_API_URL=http://your-server:8000

# OpenRouter API — https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# MiniMax API — https://platform.minimax.io
MINIMAX_API_KEY=your_key
MINIMAX_BASE_URL=https://api.minimax.io/v1
```

**😢 Trust me, closed-source models are [much better](comparison.md) (and sometimes even cheaper...) than open-source ones if you do not fine-tune them...**

### 3. Validate Configuration

```bash
python config.py
```

---

## Three-Step Process

### Step 0 (Optional): Cross-Validate Whisper Transcriptions

If you have 3 transcription texts for the same audio, you can cross-validate them using an LLM to produce the most accurate transcript. Place your text files in the `input/` folder:

```
input/
├── ep001/
│   ├── 0.txt    # Whisper attempt 1
│   ├── 1.txt    # Whisper attempt 2
│   └── 2.txt    # Whisper attempt 3
```

**Standalone usage:**
```bash
python cross_validator.py --episode_id ep001 --output_dir output/ep001/
```

**Or via step1:**
```bash
python step1_transcribe.py --from_texts --episode_id ep001 --output_dir output/ep001/
```

The LLM compares all 3 versions, keeps majority-agreed text, and flags ambiguous parts in `cross_validation_warnings`.

**Output** (saved to `output/ep001/`):
- `ep001_transcription.json` — For Step 2 (includes warnings)
- `ep001_english.txt` — Plain text

---

### Step 1: Audio Transcription

**Whisper mode** (default):
```bash
python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --output_dir output/ep001/
```

**OpenRouter mode** (closed-source models, e.g. `gpt-audio-mini`):
```bash
python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --method openrouter --temperature 0.2 --output_dir output/ep001/
```

**Multi-transcription mode** (transcribe N times for cross-validation):
```bash
# Transcribe 3 times → saves input/ep001/0.txt, 1.txt, 2.txt
python step1_transcribe.py --input lecture.mp3 --episode_id ep001 --method openrouter --runs 3

# Then cross-validate
python step1_transcribe.py --from_texts --episode_id ep001 --output_dir output/ep001/
```

**Output** (saved to `output/ep001/`):
- `ep001_transcription.json` — For Step 2
- `ep001_english.txt` — English text

---

### Step 2: Terminology & Translation

```bash
python step2_translate.py --input output/ep001/ep001_transcription.json --episode_id ep001 --output_dir output/ep001/
```

**Output** (saved to `output/ep001/`):
- `ep001_translation.json` — For Step 3
- `ep001_chinese.txt` — Chinese translation
- `ep001_summary.txt` — ~200 word summary + extracted terms

---

### Step 3: Audio Generation

```bash
python step3_audio.py --input output/ep001/ep001_translation.json --voice_sample voice.mp3 --output output/ep001/ep001_chinese.mp3
```

**Output:**
- `ep001_chinese.mp3` — Final Chinese audio

---

## Complete Workflow Example

Step 1 has four mutually-exclusive modes — pick **one**:

```bash
# ── Step 1 option A: Cross-validate existing text files (LLM merges 3+ transcriptions) ──
python step1_transcribe.py --from_texts -e ep001 -o output/ep001/

# ── Step 1 option B: Transcribe audio directly via Whisper API ──
python step1_transcribe.py -i lecture.mp3 -e ep001 -o output/ep001/

# ── Step 1 option C: Transcribe audio directly via OpenRouter (closed-source model) ──
python step1_transcribe.py -i lecture.mp3 -e ep001 --method openrouter -o output/ep001/

# ── Step 1 option D: Multi-transcribe → cross-validate (recommended for accuracy) ──
python step1_transcribe.py -i lecture.mp3 -e ep001 --method openrouter --runs 3
python step1_transcribe.py --from_texts -e ep001 -o output/ep001/

# ── Step 2: Translate (same regardless of Step 1 option) ──
python step2_translate.py -i output/ep001/ep001_transcription.json -e ep001 -o output/ep001/

# ── Step 3: Generate Audio ──
python step3_audio.py -i output/ep001/ep001_translation.json -v voice.mp3 -o output/ep001/ep001_chinese.mp3
```

---

## Dry Run Mode

Add `--dry_run` (or `-d`) to any command to validate without making API calls:

```bash
python step1_transcribe.py -i lecture.mp3 -e ep001 --dry_run
python step2_translate.py -i output/ep001/ep001_transcription.json -e ep001 --dry_run
python step3_audio.py -i output/ep001/ep001_translation.json -v voice.mp3 -o output.mp3 --dry_run
```

## Legacy One-Command Pipeline

(Note that I update the code frequently, and this part is unverified)

`main.py` runs all three steps in a single command:

```bash
python main.py --input lecture.mp3 --voice_sample voice.mp3 --output output.mp3 --episode_id ep001
```

Additional options: `--dry_run`, `--model`, `--no_search`, `--enable_reasoning`, `--polish_segment_chars`, `--chinese_only_terms`.

---

## License

MIT
