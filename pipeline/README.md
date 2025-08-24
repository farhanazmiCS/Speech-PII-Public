# SpeechPII Pipeline – CLI

## Overview
This pipeline orchestrates:
1. **ASR transcription** (greedy or N-best)  
2. **LLM-based correction**  
3. **LLM-based PII tagging**  
4. **(Deprecated) Vosk word timestamps**  
5. **Alignment & PII extraction**  
6. **Evaluation** (index-based)

You can run end-to-end or pick specific steps. Each command reads/writes CSV/JSON so you can branch or resume anywhere.

---

## Installation

```bash
pip install typer pandas scikit-learn vosk
```

If you use Hugging Face Whisper or OpenAI LLMs, install their dependencies and set:

```bash
export OPENAI_API_KEY=your_api_key
```
⸻

Project Layout

```bash
.
├─ cli.py
├─ pipeline.py
├─ utils.py
├─ models/
│  └─ vosk-model-en-us-0.42-gigaspeech/
└─ data/
   ├─ transcriptions.csv
   ├─ transcriptions_corrected.csv
   ├─ transcriptions_tagged.csv
   └─ ...
```

⸻

CLI Usage

```bash
0. Help command

python cli.py --help
```
⸻

```bash
1. Transcribe

python cli.py transcribe --help

Options:
	•	--model-name TEXT Whisper model (default: f-azm17/whisper-small_en_seed_gretel_similar0.3-default-tokenizer)
	•	--llm-model TEXT LLM model (default: gpt-4o)
	•	--audio-dirs TEXT... Audio directories
	•	--n-best INT Number of hypotheses (default: 1)
	•	--out-csv TEXT Output CSV (default: data/transcriptions.csv)

Example:

python cli.py transcribe \
  --model-name f-azm17/whisper-small_en_... \
  --n-best 5 \
  --out-csv data/transcriptions_5best.csv
```
⸻

```bash
2. Correct

python cli.py correct --help

Options:
	•	--in-csv TEXT Input CSV (default: data/transcriptions.csv)
	•	--out-csv TEXT Output CSV (default: data/transcriptions_corrected.csv)

Example:

python cli.py correct \
  --in-csv data/transcriptions.csv \
  --out-csv data/transcriptions_corrected.csv
```

⸻

```bash
3. Tag

python cli.py tag --help

Options:
	•	--in-csv TEXT Input CSV (default: data/transcriptions_corrected.csv)
	•	--text-col TEXT Column with text (default: transcript)
	•	--method TEXT Tagging method (default: zero_shot_icl)
	•	--out-csv TEXT Output CSV (default: data/transcriptions_tagged.csv)

Example:

python cli.py tag \
  --in-csv data/transcriptions_corrected.csv \
  --method few_shot_cot \
  --out-csv data/transcriptions_tagged_few_shot_cot.csv
```

⸻

```bash
4. Vosk Timestamps [Deprecated]

python cli.py vosk --help

Options:
	•	--in-csv TEXT Input CSV with file column (default: data/transcriptions.csv)
	•	--audio-col TEXT Column name for audio path (default: file)
	•	--vosk-model-dir TEXT Path to Vosk model (default: models/vosk-model-en-us-0.42-gigaspeech)
	•	--out-json TEXT Output JSON (default: ../../data/vosk_output/vosk_output.json)

Example:

python cli.py vosk \
  --in-csv data/transcriptions.csv \
  --vosk-model-dir models/vosk-model-en-us-0.42-gigaspeech \
  --out-json data/vosk_output.json
```

⸻

```bash
5. Extract Clean Text & PII Tuples

python cli.py extract --help

Arguments:
	•	input_csv (must contain file, tagged)
	•	output_csv (written with file, clean_text, pii_tuples)

Options:
	•	--allowed-labels/-l Labels to retain (default: EMAIL, NRIC, CREDIT_CARD, PHONE, PASSPORT_NUM, BANK_ACCOUNT, CAR_PLATE, PERSON)

Example:

python cli.py extract \
  data/transcriptions_tagged.csv \
  data/transcriptions_extracted.csv \
  -l EMAIL -l PHONE
```

⸻

```bash
6. Extract Using Timestamps [Deprecated]

python cli.py extract-using-timestamps --help

Options:
	•	--method TEXT Extraction method
	•	--in-csv TEXT Input CSV
	•	--out-csv TEXT Output CSV
	•	--vosk-json TEXT JSON with word timestamps (default: ../../data/vosk_output/vosk_output.json)

Example:

python cli.py extract-using-timestamps \
  --method align \
  --in-csv data/transcriptions_tagged.csv \
  --out-csv data/triplets.csv \
  --vosk-json data/vosk_output.json
```

⸻

```bash
7. Evaluate (Time-Offset Tolerance) [Deprecated]

python cli.py evaluate --help

Options:
	•	--gt-csv TEXT Ground-truth CSV (default: ../../data/triplets/ref_triplets_500.csv)
	•	--pred-csv TEXT Predictions CSV (default: ../../data/triplets/triplets_no_correct_zero_shot_icl.csv)
	•	--tolerance FLOAT Time-offset tolerance (default: 5)

Example:

python cli.py evaluate \
  --gt-csv data/ref_triplets.csv \
  --pred-csv data/pred_triplets.csv \
  --tolerance 2.0
```

⸻

```bash
8. Evaluate by Index (Character-Level)

python cli.py evaluate-index --help

Arguments:
	•	true_csv Ground-truth CSV (with pii_tuples)
	•	pred_csv Predictions CSV (with pii_tuples)

Options:
	•	--allowed-labels/-l Labels to count (comma-separated or multiple flags)
	•	--tolerance INT Character index tolerance (default: 5)

Example:

python cli.py evaluate-index \
  data/ref_extracted.csv \
  data/pred_extracted.csv \
  -l EMAIL -l PHONE \
  --tolerance 3
```

⸻


