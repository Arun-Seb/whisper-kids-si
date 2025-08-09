# WhisperSI — Fine-Tuning Whisper for Children's Speech Intelligibility Assessment

> *Fine-tuning Whisper for automated speech intelligibility assessment in children with hearing loss.*

---

## Inspiration & Background

This repository is inspired by and directly extends the following publication:

> **Automated Speech Intelligibility Assessment Using AI-Based Transcription in Children with Cochlear Implants, Hearing Aids, and Normal Hearing**  
> Vicky W. Zhang, Arun Sebastian, Jessica J.M. Monaghan  
> *Journal of Clinical Medicine*, 2025, 14, 5280  
> https://doi.org/10.3390/jcm14155280

In that study, we evaluated several off-the-shelf speech-to-text (STT) models — including Whisper, Wav2Vec 2.0, DeepSpeech, S2T Transformer, and Google Speech Recognition — and found that **Whisper Large V2** achieved the highest transcription accuracy on children's speech, with agreement comparable to naïve human listeners (ICC > 0.95 in normal hearing and cochlear implant groups).

While the published study used Whisper Large V2 **without fine-tuning**, a natural next step was to explore whether adapting the model to children's speech through fine-tuning could further improve performance — particularly for children with hearing aids, where the off-the-shelf model showed higher word error rates. **This repository contains the fine-tuning experiments we tried as a follow-up to that work**, implemented via the Hugging Face Transformers library.

---

## Overview

This repository provides scripts to:

1. **Fine-tune** OpenAI's [Whisper](https://github.com/openai/whisper) model on children's speech data.
2. **Evaluate** a fine-tuned (or baseline) Whisper model using Word Error Rate (WER).

The goal is to improve upon the already strong baseline performance of Whisper Large V2 by adapting it to the acoustic characteristics of young children's speech — including higher fundamental frequency, shorter vocal tract length, and greater articulatory variability.

---

## Repository Structure

```
.
├── finetune_whisper_children.py   # Fine-tuning script
├── evaluate_whisper.py            # Inference + WER evaluation
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Data Format

Organise your dataset as follows:

```
data/
  train/
    audio/          # .wav files, 16 kHz mono
    metadata.csv    # columns: file_name, transcription
  test/
    audio/
    metadata.csv
```

Example `metadata.csv`:

```csv
file_name,transcription
child01_s01.wav,the boy walked to the table
child01_s02.wav,the dog is under the chair
```

> **Note:** The dataset used in the paper is not publicly available due to ethical restrictions. See the [Data Availability Statement](https://doi.org/10.3390/jcm14155280) in the publication.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Fine-Tuning

```bash
python finetune_whisper_children.py \
    --data_dir     ./data \
    --model_name   openai/whisper-large-v2 \
    --output_dir   ./whisper-children-finetuned \
    --epochs       5 \
    --batch_size   8 \
    --learning_rate 1e-5 \
    --fp16
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Root folder with `train/` and `test/` |
| `--model_name` | `openai/whisper-large-v2` | Hugging Face model id |
| `--output_dir` | `./whisper-children-finetuned` | Where to save checkpoints |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `8` | Per-device batch size |
| `--learning_rate` | `1e-5` | Learning rate |
| `--warmup_steps` | `100` | LR warmup steps |
| `--fp16` | `True` | Mixed-precision training |

Training metrics (loss, WER) are logged to TensorBoard:

```bash
tensorboard --logdir ./whisper-children-finetuned/runs
```

---

## Evaluation

```bash
# Evaluate with WER (requires metadata.csv with references)
python evaluate_whisper.py \
    --audio_dir  ./data/test/audio \
    --metadata   ./data/test/metadata.csv \
    --model_dir  ./whisper-children-finetuned \
    --output_csv results.csv

# Transcribe only (no reference transcriptions needed)
python evaluate_whisper.py \
    --audio_dir      ./my_recordings \
    --model_dir      openai/whisper-large-v2 \
    --transcribe_only
```

Per-file results are saved to `results.csv`.

---

## Model Performance (from the paper)

Whisper Large V2 evaluated on 5-year-old children's speech:

| Group | ICC (AI vs. Listeners) | WER — AI | WER — Human avg |
|---|---|---|---|
| Normal Hearing (NH) | 0.96 | 20.5% | ~16.8% |
| Cochlear Implant (CI) | 0.96 | 22.8% | ~20.0% |
| Hearing Aid (HA) | 0.91 | 40.7% | ~32.0% |

ICC values > 0.9 indicate *excellent* inter-rater agreement.

---

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2025whisper_children_si,
  title   = {Automated Speech Intelligibility Assessment Using AI-Based Transcription
             in Children with Cochlear Implants, Hearing Aids, and Normal Hearing},
  author  = {Zhang, Vicky W. and Sebastian, Arun and Monaghan, Jessica J.M.},
  journal = {Journal of Clinical Medicine},
  volume  = {14},
  number  = {15},
  pages   = {5280},
  year    = {2025},
  doi     = {10.3390/jcm14155280}
}
```

---



---

## License

MIT
