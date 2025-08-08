"""
Evaluate / Transcribe with a Fine-tuned Whisper Model
======================================================
Associated study:
  Zhang, V.W., Sebastian, A., & Monaghan, J.J.M. (2025). J. Clin. Med., 14, 5280.
  https://doi.org/10.3390/jcm14155280

Transcribes audio files using a fine-tuned (or base) Whisper model and
reports per-file and overall Word Error Rate (WER).

Usage
-----
  # Evaluate against reference transcriptions
  python evaluate_whisper.py --audio_dir ./data/test/audio \
                             --metadata   ./data/test/metadata.csv \
                             --model_dir  ./whisper-children-finetuned

  # Transcribe only (no reference required)
  python evaluate_whisper.py --audio_dir ./my_recordings \
                             --model_dir  openai/whisper-large-v2 \
                             --transcribe_only
"""

import argparse
import csv
import os

import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from evaluate import load as load_metric


def load_metadata(metadata_csv: str):
    """Return a dict mapping file_name -> reference transcription."""
    mapping = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["file_name"]] = row["transcription"].strip().lower()
    return mapping


def transcribe_file(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: torch.device,
) -> str:
    """Load a WAV file and return the Whisper transcription (lowercase, stripped)."""
    audio, sr = librosa.load(audio_path, sr=16_000, mono=True)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features,
            language="english",
            task="transcribe",
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip().lower()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on children's speech"
    )
    parser.add_argument(
        "--audio_dir", required=True, help="Folder containing .wav files"
    )
    parser.add_argument(
        "--metadata", default=None, help="CSV with file_name,transcription columns"
    )
    parser.add_argument(
        "--model_dir",
        default="./whisper-children-finetuned",
        help="Path to fine-tuned model (or a Hugging Face model id)",
    )
    parser.add_argument(
        "--output_csv",
        default="results.csv",
        help="Where to save per-file transcription results",
    )
    parser.add_argument(
        "--transcribe_only",
        action="store_true",
        help="Skip WER computation (no reference transcriptions needed)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model from: {args.model_dir}")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load references if available
    references = {}
    if args.metadata and not args.transcribe_only:
        references = load_metadata(args.metadata)

    # Gather audio files
    audio_files = sorted(
        f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav")
    )
    if not audio_files:
        print(f"No .wav files found in {args.audio_dir}")
        return

    results = []
    predictions, ground_truths = [], []

    for fname in audio_files:
        audio_path = os.path.join(args.audio_dir, fname)
        hypothesis = transcribe_file(audio_path, model, processor, device)
        reference = references.get(fname, "")

        row = {"file": fname, "hypothesis": hypothesis, "reference": reference}
        results.append(row)
        predictions.append(hypothesis)
        if reference:
            ground_truths.append(reference)

        ref_display = f"  REF : {reference}" if reference else ""
        print(f"[{fname}]\n  HYP : {hypothesis}{ref_display}\n")

    # Compute overall WER
    if ground_truths and not args.transcribe_only:
        wer_metric = load_metric("wer")
        overall_wer = 100 * wer_metric.compute(
            predictions=predictions[: len(ground_truths)],
            references=ground_truths,
        )
        print(f"{'='*50}")
        print(f"Overall WER : {overall_wer:.2f}%  ({len(ground_truths)} files)")
        print(f"{'='*50}")

    # Save CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "hypothesis", "reference"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
