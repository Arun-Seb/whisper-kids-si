"""
Fine-tuning Whisper for Children's Speech Intelligibility Assessment
=====================================================================
Associated study:
  "Automated Speech Intelligibility Assessment in Children: Comparing AI-Based
   Transcription with Traditional Human Methods"
  Zhang, V.W., Sebastian, A., & Monaghan, J.J.M. (2025). J. Clin. Med., 14, 5280.
  https://doi.org/10.3390/jcm14155280

This script fine-tunes OpenAI Whisper (default: large-v2) on a children's speech
dataset using the Hugging Face Transformers library.

Usage
-----
  python finetune_whisper_children.py --data_dir ./data \
                                      --model_name openai/whisper-large-v2 \
                                      --output_dir ./whisper-children-finetuned \
                                      --epochs 5

Expected data_dir layout
------------------------
  data/
    train/
      audio/   *.wav  (16 kHz, mono)
      metadata.csv    (columns: file_name, transcription)
    test/
      audio/   *.wav
      metadata.csv

Requirements
------------
  pip install transformers datasets accelerate evaluate jiwer soundfile librosa
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads inputs and labels to the longest sequence in each batch."""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split audio inputs and labels (they have different padding needs)
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it is ignored in the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove bos token if it was appended during collation
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_children_dataset(data_dir: str, processor: WhisperProcessor) -> DatasetDict:
    """
    Load train/test splits from a local directory.

    Each split must have:
      - audio/ folder with .wav files (16 kHz mono)
      - metadata.csv with columns: file_name, transcription
    """
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_dir, "train", "metadata.csv"),
            "test": os.path.join(data_dir, "test", "metadata.csv"),
        },
    )

    # Prepend the audio directory path to each file_name
    def add_audio_path(split_name):
        audio_dir = os.path.join(data_dir, split_name, "audio")

        def _add_path(example):
            example["audio"] = os.path.join(audio_dir, example["file_name"])
            return example

        return _add_path

    dataset["train"] = dataset["train"].map(add_audio_path("train"))
    dataset["test"] = dataset["test"].map(add_audio_path("test"))

    # Cast the audio column so HF automatically resamples to 16 kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    return dataset


def prepare_dataset(batch, processor: WhisperProcessor):
    """Extract log-Mel features from raw audio and tokenise the transcription."""
    audio = batch["audio"]

    # Compute log-Mel spectrogram (input_features)
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    ).input_features[0]

    # Tokenise the reference transcription
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch


def compute_metrics(pred, tokenizer, metric):
    """Compute Word Error Rate (WER) for model evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token id before decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for children's speech"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory containing train/ and test/ subdirectories",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-large-v2",
        help="Hugging Face model id (e.g. openai/whisper-large-v2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./whisper-children-finetuned",
        help="Directory to save checkpoints and the final model",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language for transcription",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument(
        "--max_input_length_seconds",
        type=float,
        default=30.0,
        help="Discard training samples longer than this (Whisper max is 30 s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load processor (feature extractor + tokeniser)
    # ------------------------------------------------------------------
    print(f"Loading processor from: {args.model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )

    # ------------------------------------------------------------------
    # 2. Load and preprocess dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_children_dataset(args.data_dir, processor)

    # Filter out samples longer than Whisper's 30-second context window
    max_samples = int(args.max_input_length_seconds * 16_000)
    dataset = dataset.filter(
        lambda x: len(x["audio"]["array"]) <= max_samples,
        desc="Filtering long audio",
    )

    print("Preprocessing dataset (extracting features and tokenising)...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=1,
        desc="Preparing dataset",
    )

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Tell the model which language / task to use at inference time
    model.generation_config.language = args.language.lower()
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None  # let the processor handle this

    # ------------------------------------------------------------------
    # 4. Data collator and metric
    # ------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, wer_metric),
        tokenizer=processor.feature_extractor,
    )

    # ------------------------------------------------------------------
    # 7. Train and save
    # ------------------------------------------------------------------
    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
