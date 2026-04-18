import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import evaluate
import numpy as np
import torch
import transformers
from datasets import Audio, DatasetDict, concatenate_datasets
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


transformers.logging.set_verbosity_error()

MODEL_NAME = "openai/whisper-small"
AUDIO_SAMPLING_RATE = 16000
TOKENIZER_MAX_LENGTH = 448


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Whisper on Common Voice subsets."
    )

    parser.add_argument("--language-name", type=str, default="Hebrew")
    parser.add_argument("--language-code", type=str, default="he")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a DatasetDict saved with load_from_disk(). Defaults to ./common_voice_subset/<language-code>",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Directory where run outputs will be saved.",
    )
    parser.add_argument(
        "--subset-indices-path",
        type=str,
        default="kcenter/selected_indices_cosinek5000.txt",
        help="Optional path to a txt file containing selected train indices.",
    )
    parser.add_argument(
        "--disable-subset-selection",
        action="store_true",
        help="Use the full selected train split without applying subset indices.",
    )

    parser.add_argument("--max-train-examples", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--validation-start", type=int, default=2000)
    parser.add_argument("--validation-end", type=int, default=3000)

    parser.add_argument("--preprocess-chunk-size", type=int, default=10000)

    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--save-eval-steps", type=int, default=2400)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--early-stopping-patience", type=int, default=3)

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 training. By default, fp16 is enabled automatically on CUDA.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even if CUDA is available.",
    )

    return parser.parse_args()


def resolve_paths(args):
    project_root = Path(__file__).resolve().parent

    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path is not None
        else project_root / "common_voice_subset" / args.language_code
    )

    subset_indices_path = (
        Path(args.subset_indices_path)
        if args.subset_indices_path is not None
        else None
    )
    if subset_indices_path is not None and not subset_indices_path.is_absolute():
        subset_indices_path = project_root / subset_indices_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    return dataset_path, subset_indices_path, output_root


def load_dataset_splits(
    dataset_path: Path,
    max_train_examples: int,
    test_size: int,
    validation_start: int,
    validation_end: int,
    subset_indices_path: Optional[Path],
    disable_subset_selection: bool,
) -> DatasetDict:
    dataset = DatasetDict.load_from_disk(str(dataset_path))

    print(f"Loaded train set: {len(dataset['train'])} samples")
    print(f"Loaded test set: {len(dataset['test'])} samples")

    dataset["validation"] = dataset["test"].select(
        range(validation_start, min(validation_end, len(dataset["test"])))
    )
    dataset["test"] = dataset["test"].select(range(min(test_size, len(dataset["test"]))))
    dataset["train"] = dataset["train"].select(
        range(min(max_train_examples, len(dataset["train"])))
    )

    if not disable_subset_selection and subset_indices_path is not None and subset_indices_path.exists():
        selected_indices = np.loadtxt(subset_indices_path, dtype=int)
        dataset["train"] = dataset["train"].select(selected_indices)
        print(f"Applied subset indices from: {subset_indices_path}")
        print(f"Selected {len(selected_indices)} train examples")
    elif disable_subset_selection:
        print("Subset selection disabled. Using full selected train split.")
    else:
        print("No valid subset indices file found. Using full selected train split.")

    return dataset


def build_tokenizer_and_processor(language_name: str):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME,
        language=language_name,
        task="transcribe",
        truncation=True,
        model_max_length=TOKENIZER_MAX_LENGTH,
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=language_name,
        task="transcribe",
    )
    return feature_extractor, tokenizer, processor


def preview_tokenization(dataset: DatasetDict, tokenizer: WhisperTokenizer) -> None:
    input_text = dataset["train"][0]["sentence"]
    labels = tokenizer(input_text).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_text}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_without_special}")
    print(f"Are equal:             {input_text == decoded_without_special}")


def prepare_dataset_fn(
    batch: Dict[str, Any],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> Dict[str, Any]:
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    batch["labels"] = tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=TOKENIZER_MAX_LENGTH,
    ).input_ids

    return batch


def process_in_chunks(dataset, prepare_fn, chunk_size: int):
    processed_dataset = None

    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dataset))
        chunk = dataset.select(range(start_idx, end_idx))

        processed_chunk = chunk.map(
            prepare_fn,
            remove_columns=chunk.column_names,
            num_proc=1,
        )

        if processed_dataset is None:
            processed_dataset = processed_chunk
        else:
            processed_dataset = concatenate_datasets([processed_dataset, processed_chunk])

    return processed_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class RunStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training is starting.")


def compute_metrics_builder(tokenizer: WhisperTokenizer):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


def build_output_dir(output_root: Path, language_code: str) -> Path:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    return output_root / f"{timestamp}_{language_code}"


def should_use_fp16(args) -> bool:
    if args.no_fp16:
        return False
    if args.fp16:
        return True
    return torch.cuda.is_available()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path, subset_indices_path, output_root = resolve_paths(args)

    print("Device:", device)
    print("Language:", args.language_name)
    print("Language code:", args.language_code)
    print("Dataset path:", dataset_path)

    dataset = load_dataset_splits(
        dataset_path=dataset_path,
        max_train_examples=args.max_train_examples,
        test_size=args.test_size,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        subset_indices_path=subset_indices_path,
        disable_subset_selection=args.disable_subset_selection,
    )
    print(dataset)

    feature_extractor, tokenizer, processor = build_tokenizer_and_processor(args.language_name)
    preview_tokenization(dataset, tokenizer)

    print("Casting audio column...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))

    prepare_fn = lambda batch: prepare_dataset_fn(batch, feature_extractor, tokenizer)

    dataset["train"] = process_in_chunks(
        dataset["train"], prepare_fn, args.preprocess_chunk_size
    )
    print("Done preprocessing train split.")

    dataset["test"] = process_in_chunks(
        dataset["test"], prepare_fn, args.preprocess_chunk_size
    )
    print("Done preprocessing test split.")

    dataset["validation"] = process_in_chunks(
        dataset["validation"], prepare_fn, args.preprocess_chunk_size
    )
    print("Done preprocessing validation split.")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = args.language_name.lower()
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    output_dir = build_output_dir(output_root, args.language_code)
    print("Output dir:", output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=args.save_eval_steps,
        eval_steps=args.save_eval_steps,
        gradient_checkpointing=True,
        fp16=should_use_fp16(args),
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            RunStartCallback(),
        ],
    )

    if torch.cuda.is_available():
        print("Using GPU")
        torch.cuda.empty_cache()
        model.to(device)

    print("Starting initial evaluation...")
    trainer.evaluate()

    start_time = datetime.now()
    print("Starting training...")
    trainer.train()
    end_time = datetime.now()

    print(f"Training took {(end_time - start_time).total_seconds() / 60:.2f} minutes")

    # trainer.save_model(str(output_dir)) # uncomment if you want to save the model


    print("Final evaluation on the test set...")
    final_metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"Final evaluation metrics: {final_metrics}")


if __name__ == "__main__":
    main()