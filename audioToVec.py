import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, DatasetDict
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor


MODEL_NAME = "openai/whisper-small"
AUDIO_SAMPLING_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Whisper encoder embeddings from audio samples."
    )
    parser.add_argument("--language-name", type=str, default="Danish")
    parser.add_argument("--language-code", type=str, default="da")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to DatasetDict on disk. Defaults to ./common_voice_subset/<language-code>",
    )
    parser.add_argument("--max-train-examples", type=int, default=20000)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of dataset rows processed per outer loop chunk.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output .npy path. Defaults to ./whisper_train_embeddings_<language-name-lower>.npy",
    )
    return parser.parse_args()


def resolve_paths(args):
    project_root = Path(__file__).resolve().parent

    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path is not None
        else project_root / "common_voice_subset" / args.language_code
    )
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = project_root / f"whisper_train_embeddings_{args.language_name.lower()}.npy"

    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return dataset_path, output_path


def extract_embedding(example, processor: WhisperProcessor, model: WhisperModel, device: torch.device):
    audio = example["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(input_features)

    embedding = encoder_outputs.last_hidden_state.mean(dim=1)
    return {"embedding": embedding.cpu().numpy().flatten()}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path, output_path = resolve_paths(args)

    print("Device:", device)
    print("Language:", args.language_name)
    print("Language code:", args.language_code)
    print("Dataset path:", dataset_path)
    print("Output path:", output_path)

    common_voice = DatasetDict.load_from_disk(str(dataset_path))

    print(f"Loaded train set: {len(common_voice['train'])} samples")
    print(f"Loaded test set: {len(common_voice['test'])} samples")

    train_size = min(args.max_train_examples, len(common_voice["train"]))
    common_voice["train"] = common_voice["train"].select(range(train_size))
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME).to(device).eval()

    all_embeddings = []

    train_dataset = common_voice["train"]
    for start_idx in tqdm(range(0, len(train_dataset), args.chunk_size)):
        end_idx = min(start_idx + args.chunk_size, len(train_dataset))
        batch = train_dataset.select(range(start_idx, end_idx))

        processed_batch = batch.map(
            lambda example: extract_embedding(example, processor, model, device),
            remove_columns=batch.column_names,
        )
        all_embeddings.extend(processed_batch["embedding"])

    all_embeddings = np.asarray(all_embeddings)
    print("Embeddings shape:", all_embeddings.shape)

    np.save(output_path, all_embeddings)
    print(f"Saved embeddings to: {output_path}")


if __name__ == "__main__":
    main()