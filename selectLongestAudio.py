import argparse
from pathlib import Path

import numpy as np
from datasets import Audio, DatasetDict
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the longest audio samples from a dataset."
    )
    parser.add_argument("--language-name", type=str, default="Hebrew")
    parser.add_argument("--language-code", type=str, default="he")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to DatasetDict on disk. Defaults to ./common_voice_subset/<language-code>",
    )
    parser.add_argument("--max-train-examples", type=int, default=20000)
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000, 15000],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output index files.",
    )
    return parser.parse_args()


def resolve_paths(args):
    project_root = Path(__file__).resolve().parent
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path is not None
        else project_root / "common_voice_subset" / args.language_code
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return dataset_path, output_dir


def load_train_split(dataset_path: Path, max_examples: int):
    dataset = DatasetDict.load_from_disk(str(dataset_path))
    train_size = min(max_examples, len(dataset["train"]))
    train_split = dataset["train"].select(range(train_size))
    return train_split.cast_column("audio", Audio(decode=True))


def compute_durations(dataset) -> np.ndarray:
    durations = []
    total_examples = len(dataset)

    print(f"Computing durations for {total_examples} examples...")

    for i in range(total_examples):
        example = dataset[i]
        audio = example["audio"]
        array = audio["array"]
        sample_rate = audio["sampling_rate"]

        duration = float(len(array) / sample_rate) if array is not None and sample_rate else 0.0
        durations.append(duration)

        if (i + 1) % 1000 == 0 or i + 1 == total_examples:
            print(f"Processed {i + 1}/{total_examples}")

    return np.asarray(durations, dtype=np.float64)


def save_top_k_indices(
    sorted_indices: np.ndarray,
    durations: np.ndarray,
    output_dir: Path,
    language_code: str,
    top_k_values: List[int],
):
    for k in top_k_values:
        selected_indices = sorted_indices[:k]
        output_path = output_dir / f"selected_longest_{language_code}_{k}.txt"

        np.savetxt(output_path, selected_indices, fmt="%d")

        print(
            f"Saved top {k} to {output_path} "
            f"(min={durations[selected_indices].min():.2f}s, "
            f"max={durations[selected_indices].max():.2f}s, "
            f"total={durations[selected_indices].sum() / 3600:.2f}h)"
        )


def main():
    args = parse_args()
    dataset_path, output_dir = resolve_paths(args)

    print(f"Language: {args.language_name}")
    print(f"Language code: {args.language_code}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output dir: {output_dir}")

    train_dataset = load_train_split(dataset_path, args.max_train_examples)
    print(train_dataset)

    durations = compute_durations(train_dataset)
    sorted_indices_desc = np.argsort(durations)[::-1]

    save_top_k_indices(
        sorted_indices=sorted_indices_desc,
        durations=durations,
        output_dir=output_dir,
        language_code=args.language_code,
        top_k_values=args.top_k_values,
    )


if __name__ == "__main__":
    main()