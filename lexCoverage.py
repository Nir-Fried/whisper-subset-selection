import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict

from typing import List, Tuple, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute lexical coverage and diversity for full, random, and selected subsets."
    )
    parser.add_argument("--language-code", type=str, default="he")
    parser.add_argument("--text-field", type=str, default="sentence")
    parser.add_argument("--max-train-examples", type=int, default=20000)
    parser.add_argument("--random-subset-size", type=int, default=2000)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to DatasetDict on disk. Defaults to ./common_voice_subset/<language-code>",
    )
    parser.add_argument(
        "--indices-path",
        type=str,
        default="kcenter/selected_indices_cosinek2000.txt",
        help="Optional subset index file.",
    )
    parser.add_argument(
        "--save-csv-path",
        type=str,
        default=None,
        help="Optional path to save the summary table as CSV.",
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

    indices_path = Path(args.indices_path) if args.indices_path is not None else None
    if indices_path is not None and not indices_path.is_absolute():
        indices_path = project_root / indices_path

    save_csv_path = Path(args.save_csv_path) if args.save_csv_path is not None else None
    if save_csv_path is not None and not save_csv_path.is_absolute():
        save_csv_path = project_root / save_csv_path

    return dataset_path, indices_path, save_csv_path


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def summarize_word_diversity(texts: List[str]) -> dict:
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))

    vocab = set(tokens)
    total_tokens = len(tokens)
    unique_words = len(vocab)
    type_token_ratio = unique_words / total_tokens if total_tokens > 0 else 0.0

    return {
        "num_sentences": len(texts),
        "total_tokens": total_tokens,
        "unique_words": unique_words,
        "type_token_ratio": type_token_ratio,
        "vocab": vocab,
    }


def compute_coverage(subset_vocab: set, full_vocab: set) -> float:
    return len(subset_vocab & full_vocab) / len(full_vocab) if full_vocab else 0.0


def load_texts(dataset_path: Path, text_field: str, max_examples: int) -> List[str]:
    dataset = DatasetDict.load_from_disk(str(dataset_path))
    train_size = min(max_examples, len(dataset["train"]))
    dataset["train"] = dataset["train"].select(range(train_size))
    return dataset["train"][text_field]


def evaluate_random_subsets(
    texts: List[str],
    full_vocab: set,
    random_subset_size: int,
    random_seeds: List[int],
) -> Tuple[pd.Series, pd.Series]:
    metrics = []

    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=random_subset_size, replace=False)
        subset_texts = [texts[i] for i in sorted(indices)]

        subset_metrics = summarize_word_diversity(subset_texts)
        subset_metrics["coverage_vs_full"] = compute_coverage(subset_metrics["vocab"], full_vocab)
        metrics.append(subset_metrics)

        print(
            f"Random seed {seed}: "
            f"{subset_metrics['unique_words']} unique words, "
            f"coverage {subset_metrics['coverage_vs_full']:.3f}"
        )

    metrics_df = pd.DataFrame(
        [
            {
                "unique_words": m["unique_words"],
                "coverage_vs_full": m["coverage_vs_full"],
                "type_token_ratio": m["type_token_ratio"],
            }
            for m in metrics
        ]
    )

    return metrics_df.mean(), metrics_df.std()


def evaluate_selected_subset(texts: List[str], indices_path: Optional[Path], full_vocab: set):
    if indices_path is None or not indices_path.exists():
        print(f"Warning: indices file not found: {indices_path}")
        return None, None

    indices = np.loadtxt(indices_path, dtype=int)
    subset_texts = [texts[i] for i in indices]

    metrics = summarize_word_diversity(subset_texts)
    metrics["coverage_vs_full"] = compute_coverage(metrics["vocab"], full_vocab)

    print(
        f"Selected subset: {metrics['unique_words']} unique words, "
        f"coverage {metrics['coverage_vs_full']:.3f}"
    )

    return metrics, indices


def build_results_table(
    full_metrics: dict,
    random_mean: pd.Series,
    random_std: pd.Series,
    selected_metrics: Optional[dict],
    selected_indices,
    max_train_examples: int,
    random_subset_size: int,
) -> pd.DataFrame:
    rows = [
        {
            "subset": f"full_{max_train_examples}",
            "unique_words": full_metrics["unique_words"],
            "coverage_vs_full": 1.0,
            "type_token_ratio": full_metrics["type_token_ratio"],
        },
        {
            "subset": f"random_{random_subset_size} (mean±std)",
            "unique_words": f"{random_mean['unique_words']:.1f} ± {random_std['unique_words']:.1f}",
            "coverage_vs_full": f"{random_mean['coverage_vs_full']:.3f} ± {random_std['coverage_vs_full']:.3f}",
            "type_token_ratio": f"{random_mean['type_token_ratio']:.4f} ± {random_std['type_token_ratio']:.4f}",
        },
    ]

    if selected_metrics is not None and selected_indices is not None:
        rows.append(
            {
                "subset": f"selected_{len(selected_indices)}",
                "unique_words": selected_metrics["unique_words"],
                "coverage_vs_full": selected_metrics["coverage_vs_full"],
                "type_token_ratio": selected_metrics["type_token_ratio"],
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dataset_path, indices_path, save_csv_path = resolve_paths(args)

    print(f"Loading dataset from: {dataset_path}")
    texts = load_texts(dataset_path, args.text_field, args.max_train_examples)

    print(f"Analyzing full {args.max_train_examples} dataset...")
    full_metrics = summarize_word_diversity(texts)
    print(f"Full set: {full_metrics['unique_words']} unique words.")

    random_mean, random_std = evaluate_random_subsets(
        texts=texts,
        full_vocab=full_metrics["vocab"],
        random_subset_size=args.random_subset_size,
        random_seeds=args.random_seeds,
    )

    selected_metrics, selected_indices = evaluate_selected_subset(
        texts=texts,
        indices_path=indices_path,
        full_vocab=full_metrics["vocab"],
    )

    results_df = build_results_table(
        full_metrics=full_metrics,
        random_mean=random_mean,
        random_std=random_std,
        selected_metrics=selected_metrics,
        selected_indices=selected_indices,
        max_train_examples=args.max_train_examples,
        random_subset_size=args.random_subset_size,
    )

    print("\n=== Lexical Diversity Summary ===")
    print(results_df.to_string(index=False))

    if save_csv_path is not None:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_csv_path, index=False, encoding="utf-8")
        print(f"Saved summary table to: {save_csv_path}")


if __name__ == "__main__":
    main()