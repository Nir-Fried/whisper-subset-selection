from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import Audio, DatasetDict


LANGUAGE_CODE = "he"
MAX_TRAIN_EXAMPLES = 20_000

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "common_voice_subset" / LANGUAGE_CODE
HISTOGRAM_OUTPUT_PATH = PROJECT_ROOT / "sentence_and_audio_histograms.png"
RESULTS_OUTPUT_PATH = PROJECT_ROOT / "subset_selection_results.png"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    }
)


def load_train_split(dataset_path: Path, max_examples: int):
    dataset = DatasetDict.load_from_disk(str(dataset_path))
    train_split = dataset["train"].select(range(min(max_examples, len(dataset["train"]))))
    return train_split.cast_column("audio", Audio(decode=True))


def plot_histograms() -> None:
    train_set = load_train_split(DATASET_PATH, MAX_TRAIN_EXAMPLES)
    print("Number of samples in train set:", len(train_set))

    sentence_lengths = []
    audio_lengths = []

    for sample in train_set:
        text = sample["sentence"].strip()
        sentence_lengths.append(len(text.split()) if text else 0)

        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        duration_seconds = len(audio_array) / sample_rate if audio_array is not None and sample_rate else 0.0
        audio_lengths.append(duration_seconds)

    sentence_lengths = np.asarray(sentence_lengths)
    audio_lengths = np.asarray(audio_lengths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(
        sentence_lengths,
        bins=np.arange(sentence_lengths.min(), sentence_lengths.max() + 2),
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
    )
    ax1.set_title("Sentence Lengths (Words)", pad=10)
    ax1.set_xlabel("Number of words")
    ax1.set_ylabel("Number of sentences")
    ax1.set_xlim(left=0)

    ax2.hist(
        audio_lengths,
        bins=50,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
    )
    ax2.set_title("Audio Durations (Seconds)", pad=10)
    ax2.set_xlabel("Duration (s)")
    ax2.set_ylabel("Number of samples")
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(HISTOGRAM_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram plot to: {HISTOGRAM_OUTPUT_PATH}")


def plot_results() -> None:
    subset_sizes = np.array([500, 1000, 2000, 5000, 10000, 15000])

    hebrew = {
        "Euclidean": {
            "WER": np.array([41.64, 40.17, 38.17, 36.24, 33.45, 32.16]),
            "CER": np.array([16.68, 15.86, 15.07, 14.56, 13.43, 12.86]),
        },
        "Cosine": {
            "WER": np.array([40.44, 39.10, 38.26, 35.93, 33.76, 32.50]),
            "CER": np.array([16.95, 15.53, 15.43, 14.26, 13.80, 12.92]),
        },
        "Random": {
            "WER": np.array([42.39, 40.20, 38.87, 36.18, 33.39, 32.21]),
            "CER": np.array([17.44, 16.10, 15.60, 14.57, 13.31, 12.84]),
        },
        "Longest-k": {
            "WER": np.array([40.73, 39.59, 38.37, 35.62, 33.81, 31.96]),
            "CER": np.array([16.72, 16.04, 15.53, 14.32, 14.12, 12.83]),
        },
    }

    danish = {
        "Euclidean": {
            "WER": np.array([49.35, 47.01, 44.23, 40.13, 36.55, 35.06]),
            "CER": np.array([20.01, 19.11, 17.86, 16.05, 14.71, 14.06]),
        },
        "Cosine": {
            "WER": np.array([48.87, 46.68, 43.86, 39.71, 36.87, 34.82]),
            "CER": np.array([19.97, 18.93, 17.74, 16.00, 15.02, 13.89]),
        },
        "Random": {
            "WER": np.array([50.69, 47.35, 44.78, 40.64, 37.36, 35.21]),
            "CER": np.array([21.12, 19.34, 18.20, 16.36, 15.21, 14.21]),
        },
        "Longest-k": {
            "WER": np.array([49.89, 47.65, 44.23, 40.00, 36.73, 34.96]),
            "CER": np.array([20.57, 19.59, 17.98, 16.20, 14.85, 14.08]),
        },
    }

    methods = ["Euclidean", "Cosine", "Random", "Longest-k"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax_hebrew_wer, ax_hebrew_cer, ax_danish_wer, ax_danish_cer = axes.flatten()

    def plot_language_metric(ax, data: dict, metric: str, title: str) -> None:
        for method in methods:
            ax.plot(subset_sizes, data[method][metric], marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("Subset size")
        ax.set_ylabel(metric)

    plot_language_metric(ax_hebrew_wer, hebrew, "WER", "Hebrew - WER")
    plot_language_metric(ax_hebrew_cer, hebrew, "CER", "Hebrew - CER")
    plot_language_metric(ax_danish_wer, danish, "WER", "Danish - WER")
    plot_language_metric(ax_danish_cer, danish, "CER", "Danish - CER")

    handles, labels = ax_hebrew_wer.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)

    tick_values = [500, 2000, 5000, 10000, 15000]
    for ax in axes.flatten():
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_values)
        ax.tick_params(labelbottom=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(RESULTS_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved result plot to: {RESULTS_OUTPUT_PATH}")


if __name__ == "__main__":
    plot_histograms()
    plot_results()