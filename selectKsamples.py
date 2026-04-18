import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative embedding indices with greedy k-center."
    )
    parser.add_argument("--language", type=str, default="danish")
    parser.add_argument(
        "--embedding-path",
        type=str,
        default=None,
        help="Path to .npy embedding file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="kcenter",
        help="Used only when --embedding-path is not given.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000, 5000, 10000, 15000],
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output index files.",
    )
    return parser.parse_args()


def resolve_paths(args):
    project_root = Path(__file__).resolve().parent

    if args.embedding_path is not None:
        embedding_path = Path(args.embedding_path)
    else:
        embedding_path = Path(args.input_dir) / f"whisper_train_embeddings_{args.language}.npy"

    if not embedding_path.is_absolute():
        embedding_path = project_root / embedding_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return embedding_path, output_dir


def load_embeddings(file_path: Path) -> np.ndarray:
    embeddings = np.load(file_path)
    print("Loaded embeddings with shape:", embeddings.shape)
    return embeddings


def k_center_greedy_euclidean(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = x.shape[0]
    selected_indices = [rng.integers(0, n_samples)]

    for i in range(1, k):
        if i % 100 == 0 or i == k - 1:
            print(f"Iteration {i}/{k}")

        distances = pairwise_distances(x, x[selected_indices], metric="euclidean")
        min_distances = distances.min(axis=1)
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)

    return np.array(selected_indices, dtype=np.int64)


def k_center_greedy_cosine(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = x.shape[0]
    selected_indices = [rng.integers(0, n_samples)]

    max_similarities = cosine_similarity(x, x[selected_indices])[:, 0]

    for i in range(1, k):
        if i % 100 == 0 or i == k - 1:
            print(f"Iteration {i}/{k}")

        min_cosine_distances = 1.0 - max_similarities
        next_index = np.argmax(min_cosine_distances)
        selected_indices.append(next_index)

        new_similarities = cosine_similarity(x, x[[next_index]])[:, 0]
        max_similarities = np.maximum(max_similarities, new_similarities)

    return np.array(selected_indices, dtype=np.int64)


def maybe_normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    print(
        "Embedding norm stats:",
        f"min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}",
    )

    if np.allclose(norms, 1.0, atol=1e-3):
        print("Embeddings already normalized.")
        return x

    print("Normalizing embeddings to unit norm...")
    return normalize(x, norm="l2")


def save_selected_indices(
    indices: np.ndarray,
    k: int,
    output_dir: Path,
    method: str,
    language: str,
) -> None:
    prefix = "selected_indices_cosine" if method == "cosine" else "selected_indices"
    output_path = output_dir / f"{prefix}_k{k}_{language}.txt"
    np.savetxt(output_path, indices, fmt="%d")
    print(f"Saved {k} selected indices to {output_path}")


def main():
    args = parse_args()
    embedding_path, output_dir = resolve_paths(args)
    rng = np.random.default_rng(args.random_seed)

    print("Embedding path:", embedding_path)
    print("Output dir:", output_dir)

    embeddings = load_embeddings(embedding_path)

    if args.method == "cosine":
        embeddings = maybe_normalize_embeddings(embeddings)
        selection_fn = k_center_greedy_cosine
        print("Using cosine-based greedy k-center selection.")
    else:
        selection_fn = k_center_greedy_euclidean
        print("Using Euclidean greedy k-center selection.")

    for k in args.k_values:
        print(f"Selecting {k} samples...")
        selected_indices = selection_fn(embeddings, k, rng)
        save_selected_indices(selected_indices, k, output_dir, args.method, args.language)

    print("Done.")


if __name__ == "__main__":
    main()