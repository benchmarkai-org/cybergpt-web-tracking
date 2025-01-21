import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import pickle
import argparse
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
from typing import List, Dict

from cybergpt.datasets.loaders import WebTrackingLoader
from cybergpt.datasets.websites import WebsiteDataset

load_dotenv()

LONG_PATTERN = "Monday 15:10, Visits: youtube.com (63s) -> google.com ({t}s)"
MEDIUM_PATTERN = "Web Visit: google.com ({t}s)"
SHORT_PATTERN = "google.com ({t}s)"
VERY_SHORT_PATTERN = "{t}"

MODELS = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "e5": SentenceTransformer("intfloat/e5-base-v2"),
    "openai-small": "text-embedding-3-small",
    "openai-large": "text-embedding-3-large",
}


def generate_log_entries(pattern: str, time_intervals: List[int]) -> List[str]:
    return [pattern.format(t=t) for t in time_intervals]


def get_embedding(text: str, model_name: str) -> np.ndarray:
    if model_name in ["openai-small", "openai-large"]:
        response = openai.embeddings.create(model=MODELS[model_name], input=text)
        embedding = response.data[0].embedding
        return np.array(embedding)
    else:
        embedding = MODELS[model_name].encode(text)
        return embedding


def analyse_embeddings(
    models: List[str], pattern: str, time_intervals: List[int]
) -> Dict:
    """Analyse embeddings for all models and time intervals."""
    results = {}
    log_entries = generate_log_entries(pattern, time_intervals)

    for model_name in models:
        print(f"\nProcessing {model_name}...")
        embeddings = []

        for text in tqdm(log_entries):
            embedding = get_embedding(text, model_name)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        results[model_name] = embeddings

    return results


def get_time_distributions(data_csv: str) -> Dict[str, list]:
    user_data = WebTrackingLoader(data_csv).user_data
    dataset = WebsiteDataset(user_data)
    sequences = dataset.extract_sequences(
        split="pause", pause_threshold=3600, combine_sequential=True
    )
    day_dist = [s["start_time"].day_of_week for s in sequences]
    length_dist = [len(s["df"]) for s in sequences]
    total_duration_dist = [s["df"]["active_seconds"].sum() for s in sequences]
    durations = []
    for s in sequences:
        durations += s["df"]["active_seconds"].to_list()
    return {
        "day": day_dist,
        "length": length_dist,
        "total_duration": total_duration_dist,
        "durations": durations,
    }


def get_time_intervals(durations: List[int], dq: float, d: int) -> List[int]:
    # Calculate quantiles from duration distribution and remove duplicates
    dist_time_intervals = sorted(
        list(
            set(
                np.quantile(
                    durations,
                    q=np.arange(0, 1.0, dq),
                )
                .astype(int)
                .tolist()
            )
        )
    )
    # Add some fixed time intervals for diversity
    base_time_intervals = []
    n = 10
    N = 100
    while n < max(dist_time_intervals):
        for i in range(n, N, d):
            base_time_intervals.append(i)
        n = N
        N = 10 * N
        d = 10 * d

    return sorted(list(set(base_time_intervals + dist_time_intervals)))


def main():
    """
    Example usage:
    python -m cybergpt.scripts.embed_durations_eval \
        --data_csv data/web_tracking/web_routineness_release/raw/browsing.csv \
        --output_dir data/embeddings \
        --dq 0.005 \
        --d 5
    """
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.append(project_root)

    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_csv",
        type=str,
        default="data/web_tracking/web_routineness_release/raw/browsing.csv",
    )
    args.add_argument("--output_dir", type=str, default="data/embeddings")
    args.add_argument("--dq", type=float, default=0.005)
    args.add_argument("--d", type=int, default=5)
    args = args.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    dq = float(args.dq)
    d = int(args.d)

    print("Computing data time distributions...")
    time_distributions = get_time_distributions(args.data_csv)

    TIME_INTERVALS = get_time_intervals(time_distributions["durations"], dq, d)
    print(f"Total time intervals: {len(TIME_INTERVALS)}")

    print("Analyzing embeddings...")
    print("Long pattern...")
    long_results = analyse_embeddings(MODELS, LONG_PATTERN, TIME_INTERVALS)
    print("Medium pattern...")
    medium_results = analyse_embeddings(MODELS, MEDIUM_PATTERN, TIME_INTERVALS)
    print("Short pattern...")
    short_results = analyse_embeddings(MODELS, SHORT_PATTERN, TIME_INTERVALS)
    print("Very short pattern...")
    very_short_results = analyse_embeddings(MODELS, VERY_SHORT_PATTERN, TIME_INTERVALS)

    results = {
        "long": long_results,
        "medium": medium_results,
        "short": short_results,
        "very_short": very_short_results,
    }
    output = {
        "time_distributions": time_distributions,
        "results": results,
    }
    with open(os.path.join(args.output_dir, "duration_embeddings.pkl"), "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
