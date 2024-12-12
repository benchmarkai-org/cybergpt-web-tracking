import argparse
from typing import List, Tuple
import numpy as np
from typing import List, Optional
import os
import pickle
import tiktoken
from dotenv import load_dotenv
from tqdm.auto import tqdm

from cybergpt.datasets.websites import WebsiteDataset
from cybergpt.datasets.loaders import WebTrackingLoader
from cybergpt.models.embed.models import MODELS


class WebsiteTrackingEmbedder:
    def __init__(
        self,
        output_dir: str = "data/embeddings",
    ):
        self.output_dir = output_dir

    @staticmethod
    def convert_sequence_to_string(sequence_dict: dict, max_tokens: int = 8192) -> str:
        start_time, df = sequence_dict["start_time"], sequence_dict["df"]

        prefix = f"{start_time.strftime('%A')} {start_time.strftime('%H:%M')}, Visits: "

        site_visits = [
            f"{row['domain']} ({row['active_seconds']}s)" for _, row in df.iterrows()
        ]

        # Start with the full string and remove middle tokens if too long
        full_string = prefix + " -> ".join(site_visits)
        encoding = tiktoken.get_encoding("cl100k_base")

        if len(encoding.encode(full_string)) <= max_tokens:
            return full_string

        start_visits = site_visits[: len(site_visits) // 2]
        end_visits = site_visits[len(site_visits) // 2 + 1 :]
        while len(encoding.encode(full_string)) > max_tokens:
            if len(start_visits) < len(end_visits):
                end_visits.pop(0)
            else:
                start_visits.pop()
            full_string = prefix + " -> ".join(start_visits + ["..."] + end_visits)

        return full_string

    def preprocess_dataset(
        self, dataset: Optional[WebsiteDataset] = None, from_pickle: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Preprocess the dataset to extract sequences and save them to a pickle file.
        """
        if dataset is None and not from_pickle:
            raise ValueError("Either dataset or from_pickle=True must be provided")

        os.makedirs(self.output_dir, exist_ok=True)
        if from_pickle:
            sequences = pickle.load(
                open(os.path.join(self.output_dir, "sequences.pkl"), "rb")
            )
        else:
            sequences = dataset.extract_sequences(
                split="pause", pause_threshold=3600, combine_sequential=True
            )
            pickle.dump(
                sequences, open(os.path.join(self.output_dir, "sequences.pkl"), "wb")
            )

        print("Converting sequences to strings...")
        string_sequences = [
            self.convert_sequence_to_string(s)
            for s in tqdm(sequences, desc="Processing sequences")
        ]
        labels = [s["user_id"] for s in sequences]

        print("Saving preprocessed dataset...")
        pickle.dump(
            {"string_sequences": string_sequences, "labels": labels},
            open(os.path.join(self.output_dir, "preprocessed_dataset.pkl"), "wb"),
        )
        return string_sequences, labels

    def embed_sequences(self, model_name: str) -> np.ndarray:
        """
        Embed sequences using TF-IDF and return the embeddings.
        """
        if not os.path.exists(
            os.path.join(self.output_dir, "preprocessed_dataset.pkl")
        ):
            raise ValueError(
                "Preprocessed dataset does not exist. Please preprocess the dataset first."
            )

        data = pickle.load(
            open(os.path.join(self.output_dir, "preprocessed_dataset.pkl"), "rb")
        )
        string_sequences, _ = data["string_sequences"], data["labels"]

        if model_name not in MODELS:
            raise ValueError(f"Invalid model: {model_name}")
        model = MODELS[model_name]

        embeddings = model.get_embeddings(string_sequences)
        pickle.dump(
            embeddings,
            open(os.path.join(self.output_dir, f"embeddings_{model_name}.pkl"), "wb"),
        )
        return embeddings


if __name__ == "__main__":
    """
    Usage:

    python -m cybergpt.models.embed.websites \
        --data_csv data/web_tracking/web_routineness_release/raw/browsing.csv \
        --output_dir data/embeddings \
        --models openai openai-large minilm e5 tfidf \
        --sample_size 200
    """
    load_dotenv()

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
    args.add_argument("--models", type=str, nargs="+", default=["openai"])
    args.add_argument("--sample_size", type=int, default=None)
    args = args.parse_args()

    user_data = WebTrackingLoader(args.data_csv).user_data
    if args.sample_size:
        sampled_keys = np.random.choice(
            list(user_data.keys()), size=args.sample_size, replace=False
        )
        user_data = {k: user_data[k] for k in sampled_keys}
    dataset = WebsiteDataset(user_data)

    embedder = WebsiteTrackingEmbedder(output_dir=args.output_dir)
    embedder.preprocess_dataset(dataset=dataset)

    for model_name in args.models:
        embedder.embed_sequences(model_name=model_name)
