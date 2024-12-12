import argparse
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
from dotenv import load_dotenv

from cybergpt.datasets.websites import WebsiteDataset
from cybergpt.datasets.loaders import WebTrackingLoader
from cybergpt.models.embed.websites import WebsiteTrackingEmbedder
from cybergpt.prompting.clusters import profile_classes


def main(
    data_csv: str,
    output_dir: str,
    sample_size: int,
    model: str,
    max_tokens: int,
    test_size: float,
    from_sequences_file: bool,
):
    load_dotenv()
    if from_sequences_file:
        dataset = pickle.load(open(os.path.join(output_dir, "sequences.pkl"), "rb"))
    else:
        user_data = WebTrackingLoader(data_csv).user_data
        if sample_size:
            sampled_keys = np.random.choice(
                list(user_data.keys()), size=sample_size, replace=False
            )
            user_data = {k: user_data[k] for k in sampled_keys}
        dataset = WebsiteDataset(user_data)
        sequences = dataset.extract_sequences(
            split="pause", pause_threshold=3600, combine_sequential=True
        )

        string_sequences = [
            WebsiteTrackingEmbedder.convert_sequence_to_string(s)
            for s in tqdm(sequences, desc="Processing sequences")
        ]
        labels = [s["user_id"] for s in sequences]

        class_sequences = {
            str(label): [seq for seq, l in zip(string_sequences, labels) if l == label]
            for label in set(labels)
        }

        # Split the class sequences into train and test
        class_sizes = {k: len(v) for k, v in class_sequences.items()}
        test_indices = {
            k: np.random.choice(range(n), size=int(n * test_size), replace=False)
            for k, n in class_sizes.items()
        }
        train_sequences = {
            k: [v[i] for i in range(len(v)) if i not in test_indices[k]]
            for k, v in class_sequences.items()
        }
        test_sequences = {
            k: [v[i] for i in test_indices[k]] for k, v in class_sequences.items()
        }

        # Remove empty classes
        train_sequences = {k: v for k, v in train_sequences.items() if len(v) > 0}
        test_sequences = {
            k: v for k, v in test_sequences.items() if k in train_sequences
        }

        os.makedirs(output_dir, exist_ok=True)
        dataset = {
            "train_sequences": train_sequences,
            "test_sequences": test_sequences,
        }
        pickle.dump(dataset, open(os.path.join(output_dir, "sequences.pkl"), "wb"))

    train_sequences = dataset["train_sequences"]
    test_sequences = dataset["test_sequences"]

    print("Profiling classes...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = open("cybergpt/prompting/ic_cluster_system_prompt.txt", "r").read()
    responses = profile_classes(
        client,
        system_prompt,
        train_sequences,
        model_name=model,
        max_tokens=int(max_tokens),
    )

    pickle.dump(
        responses,
        open(os.path.join(output_dir, "ic_output.pkl"), "wb"),
    )


if __name__ == "__main__":
    """
    Example usage:
    python -m cybergpt.scripts.cluster_user_profiles \
        --data_csv data/web_tracking/web_routineness_release/raw/browsing.csv \
        --output_dir data/prompting \
        --sample_size 10 \
        --model gpt-4o-mini
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
    args.add_argument("--output_dir", type=str, default="data/prompting")
    args.add_argument("--sample_size", type=int, default=None)
    args.add_argument("--model", type=str, default="gpt-4o-mini")
    args.add_argument("--max_tokens", type=int, default=100000)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--from_sequences_file", type=bool, default=False)
    args = args.parse_args()
    main(
        args.data_csv,
        args.output_dir,
        args.sample_size,
        args.model,
        args.max_tokens,
        args.test_size,
        args.from_sequences_file,
    )
