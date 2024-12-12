import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, List, Union
from tqdm.auto import tqdm

from cybergpt.datasets.websites import WebsiteDataset
from cybergpt.datasets.loaders import WebTrackingLoader
from cybergpt.models.embed.websites import WebsiteTrackingEmbedder


def build_contrastive_dataset(
    values: List[Any],
    labels: List[str],
    test_size: Union[float, int] = 0.1,
    random_state: int = 42,
) -> List[dict]:
    """Build a contrastive dataset for classification tasks.

    Args:
        values (List[Any]): The values to build the contrastive dataset from.
        labels (List[str]): The labels for the values.
        test_size (Union[float, int], optional): The size of the test set. Defaults to 0.1.
        random_state (int, optional): The random state to use for the random sampling. Defaults to 42.

    Raises:
        ValueError: If the test_size is not a float or int.

    Returns:
        List[dict]: The contrastive dataset consisting for each class of values corresponding to the class,
            a test set also corresponding to the class, and a randomly sampled set of values of the same size
            as test set from other classes.
    """
    random.seed(random_state)

    classes = list(set(labels))
    dataset = []
    for c in tqdm(classes, desc="Building contrastive dataset"):
        class_values = [v for v, l in zip(values, labels) if l == c]
        other_values = [v for v, l in zip(values, labels) if l != c]
        if isinstance(test_size, int):
            n_test = test_size
        elif isinstance(test_size, float):
            n_test = int(len(class_values) * test_size)
        else:
            raise ValueError(f"Invalid test_size: {test_size}")
        test_idx = random.sample(range(len(class_values)), n_test)
        test_set = [class_values[i] for i in test_idx]
        remaining_set = [v for i, v in enumerate(class_values) if i not in test_idx]
        contrastive_set = random.sample(other_values, n_test)
        dataset.append(
            {
                "class": c,
                "values": remaining_set,
                "test_values": test_set,
                "contrastive_values": contrastive_set,
            }
        )
    return dataset


if __name__ == "__main__":
    """
    Usage:

    python -m cybergpt.eval.contrastive \
        --data_csv data/web_tracking/web_routineness_release/raw/browsing.csv \
        --output_dir data/contrastive
    """
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.append(project_root)

    RANDOM_SEED = 42

    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_csv",
        type=str,
        default="data/web_tracking/web_routineness_release/raw/browsing.csv",
    )
    args.add_argument("--output_dir", type=str, default="data/contrastive")
    args = args.parse_args()

    user_data = WebTrackingLoader(args.data_csv).user_data
    dataset = WebsiteDataset(user_data)

    sequences = dataset.extract_sequences(
        split="pause", pause_threshold=3600, combine_sequential=True
    )
    sequence_strings = [
        WebsiteTrackingEmbedder.convert_sequence_to_string(s)
        for s in tqdm(sequences, desc="Converting sequences to strings")
    ]
    labels = [s["user_id"] for s in sequences]

    contrastive_dataset = build_contrastive_dataset(
        sequence_strings, labels, random_state=RANDOM_SEED
    )
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(
        contrastive_dataset,
        open(os.path.join(args.output_dir, "classification_dataset.pkl"), "wb"),
    )
