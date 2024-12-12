from typing import List, Dict, Set, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cybergpt.datasets.base import BaseNodeDataset


class SequenceDataset(BaseNodeDataset):
    def __init__(self, sequences: List[List[str]]):
        """Initialize sequence dataset with list of sequences.

        Args:
            sequences: List of sequences, where each sequence is a list of strings
        """
        super().__init__()
        self.sequences = sequences
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Build vocabulary from sequences."""
        for sequence in self.sequences:
            for vertex, next_vertex in zip(sequence, sequence[1:]):
                self.vocab.update([vertex, next_vertex])

    def _calculate_transition_metrics(
        self, sequence: List[str]
    ) -> tuple[Set, dict, float, float]:
        """Calculate transition-based metrics for a single sequence."""
        transition_counts = defaultdict(int)

        for vertex, next_vertex in zip(sequence, sequence[1:]):
            transition_counts[f"{vertex}->{next_vertex}"] += 1

        repetition_rate = 0.0
        if len(sequence) > 1:
            repetition_rate = 1 - ((len(set(sequence)) - 1) / (len(sequence) - 1))

        transition_entropy = 0.0
        if transition_counts:
            transition_probs = np.array(list(transition_counts.values())) / sum(
                transition_counts.values()
            )
            transition_entropy = entropy(transition_probs)

        return (
            transition_counts,
            repetition_rate,
            transition_entropy,
        )

    def calculate_sequence_metrics(self) -> Dict[str, Any]:
        """Calculate basic sequence metrics including lengths, character usage, and transition patterns."""
        metrics = {
            "total_sequences": len(self.sequences),
            "sequence_lengths": [],
            "unique_chars_per_sequence": [],
            "char_repetition_rates": [],
            "transition_entropy": [],
            "char_coverage": [],
        }

        for sequence in tqdm(self.sequences, desc="Calculating metrics"):
            _, repetition_rate, trans_entropy = self._calculate_transition_metrics(
                sequence
            )

            metrics["sequence_lengths"].append(len(sequence))
            metrics["unique_chars_per_sequence"].append(len(set(sequence)))
            metrics["char_coverage"].append(len(set(sequence)) / self.get_vocab_size())
            metrics["char_repetition_rates"].append(repetition_rate)
            metrics["transition_entropy"].append(trans_entropy)

        return metrics

    def calculate_sequence_similarity_metrics(self, sample_size=1000) -> dict:
        """Calculate sequence similarity metrics using TF-IDF and cosine similarity."""
        sequence_sample = self.sequences[:sample_size]
        sequence_strings = []
        for sequence in sequence_sample:
            transition_string = " ".join(
                [
                    f"{vertex}->{next_vertex}"
                    for vertex, next_vertex in zip(sequence, sequence[1:])
                ]
            )
            sequence_strings.append(transition_string)

        if not sequence_strings:
            return {
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "similarity_distribution": np.array([]),
            }

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sequence_strings)
        similarities = cosine_similarity(tfidf_matrix)

        avg_similarity = np.mean(
            similarities[np.triu_indices(similarities.shape[0], k=1)]
        )
        max_similarity = np.max(
            similarities[np.triu_indices(similarities.shape[0], k=1)]
        )

        return {
            "average_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "similarity_distribution": similarities[
                np.triu_indices(similarities.shape[0], k=1)
            ],
        }

    def calculate_transition_stats(self) -> dict:
        """Calculate transition statistics and starting character frequencies."""
        unique_transitions = set()
        starting_characters = []

        for sequence in self.sequences:
            if sequence:
                starting_characters.append(sequence[0])
                for vertex, next_vertex in zip(sequence, sequence[1:]):
                    unique_transitions.add((vertex, next_vertex))

        start_char_counts = Counter(starting_characters)
        top_starting_chars = dict(start_char_counts.most_common(5))

        return {
            "unique_transitions": len(unique_transitions),
            "top_starting_chars": top_starting_chars,
        }

    def calculate_metrics(self) -> dict:
        return {
            **self.calculate_sequence_metrics(),
            **self.calculate_sequence_similarity_metrics(),
            **self.calculate_transition_stats(),
        }

    def summary_report(self, metrics: dict):
        """Generate a summary report for the sequence dataset."""
        print(f"Sequence Dataset Summary Report")
        print("--------------------------------")
        print(f"Total sequences: {metrics['total_sequences']}")
        print(f"Average sequence length: {np.mean(metrics['sequence_lengths'])}")
        print(
            f"Average unique characters per sequence: {np.mean(metrics['unique_chars_per_sequence'])}"
        )
        print(f"Average character coverage: {np.mean(metrics['char_coverage'])}")
        print(
            f"Average character repetition rate: {np.mean(metrics['char_repetition_rates'])}"
        )
        print(f"Average transition entropy: {np.mean(metrics['transition_entropy'])}")
        print(
            f"Average TF-IDF cosine similarity: {np.mean(metrics['average_similarity'])}"
        )
        print(f"Max similarity: {np.max(metrics['similarity_distribution'])}")
        print("")
        print(f"Unique transitions: {metrics['unique_transitions']}")
        print(f"Top 5 starting characters: {metrics['top_starting_chars']}")
        print("")


def plot_metrics(metrics):
    """Plot sequence metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    sns.histplot(metrics["sequence_lengths"], bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Sequence Lengths")
    axes[0, 0].set_xlabel("Number of Edges")

    sns.histplot(metrics["unique_chars_per_sequence"], bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Unique Characters per Sequence")
    axes[0, 1].set_xlabel("Number of Unique Characters")

    sns.histplot(metrics["char_repetition_rates"], bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of Character Repetition Rates")
    axes[1, 0].set_xlabel("Repetition Rate (higher = more repetitive)")

    sns.histplot(metrics["transition_entropy"], bins=50, ax=axes[1, 1])
    axes[1, 1].set_title("Transition Entropy Distribution")
    axes[1, 1].set_xlabel("Entropy")

    plt.tight_layout()
    return plt.gcf()
