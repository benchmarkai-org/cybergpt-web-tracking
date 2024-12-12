import argparse
import pickle
import random
import tiktoken
import json
import dotenv
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from tqdm.auto import tqdm


def _build_system_prompt(
    base_system_prompt: str, values: List[str], max_system_tokens: int
) -> Tuple[str, List[str]]:
    """Constructs a system prompt by appending values until the token limit is reached."""
    full_system_prompt = f"{base_system_prompt}\n<HISTORY>\n"
    token_count = len(tiktoken.get_encoding("cl100k_base").encode(full_system_prompt))

    random.shuffle(values)
    train_values = []
    for i, item in enumerate(values):
        string_value = f"{item}\n"
        new_tokens = len(tiktoken.get_encoding("cl100k_base").encode(string_value))
        if token_count + new_tokens > max_system_tokens:
            break
        train_values.append(item)
        full_system_prompt += string_value
        token_count += new_tokens
    full_system_prompt += "</HISTORY>"
    return full_system_prompt, train_values


def _build_query_prompt(
    ho_values: List[str], contrastive_values: List[str]
) -> Tuple[str, List[bool], List[str]]:
    """Builds a query prompt by mixing holdout and contrastive values."""
    test_values = ho_values + contrastive_values
    belongs_to_class = [True] * len(ho_values) + [False] * len(contrastive_values)
    test_tuples = random.sample(
        list(zip(test_values, belongs_to_class)),
        len(test_values),
    )
    test_values = [x[0] for x in test_tuples]
    belongs_to_class = [x[1] for x in test_tuples]

    query_prompt = ""
    for i, item in enumerate(test_values):
        query_prompt += f"{i+1}. {item}\n"

    return query_prompt, belongs_to_class, test_values


def _parse_response(response: Any) -> Tuple[Dict[str, Any], str]:
    """Parses the response from the OpenAI API."""
    response_text = response.choices[0].message.content
    try:
        response_json = json.loads(
            response_text.replace("```json", "").replace("```", "")
        )
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {response_text}")
        response_json = None
    return response_json, response_text


def classify_items(
    client: OpenAI,
    system_prompt: str,
    values: List[str],
    ho_values: List[str],
    contrastive_values: List[str],
    max_system_tokens: int = 80000,
) -> Dict[str, Any]:
    """
    Classifies items using OpenAI API by constructing a system prompt using the values given for in-context
    learning and constructing a query based on the held out and contrastive values.
    """
    full_system_prompt, train_values = _build_system_prompt(
        system_prompt, values, max_system_tokens
    )

    query_prompt, belongs_to_class, test_values = _build_query_prompt(
        ho_values, contrastive_values
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": query_prompt},
        ],
    )
    response_json, response_text = _parse_response(response)

    return {
        "response_text": response_text,
        "belongs_to_class": belongs_to_class,
        "response_json": response_json,
        "train_values": train_values,
        "test_values": test_values,
    }


def compute_metrics(results: List[Dict[str, Any]]):
    """Computes metrics for the prompting classification results."""
    confidence_levels = ["low", "medium", "high"]
    labels = confidence_levels + ["overall"]

    tps = {c: [] for c in labels}
    tns = {c: [] for c in labels}
    fps = {c: [] for c in labels}
    fns = {c: [] for c in labels}
    for r in results:
        y = r["belongs_to_class"]
        y_hat = [t["match"] for t in r["response_json"]]
        y_conf = [t["confidence"] for t in r["response_json"]]
        tps["overall"].append(
            sum([y_i == y_hat_i for y_i, y_hat_i in zip(y, y_hat) if y_i == True])
        )
        tns["overall"].append(
            sum([y_i == y_hat_i for y_i, y_hat_i in zip(y, y_hat) if y_i == False])
        )
        fps["overall"].append(
            sum([y_i != y_hat_i for y_i, y_hat_i in zip(y, y_hat) if y_i == False])
        )
        fns["overall"].append(
            sum([y_i != y_hat_i for y_i, y_hat_i in zip(y, y_hat) if y_i == True])
        )
        for c in confidence_levels:
            if not any(y_conf_i == c for y_conf_i in y_conf):
                continue
            tps[c].append(
                sum(
                    [
                        y_i == y_hat_i
                        for y_i, y_hat_i, y_conf_i in zip(y, y_hat, y_conf)
                        if y_conf_i == c and y_i == True
                    ]
                )
            )
            tns[c].append(
                sum(
                    [
                        y_i == y_hat_i
                        for y_i, y_hat_i, y_conf_i in zip(y, y_hat, y_conf)
                        if y_conf_i == c and y_i == False
                    ]
                )
            )
            fps[c].append(
                sum(
                    [
                        y_i != y_hat_i
                        for y_i, y_hat_i, y_conf_i in zip(y, y_hat, y_conf)
                        if y_conf_i == c and y_i == False
                    ]
                )
            )
            fns[c].append(
                sum(
                    [
                        y_i != y_hat_i
                        for y_i, y_hat_i, y_conf_i in zip(y, y_hat, y_conf)
                        if y_conf_i == c and y_i == True
                    ]
                )
            )

    accuracies = {
        c: (np.array(tps[c]) + np.array(tns[c]))
        / (np.array(tps[c]) + np.array(tns[c]) + np.array(fps[c]) + np.array(fns[c]))
        for c in labels
    }

    tp = {c: np.sum(tps[c]) for c in labels}
    fp = {c: np.sum(fps[c]) for c in labels}
    tn = {c: np.sum(tns[c]) for c in labels}
    fn = {c: np.sum(fns[c]) for c in labels}
    precision = {c: tp[c] / (tp[c] + fp[c]) for c in labels}
    recall = {c: tp[c] / (tp[c] + fn[c]) for c in labels}
    f1 = {c: 2 * precision[c] * recall[c] / (precision[c] + recall[c]) for c in labels}
    accuracy = {c: (tp[c] + tn[c]) / (tp[c] + tn[c] + fp[c] + fn[c]) for c in labels}

    return {
        "aggregated": {
            c: {
                "precision": precision[c],
                "recall": recall[c],
                "f1": f1[c],
                "accuracy": accuracy[c],
                "tp": tp[c],
                "tn": tn[c],
                "fp": fp[c],
                "fn": fn[c],
            }
            for c in labels
        },
        "individual": {
            c: {
                "accuracy": accuracies[c],
            }
            for c in labels
        },
    }


def plot_metrics(metrics: Dict[str, Any]):
    """Plots the metrics using three vertically stacked subplots."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Overall metrics
    metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    values = [metrics["aggregated"]["overall"][m] for m in metrics_to_plot]
    bars = ax1.bar(x, values, width)

    for i, v in enumerate(values):
        if metrics_to_plot[i] == "precision":
            numerator = metrics["aggregated"]["overall"]["tp"]
            denominator = (
                metrics["aggregated"]["overall"]["tp"]
                + metrics["aggregated"]["overall"]["fp"]
            )
        elif metrics_to_plot[i] == "recall":
            numerator = metrics["aggregated"]["overall"]["tp"]
            denominator = (
                metrics["aggregated"]["overall"]["tp"]
                + metrics["aggregated"]["overall"]["fn"]
            )
        elif metrics_to_plot[i] == "f1":
            ax1.text(x[i], v, f"{v:.3f}", ha="center", va="bottom")
            continue
        else:  # accuracy
            numerator = (
                metrics["aggregated"]["overall"]["tp"]
                + metrics["aggregated"]["overall"]["tn"]
            )
            denominator = sum(
                [metrics["aggregated"]["overall"][k] for k in ["tp", "tn", "fp", "fn"]]
            )

        ax1.text(
            x[i], v, f"{v:.3f}\n{numerator}/{denominator}", ha="center", va="bottom"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.axhline(y=0.5, color="gray", linestyle=":", label="Baseline")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Overall Metrics")
    ax1.set_ylabel("Score")

    # Plot 2: Metrics by confidence level
    metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
    labels = ["overall", "low", "medium", "high"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, label in enumerate(labels):
        values = [metrics["aggregated"][label][m] for m in metrics_to_plot]
        offset = width * (i - len(labels) / 2 + 0.5)
        total = (
            metrics["aggregated"][label]["tp"]
            + metrics["aggregated"][label]["tn"]
            + metrics["aggregated"][label]["fp"]
            + metrics["aggregated"][label]["fn"]
        )
        label_with_count = f"{label} (n={total})"
        bars = ax2.bar(x + offset, values, width, label=label_with_count)

        for j, v in enumerate(values):
            ax2.text(
                x[j] + offset,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_to_plot)
    ax2.axhline(y=0.5, color="gray", linestyle=":", label="Baseline")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Metrics by Confidence Level")
    ax2.set_ylabel("Score")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: Accuracy with error bars by confidence level
    means = []
    stds = []

    for label in labels:
        accuracies = metrics["individual"][label]["accuracy"]
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))

    bars = ax3.bar(labels, means, yerr=stds, capsize=5)
    ax3.set_ylim(0, 1.1)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax3.text(i, mean, f"{mean:.3f}", ha="center", va="bottom")

    ax3.set_title("Accuracy by Confidence Level")
    ax3.set_ylabel("Accuracy")

    plt.tight_layout()
    return fig


def _parse_visits(visits_str):
    """Parse the visits string into a list of (site, duration) tuples"""
    visits_str = visits_str.replace("Visits: ", "")
    visits = visits_str.split(" -> ")
    parsed = []
    for visit in visits:
        site, duration = re.match(r"(.+?)\s*\((\d+)s\)", visit).groups()
        parsed.append((site, int(duration)))
    return parsed


def _format_duration(seconds):
    """Format duration in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}m {seconds}s"


def create_html_table(data):
    """Create an HTML table with styling from the browser history data"""
    html = """
    <style>
        .browser-history {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        .browser-history th {
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        .browser-history td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }
        .browser-history tr:hover {
            background-color: #f5f5f5;
        }
        .confidence-low { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-high { color: #dc3545; }
        .visits-cell {
            max-width: 500px;
            overflow-x: auto;
        }
        .visit-span {
            display: inline-block;
            margin: 2px;
            padding: 2px 4px;
            background-color: #e9ecef;
            border-radius: 3px;
            font-size: 0.9em;
        }
    </style>
    <table class="browser-history">
        <tr>
            <th>Time</th>
            <th>Web Visits</th>
            <th>Confidence</th>
            <th>Reasoning</th>
        </tr>
    """

    for entry in data:
        time_visits_str, confidence, analysis = entry
        day, time, visits_str = time_visits_str.split(" ", maxsplit=2)
        time = time.rstrip(",")

        visits = _parse_visits(visits_str)
        visits_html = " → ".join(
            [
                f'<span class="visit-span">{site} ({_format_duration(duration)})</span>'
                for site, duration in visits
            ]
        )

        confidence_class = f"confidence-{confidence}"

        html += f"""
        <tr>
            <td>{day} {time}</td>
            <td class="visits-cell">{visits_html}</td>
            <td class="{confidence_class}">{confidence.upper()}</td>
            <td>{analysis}</td>
        </tr>
        """

    html += "</table>"
    return html


if __name__ == "__main__":
    """
    Example usage:
    python -m cybergpt.prompting.classification \
        --dataset_path data/contrastive/classification_dataset.pkl \
        --system-prompt cybergpt/prompting/class_system_prompt.txt \
        --output-path data/contrastive/classification_results.pkl \
        --sample-size 200 \
        --max-system-tokens 90000 \
        --min-test-size 5
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_path",
        type=str,
        default="data/contrastive/classification_dataset.pkl",
    )
    args.add_argument(
        "--system-prompt",
        type=str,
        default="cybergpt/prompting/class_system_prompt.txt",
    )
    args.add_argument(
        "--output-path", type=str, default="data/contrastive/classification_results.pkl"
    )
    args.add_argument("--sample-size", type=int, default=None)
    args.add_argument("--max-system-tokens", type=int, default=80000)
    args.add_argument("--min-test-size", type=int, default=5)
    args = args.parse_args()

    RANDOM_SEED = 42
    SYSTEM_PROMPT = open(args.system_prompt, "r").read()
    max_system_tokens = int(args.max_system_tokens)

    dataset = pickle.load(open(args.dataset_path, "rb"))

    random.seed(RANDOM_SEED)

    if args.min_test_size:
        dataset = [
            d for d in dataset if len(d["test_values"]) >= int(args.min_test_size)
        ]

    if args.sample_size is not None:
        sample_size = int(args.sample_size)
        dataset = random.sample(dataset, sample_size)
        assert len(dataset) == sample_size

    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    class_results = []
    for d in tqdm(dataset, desc="Classifying dataset"):
        results = classify_items(
            client,
            SYSTEM_PROMPT,
            d["values"],
            d["test_values"],
            d["contrastive_values"],
            max_system_tokens,
        )
        results["class"] = d["class"]
        class_results.append(results)

    pickle.dump(class_results, open(args.output_path, "wb"))
