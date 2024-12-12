import json
import tiktoken
import random
from typing import Dict, List, Any, Tuple
from openai import OpenAI


def _subsample_sequences(
    clustered_sequences: Dict[str, List[str]],
    encoding: tiktoken.Encoding,
    max_tokens: int = 100000,
    random_seed: int = 42,
):
    """Subsample sequences from the clusters."""
    random.seed(random_seed)

    total_sequences = sum(len(seqs) for seqs in clustered_sequences.values())
    cluster_frequencies = {
        cluster: len(seqs) / total_sequences
        for cluster, seqs in clustered_sequences.items()
    }

    available_sequences = {k: list(v) for k, v in clustered_sequences.items()}
    sampled_sequences = {k: [] for k in clustered_sequences.keys()}

    # Sample sequences until we hit token limit
    token_count = 0
    while token_count < max_tokens and any(available_sequences.values()):
        clusters = [c for c in available_sequences.keys() if available_sequences[c]]
        if not clusters:
            break

        weights = [cluster_frequencies[c] for c in clusters]
        cluster = random.choices(clusters, weights=weights, k=1)[0]

        sequence = random.choice(available_sequences[cluster])
        sequence_tokens = len(encoding.encode(sequence))

        if token_count + sequence_tokens <= max_tokens:
            sampled_sequences[cluster].append(sequence)
            available_sequences[cluster].remove(sequence)
            token_count += sequence_tokens
        else:
            break

    # Remove empty clusters
    sampled_sequences = {k: v for k, v in sampled_sequences.items() if v}

    return sampled_sequences


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


def interpret_clusters(
    client: OpenAI,
    system_prompt: str,
    clustered_sequences: Dict[str, List[str]],
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 100000,
    random_seed: int = 42,
):
    """Interpret the clusters and return a JSON string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(system_prompt))

    clustered_sequences = _subsample_sequences(
        clustered_sequences,
        encoding,
        max_tokens - token_count,
        random_seed,
    )
    query_prompt = f"```json\n{json.dumps(clustered_sequences)}\n```"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
        ],
    )
    response_json, response_text = _parse_response(response)

    return {
        "raw_response": response_text,
        "descriptions": response_json,
    }


def cluster_results_to_html(cluster_results: Dict[str, Any]) -> str:
    """Convert cluster results to HTML for display."""
    styles = """
    <style>
        .cluster-table {
            width: 100%;
            margin: 20px 0;
            border-collapse: collapse;
            text-align: left !important;
        }
        .cluster-header {
            background-color: #f8f9fa;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
            border: 1px solid #dee2e6;
            text-align: left !important;
        }
        .summary-cell {
            padding: 15px;
            border: 1px solid #dee2e6;
            text-align: left !important;
        }
        .keyword-tag {
            display: inline-block;
            background-color: #e7f3ff;
            color: #0066cc;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 12px;
            text-align: left !important;
        }
        .pattern-container {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
            text-align: left !important;
        }
        .pattern-container:last-child {
            border-bottom: none;
        }
    </style>
    """

    html_output = styles
    for cluster_name, patterns in cluster_results.items():
        html_output += f"""
        <table class="cluster-table">
            <tr><td class="cluster-header">{cluster_name}</td></tr>
            <tr><td class="summary-cell">
        """

        for j, pattern in enumerate(patterns):
            html_output += """
            <div class="pattern-container">
                <div style="margin-bottom: 10px;"><strong>{}:</strong> {}</div>
                <div>
            """.format(
                j + 1, pattern["summary"]
            )

            for keyword in pattern["keywords"]:
                html_output += f'<span class="keyword-tag">{keyword}</span>'

            html_output += "</div></div>"

        html_output += "</td></tr></table>"
    return html_output
