import json
import tiktoken
import random
import re
from typing import Dict, List, Any, Tuple
from openai import OpenAI
from functools import reduce


def _subsample_classes(
    class_sequences: Dict[str, List[str]],
    encoding: tiktoken.Encoding,
    max_tokens: int = 100000,
    random_seed: int = 42,
    batch_limit: int = 10000,
):
    """Subsample classes of sequences until the token budget is reached."""
    random.seed(random_seed)

    available_classes = list(class_sequences.keys())
    sampled_classes = []

    token_count = 0
    while (
        token_count < max_tokens
        and available_classes
        and (len(sampled_classes) < batch_limit)
    ):
        class_ = random.choice(available_classes)
        class_tokens = sum(len(encoding.encode(seq)) for seq in class_sequences[class_])
        if token_count + class_tokens <= max_tokens:
            sampled_classes.append(class_)
            available_classes.remove(class_)
            token_count += class_tokens
        else:
            if len(sampled_classes) == 0:
                # If we hit the token limit with a single class, subsample the sequences of that class
                sampled_sequences = _subsample_sequences(
                    {class_: class_sequences[class_]},
                    encoding,
                    max_tokens,
                    random_seed,
                )
                return {
                    class_: sampled_sequences,
                }
            break

    return {c: class_sequences[c] for c in sampled_classes}


def profile_classes(
    client: OpenAI,
    system_prompt: str,
    class_sequences: Dict[str, List[str]],
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 100000,
    random_seed: int = 42,
    batch_limit: int = 50,
):
    """Profile the classes using the LLM."""
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(system_prompt))

    total_classes = len(class_sequences)
    processed_classes = 0

    random.seed(random_seed)
    responses = []

    print(f"Processing {total_classes} classes in batches...")

    while class_sequences:
        try:
            new_seed = random.randint(0, 10000000)
            classes_to_profile = _subsample_classes(
                class_sequences,
                encoding,
                max_tokens - token_count,
                new_seed,
                batch_limit,
            )

            num_classes = len(classes_to_profile)
            processed_classes += num_classes
            print(
                f"Processing batch of {num_classes} classes ({processed_classes}/{total_classes})"
            )

            query_prompt = f"```json\n{json.dumps(classes_to_profile)}\n```"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query_prompt},
                    ],
                )
                response_json, response_text = _parse_json_response(response)
                responses.append((response_json, response_text))

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                # Skip failed batch but continue processing
                print("Skipping failed batch and continuing...")

            # Remove processed classes
            class_sequences = {
                k: v for k, v in class_sequences.items() if k not in classes_to_profile
            }

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            break

    print(f"Completed processing {processed_classes} classes")
    return responses


def extract_personas(
    client: OpenAI,
    system_prompt: str,
    responses: List[Tuple[Dict[str, Any], str]],
    model_name: str = "gpt-4o",
):
    """Extract personas from the class profile responses."""
    class_dict = reduce(lambda x, y: x | y, [d[0] for d in responses])
    query_prompt = f"```json\n{json.dumps(class_dict)}\n```"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
        ],
    )
    response_json, _ = _parse_json_response(response)
    return response_json


def personas_to_html(personas: Dict[str, Any]) -> str:
    """Convert personas to HTML for display."""
    styles = """
    <style>
        .segment-table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: Arial, sans-serif;
        }
        .segment-table th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .segment-table td {
            padding: 12px;
            border: 1px solid #ddd;
            vertical-align: top;
        }
        .keyword-pill {
            display: inline-block;
            background-color: #e9ecef;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.9em;
        }
        .count-badge {
            background-color: #007bff;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.9em;
        }
    </style>
    """

    html = (
        styles
        + """
    <table class="segment-table">
        <tr>
            <th>Persona</th>
            <th>Description</th>
            <th>Keywords</th>
            <th>Count</th>
        </tr>
    """
    )

    for persona, info in personas.items():
        keywords_html = " ".join(
            [f'<span class="keyword-pill">{kw}</span>' for kw in info["keywords"]]
        )
        count_html = f'<span class="count-badge">{info["count"]}</span>'

        html += f"""
        <tr>
            <td><strong>{persona}</strong></td>
            <td>{info['description']}</td>
            <td>{keywords_html}</td>
            <td>{count_html}</td>
        </tr>
        """

    html += "</table>"
    return html


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        return f"{minutes}m"
    return f"{minutes}m {remaining_seconds}s"


def _parse_sequence(sequence):
    # Extract timestamp and visits
    timestamp, visits = sequence.split(", Visits: ")
    day, time = timestamp.split(" ")

    # Parse visits into list of tuples (site, duration)
    visit_pattern = r"(\w+\.(?:com|de))\s+\((\d+)s\)"
    visits = re.findall(visit_pattern, visits)
    return {
        "day": day,
        "time": time,
        "visits": [(site, int(duration)) for site, duration in visits],
    }


def user_profile_and_persona_to_html(user_profile: Dict[str, Any]) -> str:
    """Convert user profile and persona to HTML for display."""
    styles = """
    <style>
        .user-profile {
            font-family: Arial, sans-serif;
            max-width: 100%;
            margin: 20px 0;
        }
        .header {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .keyword-pill {
            display: inline-block;
            background-color: #e9ecef;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.9em;
        }
        .sequence {
            margin: 15px 0;
            padding: 15px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
        .timestamp {
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }
        .visit {
            display: inline-block;
            margin: 2px;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .visit-other {
            background-color: #6c757d;
            color: white;
        }
        .arrow {
            color: #adb5bd;
            margin: 0 5px;
        }
    </style>
    """

    # Create header HTML
    html = (
        styles
        + """
    <div class="user-profile">
        <div class="header">
            <p><strong>Persona:</strong> {}</p>
            <p><strong>User Profile:</strong> {}</p>
            <div>
    """.format(
            user_profile["persona"], user_profile["description"]
        )
    )

    # Add keywords
    for keyword in user_profile["keywords"]:
        html += f'<span class="keyword-pill">{keyword}</span>'

    html += "</div></div>"

    # Add sequences
    for sequence in user_profile["sequences"]:
        parsed = _parse_sequence(sequence)
        html += f"""
        <div class="sequence">
            <div class="timestamp">{parsed['day']} {parsed['time']}</div>
        """

        # Add visits
        for i, (site, duration) in enumerate(parsed["visits"]):
            html += f'<span class="visit visit-other">{site} ({_format_duration(duration)})</span>'
            if i < len(parsed["visits"]) - 1:
                html += '<span class="arrow">→</span>'

        html += "</div>"

    html += "</div>"
    return html


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


def _parse_json_response(response: Any) -> Tuple[Dict[str, Any], str]:
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
    response_json, response_text = _parse_json_response(response)

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
