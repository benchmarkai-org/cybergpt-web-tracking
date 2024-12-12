import argparse
import json
import os
import pickle
from functools import reduce
from dotenv import load_dotenv
from openai import OpenAI

from cybergpt.prompting.clusters import profile_classes


def main(personas_pkl: str, sequences_pkl: str, model_name: str = "gpt-4o-mini"):
    load_dotenv()
    persona_allocations = pickle.load(open(personas_pkl, "rb"))
    personas = persona_allocations["personas"]
    personas = dict(sorted(personas.items()))  # sort alphabetically
    user_map = persona_allocations["users"]
    sequences = pickle.load(open(sequences_pkl, "rb"))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = open(
        "cybergpt/prompting/ic_persona_eval_system_prompt.txt", "r"
    ).read()
    system_prompt = (
        f"{system_prompt}\n<PERSONAS>\n{json.dumps(personas, indent=2)}\n</PERSONAS>"
    )

    test_sequences = sequences["test_sequences"]
    responses = profile_classes(
        client,
        system_prompt,
        test_sequences,
        model_name=model_name,
        max_tokens=100000,
        random_seed=99,
        batch_limit=50,
    )

    test_user_map = reduce(lambda x, y: x | y, [d[0] for d in responses])
    output = {
        "personas": personas,
        "user_map": user_map,
        "test_user_map": test_user_map,
    }
    pickle.dump(output, open("data/prompting/ic_persona_eval.pkl", "wb"))


if __name__ == "__main__":
    """
    Example usage:
    python -m cybergpt.scripts.cluster_personas_eval \
        --personas_pkl data/prompting/ic_personas.pkl \
        --sequences_pkl data/prompting/sequences.pkl \
        --model_name gpt-4o
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--personas_pkl", type=str, default="data/prompting/ic_personas.pkl"
    )
    args.add_argument(
        "--sequences_pkl", type=str, default="data/prompting/sequences.pkl"
    )
    args.add_argument("--model_name", type=str, default="gpt-4o")
    args = args.parse_args()
    main(args.personas_pkl, args.sequences_pkl, args.model_name)
