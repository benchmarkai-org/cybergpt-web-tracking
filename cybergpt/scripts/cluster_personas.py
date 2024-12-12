import argparse
import os
import pickle
from dotenv import load_dotenv
from openai import OpenAI

from cybergpt.prompting.clusters import extract_personas


def main(response_pkl: str, model_name: str = "gpt-4o"):
    load_dotenv()
    responses = pickle.load(open(response_pkl, "rb"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = open("cybergpt/prompting/ic_persona_system_prompt.txt", "r").read()
    personas = extract_personas(
        client,
        system_prompt,
        responses,
        model_name=model_name,
    )
    pickle.dump(personas, open("data/prompting/ic_personas.pkl", "wb"))


if __name__ == "__main__":
    """
    Example usage:
    python -m cybergpt.scripts.cluster_personas \
        --response_pkl data/prompting/ic_output.pkl \
        --model_name gpt-4o
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--response_pkl", type=str, default="data/prompting/ic_output.pkl"
    )
    args.add_argument("--model_name", type=str, default="gpt-4o")
    args = args.parse_args()
    main(args.response_pkl, args.model_name)
