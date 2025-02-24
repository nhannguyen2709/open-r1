import pandas as pd
from pandarallel import pandarallel
from openai import OpenAI
import time
from tqdm import tqdm
import os
import json
from time import time
import re

# Initialize pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=8)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def get_openai_response(row):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": row["problem"] + "\nPlease put the final answer within \\boxed{}."},
            ],
            temperature=0.7,
            top_p=0.8,
            max_tokens=8192,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        generated = response.choices[0].message.content
        # print(generated)
        # print(int(extract_boxed_text(generated.split("</think>")[1])))
        if "</think>" in generated:
            return int(extract_boxed_text(generated.split("</think>")[1]))
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # Read the CSV file
    df = pd.read_parquet("openr1_int.parquet")
    df = df[:3000]
    i = 1
    while i <= 7:
        print(f"TTA # {i}")
        df[f"# {i}"] = df.parallel_apply(lambda x: get_openai_response(x), axis=1)
        i += 1
        df.to_parquet("openr1_int_sample.parquet", index=False)

    # DEBUG
    # df = df[:1]
    # df.apply(lambda x: get_openai_response(x), axis=1)



if __name__ == "__main__":
    main()