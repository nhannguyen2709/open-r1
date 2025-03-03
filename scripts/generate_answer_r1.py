import pandas as pd
from pandarallel import pandarallel
from openai import OpenAI
import time
from tqdm import tqdm
import os
import json
from time import time
import re
from datasets import load_dataset
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
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": row["problem"]},
            ],
            temperature=0.6,
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
    df = pd.read_parquet("/mnt/weka/lipsync/openr1_int.parquet")
    df_1 = df[df['problem_type'].isin(['Calculus', 'Other', 'Inequalities'])][500:]
    df_2 = df[df['problem_type'].isin(['Logic and Puzzles'])][250:]
    df = pd.concat([df_1, df_2], ignore_index=True)
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Total rows: {len(df)}")
    for j in range(0, len(df), 8):
        print(f"Processing from row {j}")
        _df = df[j:j+8]
        i = 1
        while i <= 3:
            _df[f"# {i}"] = _df.parallel_apply(lambda x: get_openai_response(x), axis=1)
            i += 1
        try:
            if os.path.exists("_openr1_int_score.parquet"):
                existing_df = pd.read_parquet("_openr1_int_score.parquet")
                combined_df = pd.concat([existing_df, _df], ignore_index=True)
                combined_df.to_parquet("_openr1_int_score.parquet", index=False)
            else:
                _df.to_parquet("_openr1_int_score.parquet", index=False)
        except:
            continue

    # DEBUG
    # df = df[:1]
    # df.apply(lambda x: get_openai_response(x), axis=1)



if __name__ == "__main__":
    main()
