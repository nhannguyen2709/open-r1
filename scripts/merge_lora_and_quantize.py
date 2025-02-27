from functools import partial
import gc
import os
import shutil

from awq import AutoAWQForCausalLM
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)

from configs import H4ArgumentParser, ModelArguments, DataArguments
from model_utils import load_model_and_tokenizer


def get_conversation(row: pd.Series, tokenizer: PreTrainedTokenizer):
    winner = row["winner"].replace("model_", "")
    chosen = row[f"response_{winner}"]
    conversation = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": chosen},
    ]
    row["message"] = tokenizer.apply_chat_template(conversation, tokenize=False)
    return row


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, _, training_args = parser.parse()
    output_dir = training_args.output_dir
    quant_dir = output_dir + f"-{model_args.quantization}"
    checkpoint_dir = training_args.resume_from_checkpoint

    model, tokenizer = load_model_and_tokenizer(
        model_args, training_args, skip_lora_loading=True
    )
    if not os.path.exists(
        os.path.join(training_args.resume_from_checkpoint, "adapter_config.json")
    ):
        print("Adapter config not found, creating one now ...")
        model.peft_config["default"].save_pretrained(checkpoint_dir)

    model.load_adapter(
        training_args.resume_from_checkpoint,
        model.active_adapter,
        is_trainable=False,
    )
    model.merge_and_unload()
    model.base_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    choice_idxs = tokenizer.encode(["A", "B"], add_special_tokens=False)
    lm_head_weight = model.lm_head.weight[choice_idxs]
    torch.save(lm_head_weight, os.path.join(output_dir, "classifier.pt"))
    print(f"Merged model and tokenizer saved to {output_dir}")

    if model_args.quantization == "":
        print("Skipping quantization")
        return

    del model
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    df = pd.read_parquet("data/train.parquet")
    df = df[df["split"] == "train"]
    df = (
        df.groupby("language")
        .apply(lambda x: x.sample(min(16, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    df = df.apply(partial(get_conversation, tokenizer=tokenizer), axis=1)
    print(f"Num. samples for calibration: {len(df)}")

    if model_args.quantization == "awq":
        model = AutoAWQForCausalLM.from_pretrained(output_dir)
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        print(f"Quantizing model with AWQ config: {quant_config}")
        model.quantize(
            tokenizer,
            quant_config,
            calib_data=df["message"].tolist(),
            n_parallel_calib_samples=32,
            max_calib_seq_len=4096,
        )
        model.save_quantized(quant_dir)
        tokenizer.save_pretrained(quant_dir)
        print(f"Quantized model and tokenizer saved to {quant_dir}")

    elif model_args.quantization == "gptq":
        gptq_config = GPTQConfig(
            bits=8,
            cache_block_outputs=True,
            damp_percent=0.01,
            group_size=128,
            sym=True,
            true_sequential=True,
            use_exllama=True,
            dataset=df["message"].tolist(),
            tokenizer=tokenizer,
        )
        print("Quantizing model with GPTQ")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir, device_map="auto", quantization_config=gptq_config
        )
        model.save_pretrained(quant_dir)
        tokenizer.save_pretrained(quant_dir)
        print(f"Quantized model and tokenizer saved to {quant_dir}")

    shutil.copy(
        os.path.join(output_dir, "classifier.pt"),
        os.path.join(quant_dir, "classifier.pt"),
    )


if __name__ == "__main__":
    main()