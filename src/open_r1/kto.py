import argparse
from typing import Optional
from dataclasses import dataclass, field

from accelerate import PartialState
from datasets import Dataset, load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl
from trl import (
    ScriptArguments,
    TrlParser,
    get_peft_config,
    setup_chat_format,
)
from open_r1.kto_trainer import KTOTrainer
from open_r1.utils.model_utils import get_quantization_config, get_tokenizer


def prepare_unpaired_pref_dataset(example: dict) -> dict:
    generations = example["generations"]
    is_reasoning_complete = example["is_reasoning_complete"]
    correctness_math_verify = example["correctness_math_verify"]

    examples = []
    for g, c1, c2 in zip(generations, correctness_math_verify, is_reasoning_complete):
        label = c1 and c2
        examples.append(
            {
                "prompt": [{"role": "user", "content": example["problem"]}],
                "completion": [{"role": "assistant", "content": g}],
                "label": label,
                "solution": example["solution"],
                "answer": example["answer"],
            }
        )
    return examples


@dataclass
class KTOConfig(trl.KTOConfig):
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    max_prompt_length: Optional[int] = field(
        default=None, metadata={"help": "The maximum length of the prompt."}
    )
    max_completion_length: Optional[int] = field(
        default=None, metadata={"help": "The maximum length of the completion."}
    )


@dataclass
class ModelConfig(trl.ModelConfig):
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={
            "help": "This sets the storage type to pack the quanitzed 4-bit params."
        },
    )


def main(script_args, training_args, model_args):
    tokenizer = get_tokenizer(model_args, training_args)
    quantization_config = get_quantization_config(model_args)
    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
    )
    if not model_args.use_peft:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=quantization_config,
        )
    else:
        ref_model = None

    # Load the dataset
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split="train"
    )
    df = dataset.to_pandas()
    df = df.apply(prepare_unpaired_pref_dataset, axis=1)
    df = pd.DataFrame(df.explode().tolist())
    train_dataset = Dataset.from_pandas(df)

    # Initialize the KTO trainer
    trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
