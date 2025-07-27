import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import argparse
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--datasets", type=str, default='')
    parser.add_argument("--server", type=str, default='1')

    args = parser.parse_args()

    model_path = args.model_path


    # ---------------------------------------------------------------------------------------
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,   # Enables 8-bit quantization
        torch_dtype=torch.bfloat16,  # Mixed precision for faster inference
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["qkv_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    print(model.device)
    # ---------------------------------------------------------------------------------------
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False
    )

    print(tokenizer.eos_token_id)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    # Tokenizing function
    def preprocess_function(example):
        # Select appropriate column based on train data
        prompt = f"{example['input']}\n\n{example['output']}"
        # prompt = f"{example['input']}\n\n{example['fol']}"
        tokenized_inputs = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    data_path = {"train": "./data/finale/predicate_train_fol_22480.json"}

    raw_dataset = load_dataset('json', data_files=data_path)
    dataset = raw_dataset["train"].train_test_split(test_size=1000, shuffle=True, seed=42)

    print('start processing')
    tokenized_datasets = dataset.map(preprocess_function)
    tokenized_datasets = {
        split: ds.remove_columns(
            [col for col in ds.column_names if col not in ("input_ids", "attention_mask", "labels")])
        for split, ds in tokenized_datasets.items()
    }

    # ---------------------------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Important for causal LM
    )

    # DDP and Gradient Accumulation
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    GRADIENT_ACCUMULATION_STEPS = args.total_batch_size // args.per_device_train_batch_size
    if ddp:
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    print(f"WORLD_SIZE={world_size}, DDP={ddp}")

    # Training Output Paths
    run_name = f'bs_{args.total_batch_size}_lr_{args.learning_rate}_datasets_{args.datasets}'
    output_path = f'./sft-logic.{args.server}/{run_name}'
    os.makedirs(output_path, exist_ok=True)

    steps_total = 150

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        eval_strategy="steps",
        eval_steps=steps_total,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=steps_total,
        save_strategy="steps",
        save_steps=steps_total,
        bf16=True,
        report_to="wandb",  # Set to "wandb" if you are using Weights and Biases for logging
        dataloader_num_workers=4, # Keep 1 if CPU does not support
        load_best_model_at_end=True,
        push_to_hub=False,
        run_name=run_name,
        remove_unused_columns=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    op_path = f'{args.model_path.split("/")[-1].strip()}_{args.total_batch_size}_lr_{args.learning_rate}_datasets_{args.datasets}'
    new_model = os.path.join('./models/', op_path)
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)


if __name__ == "__main__":
    main()
