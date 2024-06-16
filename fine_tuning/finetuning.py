import argparse
import time

import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    TextStreamer
)
from peft import  LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import EarlyStoppingCallback



if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser()

    # arguments for the script 
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="large language model name from huggingface")
    parser.add_argument("--dataset-dir", type=str, help="huggingface dataset object with train and val splits directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="directory to store the fine tuned model")
    parser.add_argument("--lora-r", type=int, default=2,  help="lora r parameter value")
    parser.add_argument("--lora-alpha", type=int, default=4, help="lora alpha parameter value")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="lora dropout parameter value")
    parser.add_argument("--use-4bit", type=bool, default=True, help="use 4-bit precision in base model loading")
    parser.add_argument("--bnb-4bit-compute-dtype", type=torch.dtype, default=torch.bfloat16, help="Compute dtype for 4-bit base model")
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    parser.add_argument("--use-nested-quant", type=bool, default=True, help="Activate nested quantization for 4-bit base models (double quantization)")
    parser.add_argument("--fp16", type=bool, default=False, help="Enable fp16 training")
    parser.add_argument("--bf16", type=bool, default=False, help="Enable bf16 training")
    parser.add_argument("--num-train-epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Batch size per GPU for training")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4, help="Batch size per GPU for evaluation")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of update steps to accumulate the gradient for")
    parser.add_argument("--gradient-checkpointing", type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--max-grad-norm", type=float, default=0.3, help="Maximum gradient normal i.e. gradient clipping value")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="initial learning rate (AdamW optimizer)")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay to apply to all layers except bias/layernorm wrights")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="learning rate scheduler to use")
    parser.add_argument("--max-steps", type=int, default=-1, help="Number of training steps (overrides num_train_epochs)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Ratio of steps for a linear warmup (from 0 to learning rate)")
    parser.add_argument("--group-by-length", type=bool, default=True, help="Group sequences into batches with same length. Saves memory and speeds up training considerably")
    parser.add_argument("--logging-steps", type=int, default=100, help="log every x update steps")
    parser.add_argument("--save-strategy", type=str, default="epoch", help="Strategy to save models after every x ")
    parser.add_argument("--save-total-limit", type=int, default=1, help="Total number of models to save while training")
    parser.add_argument("--evaluation-strategy", type=str, default="epoch", help="Run evaluation of eval set after every x ")
    parser.add_argument("--load-best-model-at-end", type=bool, default=True, help="Load the best model at the end of training.")
    parser.add_argument("--metric-for-best-model", type=str, default="eval_loss", help="Metric to use to identify best model")
    parser.add_argument("--greater-is-better", type=bool, default=False, help="whether above defined metric is better when increases or poorer")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Maximul sequence length to use while training")
    parser.add_argument("--packing", type=bool, default=False, help="Pack multiple short examples in the same input sequence to increase efficiency")
    parser.add_argument("--report-to", type=str, default=None, help="If you want to report the logging output to a hosted service like MLFlow")

    args = parser.parse_args()

    model_name = args.model_name
    dataset_dir = args.dataset_dir
    output_dir =  args.output_dir
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    use_4bit = args.use_4bit
    bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype
    bnb_4bit_quant_type = args.bnb_4bit_quant_type
    use_nested_quant = args.use_nested_quant
    num_train_epochs = args.num_train_epochs
    fp16 = args.fp16
    bf16 = args.bf16
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    gradient_checkpointing = args.gradient_checkpointing
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optim = args.optim
    lr_scheduler_type = args.lr_scheduler_type
    max_steps = args.max_steps
    warmup_ratio = args.warmup_ratio
    group_by_length = args.group_by_length
    logging_steps = args.logging_steps
    save_strategy = args.save_strategy
    save_total_limit = args.save_total_limit
    evaluation_strategy = args.evaluation_strategy
    load_best_model_at_end = args.load_best_model_at_end
    metric_for_best_model = args.metric_for_best_model
    greater_is_better = args.greater_is_better
    max_seq_length = args.max_seq_length
    packing = args.packing
    report_to = args.report_to
    device_map = {"": 0}


    # loading datasets
    dataset = datasets.load_from_disk(dataset_dir)['train']
    eval_dataset = datasets.load_from_disk(dataset_dir)['val']

    # Load tokenizer and model with QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"] # applying lora weights to all matrices
        # we are using all the modules
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    early_stop = EarlyStoppingCallback(3)
    
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        evaluation_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        load_best_model_at_end = load_best_model_at_end,
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model, train_dataset=dataset,
        eval_dataset = eval_dataset, peft_config=peft_config,
        dataset_text_field="text", max_seq_length=max_seq_length,
        tokenizer=tokenizer, args=training_arguments,
        packing=packing, callbacks=[early_stop]
    )

    # Train model
    trainer.train()

    # evaluating the model
    trainer.evaluate()

    time_stop = time.time()
    total_time_taken = (b-a)/60.0
    print('Finished Training model.........')
    print('Total time taken is : {} minutes'.format(total_time_taken))
