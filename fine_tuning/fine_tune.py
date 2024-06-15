from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, GPTQConfig, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset
import torchsummary
import os

# the create_data package is a placeholder
import create_data

output_dir = os.path.join("saved_models", "base_v0.1")
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

model_name = "TheBloke/Llama-2-70B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=GPTQConfig(bits=4, disable_exllama=True))
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

tokenizer.pad_token = tokenizer.eos_token #"<|PAD|>" 
tokenizer.padding_side = "left"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    #per_device_train_batch_size=script_args.per_device_train_batch_size,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    max_grad_norm=1,
    lr_scheduler_type="constant",
    num_train_epochs=5,
    report_to="none"
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=20,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

def prompt_and_response_train(datum):
    prompt = f"Passage:\n\n{datum['passage']}\n\nRelationships"
    response = ""
    for i,rel in enumerate(datum['rels']):
        response += f"\n\nRelationship {i+1}: a {rel['direction_head']} in the magnitude of [{rel['head']}] results in a {rel['direction_tail']} in the magnitude of [{rel['tail']}]"
    if len(datum['rels']) == 0:
        response += "\n\n<|NO RELATIONSHIPS FOUND|>"
    return f"{prompt}{response}"

def tokenize(element):
    outputs = tokenizer(
        prompt_and_response_train(element),
        truncation=True,
        padding=False,
        max_length=2048,
        return_overflowing_tokens=False,
        return_length=True,
    )
    return outputs

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
torchsummary.summary(model)
model.print_trainable_parameters()
trainer = Trainer(
    model=model,
    train_dataset=Dataset.from_list(create_data.load_data_all()).map(tokenize),
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()
trainer.model.save_pretrained(final_checkpoint_dir)