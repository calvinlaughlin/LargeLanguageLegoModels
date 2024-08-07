import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import matplotlib.pyplot as plt
import wandb, os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import transformers
from datetime import datetime

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def combine_records(data):
    combined = []
    for key in data.keys():
        combined.extend(data[key]['instructions'])
    return combined

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

train_data = load_json('train.json')
eval_data = load_json('eval.json')

combined_train = combine_records(train_data)
combined_eval = combine_records(eval_data)

save_json(combined_train, 'combined_train.json')
save_json(combined_eval, 'combined_eval.json')

train_dataset = load_dataset('json', data_files='combined_train.json', split='train')
eval_dataset = load_dataset('json', data_files='combined_eval.json', split='train')

# Model config and initialization
token = "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"
base_model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    use_auth_token=token
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_auth_token=token
)
tokenizer.pad_token = tokenizer.eos_token

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Data formatting and tokenization
def format_func(example):
    text = example['text']
    return f"### The following is text based LEGO instruction manual for someone that is visually impaired: {text}"

def gen_and_tokenize_prompt(prompt):
    return tokenizer(format_func(prompt))

tokenized_train_ds = train_dataset.map(gen_and_tokenize_prompt)
tokenized_val_ds = eval_dataset.map(gen_and_tokenize_prompt)

def plot_lengths(tokenized_train_ds, tokenized_val_ds):
    lengths = [len(x['input_ids']) for x in tokenized_train_ds]
    lengths += [len(x['input_ids']) for x in tokenized_val_ds]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

plot_lengths(tokenized_train_ds, tokenized_val_ds)

max_len = 1000  # Adjust as needed

def gen_and_tokenize_prompt2(prompt):
    result = tokenizer(
        format_func(prompt),
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_ds = train_dataset.map(gen_and_tokenize_prompt2)
tokenized_val_ds = eval_dataset.map(gen_and_tokenize_prompt2)

plot_lengths(tokenized_train_ds, tokenized_val_ds)

eval_prompt = "The following is text based LEGO instruction manual for someone that is visually impaired : # "
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True))

# Training config and execution
wandb.login()

wandb_project = "journal-finetune"
os.environ["WANDB_PROJECT"] = wandb_project

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

print_trainable_params(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
        "up_proj", "down_proj", "lm_head"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_params(model)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

model = accelerator.prepare_model(model)

project = "journal-finetune"
run_name = f"{base_model_id}-{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    args=transformers.TrainingArguments(
        output_dir="./" + run_name,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-6,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=10,
        do_eval=True,
        report_to="wandb",
        run_name=run_name
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()

# Post-training evaluation
ft_model = PeftModel.from_pretrained(model, "mistral-journal-finetune/checkpoint-300")

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))
