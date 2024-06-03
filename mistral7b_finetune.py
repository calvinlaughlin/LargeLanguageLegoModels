# You only need to run this once per machine
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets matplotlib

from datasets import load_dataset
import json

# Load the original JSON file for training
with open('train.json', 'r') as file:
    train_data = json.load(file)

# Load the original JSON file for evaluation
with open('eval.json', 'r') as file:
    eval_data = json.load(file)

# Combine all records from different fields into a single list for training
combined_train = []
for key in train_data.keys():
    # combined_train.extend(train_data[key]['introduction'])
    # combined_train.extend(train_data[key]['terms'])
    # combined_train.extend(train_data[key]['misc'])
    # combined_train.extend(train_data[key]['sorting'])
    combined_train.extend(train_data[key]['instructions'])
    # combined_train.extend(train_data[key]['ads'])
    # combined_train.extend(train_data[key]['abbreviations'])

# Combine all records from different fields into a single list for evaluation
combined_eval = []
for key in eval_data.keys():
    # combined_eval.extend(eval_data[key]['introduction'])
    # combined_eval.extend(eval_data[key]['terms'])
    # combined_eval.extend(eval_data[key]['misc'])
    # combined_eval.extend(eval_data[key]['sorting'])
    combined_eval.extend(eval_data[key]['instructions'])
    # combined_eval.extend(eval_data[key]['ads'])
    # combined_eval.extend(eval_data[key]['abbreviations']) #lets probably just have instructions

# Save the combined records into new JSON files
with open('combined_train.json', 'w') as file:
    json.dump(combined_train, file, indent=4)
with open('combined_eval.json', 'w') as file:
    json.dump(combined_eval, file, indent=4)

# Load the combined dataset
train_dataset = load_dataset('json', data_files='combined_train.json', split='train')
eval_dataset = load_dataset('json', data_files='combined_eval.json', split='train')



from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)



def formatting_func(example):
    text = example['text']
    return f"### The following is text based LEGO instruction manual for someone that is visually impaired: {text}"

    # sections = ["introduction", "terms", "misc", "sorting", "instructions", "ads", "abbreviations"]
    # text_parts = []
    
    # for section in sections:
    #     if section in example and example[section]:
    #         # If the section is a list, join its elements; otherwise, use it directly
    #         if isinstance(example[section], list):
    #             text_parts.append(" ".join(example[section]))
    #         else:
    #             text_parts.append(example[section])
    
    # # Join all parts into a single text string
    # text = " ".join(text_parts)
    # return f"### The following is text based LEGO instruction manual for someone that is visually impaired: {text}"


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Replace 'your_token' with your actual Hugging Face token
token = "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"

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # maybe no 4bit
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    use_auth_token=token  # Pass the token here
)


from transformers import AutoTokenizer

# Replace 'your_token' with your actual Hugging Face token
token = "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"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_auth_token=token  # Pass the token here
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)


max_length = 1000 # This was an appropriate max length for my dataset, should I change it?, 10,000 for this one and try lower for next

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

print(tokenized_train_dataset[1]['input_ids'])

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

eval_prompt = " The following is text based LEGO instruction manual for someone that is visually impaired : # "

# Init an eval tokenizer that doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True))


!pip install -q wandb -U

import wandb, os
wandb.login()

wandb_project = "journal-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

model = accelerator.prepare_model(model)

import transformers
from datetime import datetime

project = "journal-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-6, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=10,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)


from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-journal-finetune/checkpoint-300")


eval_prompt = "The following is text based LEGO instruction manual for someone that is visually impaired : # "
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))