from openai import OpenAI
from tqdm import tqdm  # Import tqdm for the progress bar
import json
import os

client = OpenAI(api_key="sk-proj-36RHqLPBUOVeAHSit9C5T3BlbkFJF4YPea1Mpgx5H7hDF4vp")

# Function to call GPT-3 API to clean up prompts
def clean_prompt(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo", # or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant that refines prompts for clarity and conciseness."},
        {"role": "user", "content": f'Turn this into a user prompt for an LLM, asking the LLM to generate LEGO instructions for this description. Do not include "User Prompt:":\n{prompt}'}
    ],
    max_tokens=150,
    n=1,
    stop=None,
    temperature=0.7)
    return response.choices[0].message.content.strip()

def clean_instructions_chunk(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo", # or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant that refines prompts for clarity and conciseness."},
        {"role": "user", "content": f"Please rewrite the following detailed LEGO building instructions to be more concise, while still clear and easy to follow. The goal is to simplify the language and reduce redundancy without losing essential information or steps.\n{prompt}"}
    ],
    max_tokens=1500,
    n=1,
    stop=None,
    temperature=0.7)
    return response.choices[0].message.content.strip()

def split_text(text, max_length):
    # Split text into chunks of max_length tokens
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def clean_instructions(prompt):
    max_chunk_length = 16000 
    chunks = split_text(prompt, max_chunk_length)
    cleaned_chunks = [clean_instructions_chunk(chunk) for chunk in chunks]
    return ' '.join(cleaned_chunks)

# Load the JSON data
with open('cleaned_text/sorted_sections.json') as f:
    data = json.load(f)

# Prepare the data for fine-tuning in chat format and chatML format
chat_data = []
chatML_data = []


# Use tqdm to add a progress bar to the loop
for filename, content in tqdm(data.items(), desc="Cleaning instruction sets"):
    introduction = content["introduction"]
    terms = content["terms"]
    sorting = content["sorting"]
    instructions = content["instructions"]
    abbreviations = content["abbreviations"]

    # Clean up the introduction using the GPT-3 API
    cleaned_introduction = clean_prompt(introduction)
    cleaned_instructions = clean_instructions(instructions)

    # Prepare concise and clear prompts for GPT-style chat format
    chat_data.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant providing detailed LEGO building instructions."},
            {"role": "user", "content": f"Provide step-by-step instructions for a LEGO build with this description:\n{cleaned_introduction}"},
            {"role": "assistant", "content": cleaned_instructions}
        ]
    })

    # Prepare concise and clear prompts for chatML format
    chatML_data.append(
        f"system\nYou are a helpful assistant providing detailed LEGO building instructions.\n\n"
        f"user\nProvide step-by-step instructions for a LEGO build with this description:\n{cleaned_introduction}\n\n"
        f"assistant\n{cleaned_instructions}\n\n"
    )

# Save the GPT-style chat data in JSONL format
with open('training_data/GPT-style_trainingdata.jsonl', 'w') as f:
    for entry in chat_data:
        f.write(json.dumps(entry) + '\n')

# Save the chatML data in a text file
with open('training_data/chatML_trainingdata.txt', 'w') as f:
    for entry in chatML_data:
        f.write(entry + '\n')

print("Data has been successfully converted to GPT-style chat format JSONL and chatML format text.")
