import json

# Load the JSON data
with open('lego_instructions.json') as f:
    data = json.load(f)

# Prepare the data for fine-tuning in chat format
chat_data = []

for filename, content in data.items():

    introduction = content["introduction"]
    terms = content["terms"]
    sorting = content["sorting"]
    instructions = content["instructions"]
    abbreviations = content["abbreviations"]

    if len(abbreviations) == 0:
        chat_data.append({
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant providing LEGO instructions. Some terms you should be familiar with:\n{terms}"},
                {"role": "user", "content": f"Please provide instructions for category: {category} and file: {filename}."},
                {"role": "assistant", "content": text}
            ]
        })
    else:
        chat_data.append({
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant providing LEGO instructions. Some terms you should be familiar with:\n{terms}"},
                {"role": "user", "content": f"Please provide instructions for category: {category} and file: {filename}."},
                {"role": "assistant", "content": text}
            ]
        })

# Save the prepared data in JSONL format
with open('lego_training_data_prepared_chat.jsonl', 'w') as f:
    for entry in chat_data:
        f.write(json.dumps(entry) + '\n')

print("Data has been successfully converted to chat format JSONL.")
