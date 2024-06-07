import json

# Paths to the JSONL file and the output JSONL file
jsonl_file_path = 'training_data/GPT-style_trainingdata.jsonl'
output_jsonl_file_path = 'training_data/train_guanacostyle.jsonl'

# Function to convert JSONL to the SFT / Generic Trainer format in JSONL
def convert_jsonl_to_sft_jsonl(jsonl_path, output_path):
    with open(jsonl_path, 'r') as jsonl_file, open(output_path, 'w') as output_file:
        for line in jsonl_file:
            message = json.loads(line.strip())
            conversation = []
            for msg in message['messages']:
                if msg['role'] == 'system':
                    continue  # Skip the system messages
                if msg['role'] == 'user':
                    content = msg['content'].replace('Provide step-by-step instructions for a LEGO build with this description:\n', '')
                    conversation.append(f"<s>[INST] {content}")
                else:
                    conversation.append(f"[/INST] {msg['content']} </s>")
            output_file.write(json.dumps({"text": " ".join(conversation)}) + '\n')

# Run the conversion
convert_jsonl_to_sft_jsonl(jsonl_file_path, output_jsonl_file_path)
