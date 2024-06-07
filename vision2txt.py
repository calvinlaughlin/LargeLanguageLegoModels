# Project API key: sk-proj-HMzMuOfGEJwmH58C8p2AT3BlbkFJOnzsQdIok4MngMnPNE4M
import base64
import requests
import os
import re

# OpenAI API Key
api_key = "sk-proj-ICXYnrO45vlQuUtjQK8vT3BlbkFJ0t74jmpKqIMq1Fqawd2Q"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

folder_path = '/Users/alexwang/Desktop/CS224NLego/lego_instructions/6462457.png'

# image_paths = ["/Users/alexwang/Desktop/CS224NLego/lego_instructions/6128668.png/3faca61e7dc14080a4006e3deea3d9dedYSwG78XsOHJS4MW-6.png"]
image_paths = []


def extract_number_after_dash(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r'-(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')  # Use 'inf' to handle filenames without the pattern

for file_name in os.listdir(folder_path):
    # Check if it is a file (not a directory)
    if os.path.isfile(os.path.join(folder_path, file_name)):
        image_paths.append(os.path.join(folder_path, file_name))

image_paths = sorted(image_paths, key=extract_number_after_dash)

results = []

for image_path in image_paths:
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the instructions within the image for someone that is blind. \
                            Identify each step, describing the necessary pieces with color, shape, and size needed in that step. \
                            Each brick should be described by its dimensions and carefully examined piece by piece. Match these pieces to actual lego pieces and their ID numbers. \
                            Then describe where it should be placed in the context of the previous pieces given."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Append the result to the list
    data = response.json()

    message_content = data['choices'][0]['message']['content']

    print(message_content)

    results.append(message_content)


output_file_path = 'lego_instructions_6462457.txt'
with open(output_file_path, 'w') as file:
    for result in results:
        file.write(result + "\n\n")  # Write each result as a paragraph with a newline in between

print(f"Instructions have been exported to {output_file_path}")