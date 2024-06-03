from transformers import pipeline
from PIL import Image
import os
import re

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id)

# Path to the folder containing images
folder_path = '/Users/alexwang/Desktop/CS224NLego/lego_instructions/'

# List to store file names
image_paths = []

# Function to extract the numerical value following the '-' character
def extract_number_after_dash(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r'-(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')  # Use 'inf' to handle filenames without the pattern

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if it is a file (not a directory)
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        image_paths.append(file_path)

# Sort the image paths based on the extracted number after the dash
# image_paths = sorted(image_paths, key=extract_number_after_dash)
image_paths = ["/Users/alexwang/Desktop/CS224NLego/lego_instructions/6127058.png/11768e25291345d3d8c6a2e9dd7820d1s1eVdrTTHL94JH0X-2.png"]


# Initialize an empty list to store results
results = []

# Iterate through each image path
for image_path in image_paths:
    image = Image.open(image_path)
    prompt = (
        "USER: <image>\nDescribe the instructions within the image for someone that is blind. "
        "Identify each step, describing the necessary pieces with color, shape, and size needed in that step. "
        "Each brick should be described by its dimensions and carefully examined piece by piece. "
        "Match these pieces to actual lego pieces and their ID numbers. "
        "Then describe where it should be placed in the context of the previous pieces given.\nASSISTANT:"
    )

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 500})
    result_text = outputs[0]['generated_text'] if isinstance(outputs, list) else outputs['generated_text']
    results.append(result_text)

# Write the results to a text file
output_file_path = 'llava_instruction_.txt'
with open(output_file_path, 'w') as file:
    for result in results:
        file.write(result + "\n\n")  # Write each result as a paragraph with a newline in between

print(f"Instructions have been exported to {output_file_path}")
