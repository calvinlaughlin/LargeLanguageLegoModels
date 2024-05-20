import os
import json
import re

# Directory containing your files
input_dir = 'uncleaned_text'
output_file = 'processed_instructions.json'

# Function to clean and preprocess text
def preprocess_text(text):
    # Remove unnecessary whitespace and fix common typos
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('symm', 'symmetrically')
    text = text.replace('ver', 'vertically')
    text = text.replace('hor', 'horizontally')
    # Add more replacements as necessary
    return text

# Function to extract section using regex
def extract_section(pattern, content):
    match = re.search(pattern, content, re.DOTALL)
    return preprocess_text(match.group(0)) if match else ""

# Function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract sections using regex
    introduction = extract_section(r'Welcome.*?\.', content)
    terminology = extract_section(r'(Terms we’ll be using:|Before you start building, here are some terms we’ll be using:).*?(?=Instructions:|Next, we\'ll build|Open Bag 1|When you\'re done building)', content)
    instructions_match = re.search(r'(Instructions:|Next, we\'ll build|Open Bag 1).*?(Abbreviation definitions:|Visit|Enjoy|Thank you)', content, re.DOTALL)
    abbreviations_match = re.search(r'Abbreviation definitions:.*', content, re.DOTALL)

    instructions = preprocess_text(instructions_match.group(0)) if instructions_match else ""
    abbreviations = preprocess_text(abbreviations_match.group(0)) if abbreviations_match else ""

    # Convert sections into structured format
    data = {
        'introduction': introduction,
        'terminology': terminology,
        'instructions': instructions,
        'abbreviations': abbreviations
    }

    return data

# Process all files in the directory
all_data = []
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):  # Adjust extension if necessary
        file_path = os.path.join(input_dir, filename)
        file_data = process_file(file_path)
        all_data.append(file_data)

# Save all processed data to a JSON file
with open(output_file, 'w') as json_file:
    json.dump(all_data, json_file, indent=4)

print(f"Processed data saved to {output_file}")
