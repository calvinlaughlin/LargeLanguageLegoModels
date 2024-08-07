"""
data_cleaner.py
Script for manually sorting sections (as delimited by §) into their proper categories.
Not all .txt files contain sections, so manually sorting to ensure correctness.
"""
import json
import os

def categorize_section(section):
    print("\nSection Content:\n")
    print(section[:1000])  # Print the first 1000 characters of the section to avoid long output
    print("\nSelect the category for the section:")
    print("1. Introduction")
    print("2. Terms")
    print("3. Misc")
    print("4. Sorting")
    print("5. Instructions")
    print("6. Ads")
    print("7. Abbreviations")
    choice = input("Enter the number corresponding to the category: ")

    categories = {
        "1": "introduction",
        "2": "terms",
        "3": "misc",
        "4": "sorting",
        "5": "instructions",
        "6": "ads",
        "7": "abbreviations"
    }

    return categories.get(choice, "misc")

def process_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    sections = data.split('§')

    categorized_data = {
        "introduction": [],
        "terms": [],
        "misc": [],
        "sorting": [],
        "instructions": [],
        "ads": [],
        "abbreviations": []
    }

    for section in sections:
        category = categorize_section(section)
        categorized_data[category].append(section.strip())

    return categorized_data

def main():
    directory = input("Enter the path to the directory containing the text files: ")
    all_files_data = {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            print(f"\nProcessing file: {filename}")
            categorized_data = process_file(file_path)
            all_files_data[filename] = categorized_data

    with open("sorted_sections.json", "w") as json_file:
        json.dump(all_files_data, json_file, indent=4)

    print("\nData has been categorized and saved to sorted_sections.json")

if __name__ == "__main__":
    main()
