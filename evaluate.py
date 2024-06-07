import requests
import os

api_key = "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"

folder_path = '/Users/alexwang/Desktop/CS224NLego/lego_instructions/text_files'
text_file_paths = []

for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)) and file_name.endswith('.txt'):
        text_file_paths.append(os.path.join(folder_path, file_name))

evaluation_criteria = """
Criteria for Evaluating LEGO Instruction Sets:
Clarity (15 points)
15: Instructions are exceptionally clear, easy to understand, and free from ambiguity.
12-14: Instructions are clear and mostly easy to understand with minor ambiguities.
9-11: Instructions are generally clear but may require rereading to understand.
5-8: Instructions are often unclear and confusing, making it hard to follow steps.
1-4: Instructions are very unclear, with significant ambiguity and confusion.
0: Instructions are incomprehensible or missing.

Completeness (15 points)
15: Includes every detail necessary for assembly without referring back to the visual manual.
12-14: Includes most details necessary for assembly, with only minor omissions.
9-11: Includes many necessary details but has some noticeable gaps.
5-8: Includes several necessary details, but major gaps make assembly difficult.
1-4: Includes few necessary details, making assembly nearly impossible.
0: Essential details are completely missing.

Accuracy (40 points)
40: Accurately describes all parts, their placements, and assembly steps without errors.
32-39: Accurately describes most parts and steps, with only minor errors.
24-31: Describes parts and steps accurately but has some errors.
16-23: Descriptions of parts and steps contain significant errors.
8-15: Descriptions are mostly inaccurate, leading to incorrect assembly.
0-7: Descriptions are completely inaccurate or missing.

Accessibility (15 points)
15: Written in a way that is easily understood by individuals with no visual references.
12-14: Mostly accessible, with only minor points that might need clarification for those with no visual aid.
9-11: Moderately accessible, but several points may be confusing without visual references.
5-8: Limited accessibility, with many points unclear without visual aid.
1-4: Poorly accessible, making it very difficult to follow without visual references.
0: Not accessible at all for individuals with visual impairments.

Organization (15 points)
15: Logically organized with a clear, consistent structure that guides the reader through the steps.
12-14: Well-organized, with only minor inconsistencies or jumps in the structure.
9-11: Generally organized, but some parts may be out of order or structured inconsistently.
5-8: Poorly organized, with many steps out of order or structured inconsistently.
1-4: Very poorly organized, making it hard to follow the sequence of steps.
0: No recognizable organization; steps are random or missing.

The output should be just Total Score Calculation:
Clarity: /15
Completeness: /15
Accuracy: /40
Accessibility: /15
Organization: /15
Total Score: /100
"""

def evaluate_instructions_with_gpt4o(content):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"Evaluate the following LEGO instruction set based on these criteria: {evaluation_criteria}\n\nInstruction set:\n{content}"
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    data = response.json()

    if 'choices' in data:
        return data['choices'][0]['message']['content']
    else:
        print(f"Error: {data}")
        return None

def split_text_into_chunks(text, max_chunk_size=100000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

results = []

for text_file_path in text_file_paths:
    with open(text_file_path, 'r') as file:
        content = file.read()

    chunks = split_text_into_chunks(content)
    chunk_evaluations = []

    for chunk in chunks:
        num = 0
        evaluation = evaluate_instructions_with_gpt4o(chunk)
        if evaluation:
            chunk_evaluations.append(evaluation)
            print(f"{num}/{len(chunks)}")
    
    combined_evaluation = "\n".join(chunk_evaluations)
    
    print(text_file_path)
    print(combined_evaluation)
    results.append({
        "file": text_file_path,
        "evaluation": combined_evaluation
    })

output_file_path = 'gpt4o_vision_evaluation.txt'
with open(output_file_path, 'w') as file:
    for result in results:
        file.write(f"File: {result['file']}\n")
        file.write(f"Evaluation:\n{result['evaluation']}\n")
        file.write("\n")

print(f"Evaluation has been exported to {output_file_path}")
