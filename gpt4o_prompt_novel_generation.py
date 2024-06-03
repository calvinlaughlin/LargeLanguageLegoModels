# from tqdm import tqdm
# from PIL import Image
# import requests
# from io import BytesIO
# import cv2
# from google.colab.patches import cv2_imshow

# from docx import Document
# from docx.shared import Inches, Pt
# from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

# import subprocess

# from langchain import PromptTemplate, LLMChain
# from langchain.llms import Replicate, OpenAI


import base64
import requests
import os

api_key = "sk-proj-HMzMuOfGEJwmH58C8p2AT3BlbkFJOnzsQdIok4MngMnPNE4M"

topic_text = input("describe what you'd like to build: \n") # take in the input from user of what they want

print(topic_text)

template = f"""You are a designer that is creating a novel lego set with the topic {topic_text}. 
You will give step by step written instructions for the lego set that culminate in a complete lego build.
Make sure that each step is doable in the real world and that each piece is a real lego piece, include the part number.
Firstly, state every lego piece that you will need in the build.
Then, create a small story about the build.
Finally, create the comprehensive step by step guide of the builds.
"""

print(template)

# prompt = PromptTemplate(template=(template), input_variables=["topic"])

# llm = OpenAI(temperature=0.7, model="gpt-4o", max_tokens=2000) #.7 for now but can change the temperature as much as we want
# llm_chain = LLMChain(prompt=prompt, llm=llm)


# result = llm_chain.run(
#     topic_text
# )

# with open("lego.txt", 'w+') as file: #checkpoint
#     file.write(result)

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
                    "text": template
                }
            ]
        }
    ],
    "max_tokens": 3000
}

# Make the API request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Append the result to the list
data = response.json()

message_content = data['choices'][0]['message']['content']

print(message_content)

# Write the results to a text file
output_file_path = 'novel_lego_0.txt'
with open(output_file_path, 'w') as file:
    file.write(message_content)  # Write each result as a paragraph with a newline in between