import requests
import matplotlib.pyplot as plt

# URL of the dataset
url = "https://datasets-server.huggingface.co/rows?dataset=calvinlaughlin%2Flegobuilds-training&config=default&split=train&offset=0&length=100"

# Fetching the dataset
response = requests.get(url)
data = response.json()

# Extract lengths from the dataset
lengths = [len(row['row']['text']) for row in data['rows']]

# Plotting the lengths
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Text Lengths in LEGO Builds Training Dataset')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()