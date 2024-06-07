import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Car', 'Space ship', 'House', 'Boat', 'Castle']
models = [
    "Mistral-7B finetuned", "Mistral-7B not finetuned", "GPT-3.5 finetuned", 
    "GPT-3.5 not finetuned", "Llama-2 finetuned", "Llama-2 not finetuned", "GPT-4o baseline"
]

car_avg = [65.33, 59.00, 36.75, 80.33, 55.67, 86.67, 80.33]
spaceship_avg = [63.67, 73.00, 37.00, 82.16, 36.0, 85.33, 83.33]
house_avg = [54.00, 49.67, 64.5, 80.46, 27.33, 93.66, 85.67]
boat_avg = [62.33, 71.00, 54.0, 79.33, 37.0, 87.33, 87.67]
castle_avg = [73.67, 54.67, 66.67, 77.5, 42.67, 91.33, 80.33]

# Aggregate the data
data = np.array([car_avg, spaceship_avg, house_avg, boat_avg, castle_avg])

# Normalize the data
data = data / 100

# Function to create a radar chart
def create_radar_chart(data, categories, models):
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    for i, model_data in enumerate(data.T):
        values = model_data.tolist()
        values += values[:1]
        ax.plot(angles, values, label=models[i])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('Model Performance Comparison Across Categories')
    plt.show()

create_radar_chart(data, categories, models)
