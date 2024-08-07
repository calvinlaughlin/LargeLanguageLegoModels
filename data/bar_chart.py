import matplotlib.pyplot as plt
import numpy as np

criteria = ['Clarity', 'Completeness', 'Accuracy', 'Accessibility', 'Organization', 'Total Score']
percentages = [79.13, 84.20, 80.53, 68.80, 83.20, 79.50]

x = np.arange(len(criteria))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x, percentages, width, color='skyblue', edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Criteria')
ax.set_ylabel('Scores (%)')
ax.set_title('Scores by Criteria (in %)')
ax.set_xticks(x)
ax.set_xticklabels(criteria, rotation=45, ha="right")

fig.tight_layout()

plt.show()
