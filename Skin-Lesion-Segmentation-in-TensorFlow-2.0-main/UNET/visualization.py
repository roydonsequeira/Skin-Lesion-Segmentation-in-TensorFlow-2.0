import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from CSV
df = pd.read_csv('files/score.csv')

# Extract relevant columns (assuming 'Image Name' is not a metric)
metrics = ['Acc', 'F1', 'Jaccard', 'Recall', 'Precision']

# Define your custom color palette
custom_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

# Create the pairwise scatter plot
sns.pairplot(df[metrics], diag_kind="kde", palette=sns.color_palette(custom_colors))  # Use KDE plots for diagonals

# Customize the plot (optional)
plt.suptitle("Pairwise Scatter Plots for Metrics")  # Add a title
plt.subplots_adjust(top=0.92)  # Adjust spacing to avoid title cutoff
plt.show()
