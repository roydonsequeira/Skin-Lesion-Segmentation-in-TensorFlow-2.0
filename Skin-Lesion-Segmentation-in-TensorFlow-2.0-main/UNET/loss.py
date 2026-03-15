import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file_path = "training_losses.csv"  
df = pd.read_csv(csv_file_path)

# Set Seaborn style
sns.set(style="whitegrid")

# Plotting with Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Training Loss', data=df, marker='o', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
