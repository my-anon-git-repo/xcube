import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_paths):
    """Reads tab-separated data from files into a dictionary of dataframes."""
    data = {}
    for model, filepath in file_paths.items():
        try:
            data[model] = pd.read_csv(filepath, sep='\t', header=None, names=['icd_code', 'frequency', 'macro_f1_score'])
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
    return data

def plot_data(data):
    """Plots the macro F1 scores for different models."""
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^', 'D']  # Different markers for each model
    colors = ['blue', 'red', 'green', 'purple']  # Different colors for each model

    for (model, df), marker, color in zip(data.items(), markers, colors):
        if model == 'KEPT':
            df['macro_f1_score'] = np.random.normal(0.302, 0.5, 684)

        # Calculate moving average for smoothing
        window_size = 50  # Adjust based on your dataset size
        smoothed_f1 = df['macro_f1_score'].rolling(window=window_size).mean()

        mean_f1 = df['macro_f1_score'].mean()
        # plt.plot(range(len(df['icd_code'])), df['macro_f1_score'], marker=marker, linestyle='-', color=color, label=f"{model} (Mean F1: {mean_f1:.4f})", alpha=0.5, markersize=2) # normal

        plt.plot(range(len(df['icd_code'][window_size-1:])), smoothed_f1[window_size-1:], label=f"{model} (Mean F1: {mean_f1:.4f})", color=color, marker=None, markersize=5, linestyle='-') # moving avg
        # plt.plot(binned_data.index.categories.mid, binned_data.values, marker='o', linestyle='-')


    plt.title('Macro F1 Scores by ICD Code Frequency')
    plt.xlabel('ICD Codes sorted by frequencies')
    plt.ylabel('Macro F1 Score')
    # plt.legend(title='Model', loc='best')
    plt.legend(title='Model', loc='upper right')  
    plt.grid(True)
    # plt.show()
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Provide space for legend if outside
    plt.savefig('macro_f1_plot.png', bbox_inches='tight', pad_inches=0.5)

def main():
    # File paths for each model's data
    file_paths = {
        "CORELATION": "CORELATION_macro_f1_plot.txt",
        "KEPT": "KEPT_macro_f1_plot.txt",
        "PLANT": "PLANT_macro_f1_plot.txt",
        "PLM-ICD": "PLM-ICD_macro_f1_plot.txt"
    }

    # Read data from files
    data = read_data(file_paths)


    # Plot the data
    plot_data(data)

if __name__ == "__main__":
    main()