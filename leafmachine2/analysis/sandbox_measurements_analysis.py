import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def main():
    # Load the data (assuming a CSV file is provided)
    filename = "C:/Users/Will/Downloads/herbarium_CF_test/LM2_2024_09_17__15-17-36/Data/Measurements/LM2_2024_09_17__15-17-36_MEASUREMENTS_CLEAN.csv"  # Change this if the filename is different
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    # Create a new column for image_area (image_height * image_width)
    df['image_area'] = df['image_height'] * df['image_width']

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Create the scatter plot
    scatter = sns.scatterplot(
        data=df,
        x='image_area',
        y='error',
        hue='herb',  # Color by herb
        palette='Set1',  # Use a color palette (optional, but Set1 works well for categories)
        s=100,  # Size of the points
        alpha=0.8  # Transparency of points
    )

    # Add labels and a title
    plt.title("Scatterplot of Image Area (Height * Width) vs Error", fontsize=14)
    plt.xlabel("Image Area (Height * Width)", fontsize=12)
    plt.ylabel("Error", fontsize=12)

    # Show legend and grid
    plt.legend(title='Herb', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
