import pandas as pd
import os

def merge_and_compare_csv(file1_path, file2_path):
    # Load the CSV files
    lm2_df = pd.read_csv(file1_path)
    gt_df = pd.read_csv(file2_path)

    # Merge the two dataframes on matching 'filename' and 'leaf_id' columns
    merged_df = pd.merge(lm2_df, gt_df, left_on='filename', right_on='leaf_id', how='inner')

    # Calculate error as the absolute difference between width_pixels and pixel_distance
    merged_df['error_pixels'] = (merged_df['width_pixels'] - merged_df['pixel_distance']).abs()
    merged_df['error_percent'] = (merged_df['error_pixels'] / merged_df['pixel_distance'].replace(0, pd.NA)) * 100

    # Save to the parent directory of file1_path
    output_dir = os.path.dirname(os.path.dirname(file1_path))
    output_path = os.path.join(output_dir, 'GT_comparison.csv')
    merged_df.to_csv(output_path, index=False)

    print(f"Comparison saved to {output_path}")

# Define the paths to the input files
file1_path = 'D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/Petiole_Data/petiole_widths_groundtruth_comparision_LM2.csv' # 'path/to/LM2.csv'  # Update this path
file2_path = 'D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/petiole_widths_groundtruth.csv' # 'path/to/GT.csv'   # Update this path

# Run the function
merge_and_compare_csv(file1_path, file2_path)
