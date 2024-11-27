import os
import numpy as np
import matplotlib.pyplot as plt
from utils_metrics import calculate_ddr, exponential_decay
from utils_PHATE import load_from_hdf5
from tqdm import tqdm

def test_ddr(directory):

    """
    Process all HDF5 files in a given directory and print the file name
    followed by the decay rate score calculated using the first ECT matrix.

    Parameters:
        directory (str): Path to the directory containing HDF5 files.
    """
    for filename in tqdm(os.listdir(directory), desc="Processing Files", total=len(os.listdir(directory))):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            try:
                # Load data using the provided function
                ect_data, group_labels, shapes, component_names = load_from_hdf5(file_path)
                # Use the first ECT matrix to calculate the decay rate
                ddr_score = calculate_ddr(ect_data[0])
                if np.isnan(ddr_score):
                    print(f"{filename}: {ddr_score}")

                # Print the result
                if ddr_score > 0:
                    print(f"{filename}: {ddr_score}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def plot_multiple_ddr(directory):
    """
    Process HDF5 files in a given directory, calculate DDR, and plot
    the observed decay function along with the fitted curve for up to 5 files.

    Parameters:
        directory (str): Path to the directory containing HDF5 files.
    """
    plt.figure(figsize=(12, 8))
    positive_ddr_count = 0  # Counter for positive DDR scores

    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            try:
                # Load data using the provided function
                ect_data, group_labels, shapes, component_names = load_from_hdf5(file_path)

                # Use the first ECT matrix to calculate the decay rate
                ddr_score = calculate_ddr(ect_data[0])

                # Only plot if DDR is valid and positive
                if not np.isnan(ddr_score) and ddr_score > 0:
                    # Prepare data for plotting
                    unique_values = np.unique(ect_data[0][ect_data[0] > 0])[::-1]  # Descending order
                    densities = [np.sum(ect_data[0] == value) / ect_data[0].size for value in unique_values]
                    levels = np.arange(len(densities))
                    
                    # Fitted curve using the DDR score
                    fitted_curve = exponential_decay(levels, 1, ddr_score)

                    # Plot observed densities and fitted curve
                    plt.scatter(levels, densities, label=f"Observed ({filename})", alpha=0.6)
                    plt.plot(levels, fitted_curve, label=f"Fitted (b={ddr_score:.2f}, {filename})", linestyle='--', alpha=0.8)

                    positive_ddr_count += 1

                    # Stop after processing 5 files with positive DDR scores
                    if positive_ddr_count >= 5:
                        break

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Finalize the plot
    if positive_ddr_count > 0:
        plt.xlabel("Levels (Intensity)")
        plt.ylabel("Density (Normalized)")
        plt.title("Exponential Decay Fit: Observed vs Fitted Densities")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid matrices with positive DDR scores to plot.")


if __name__ == '__main__':
    # directory_path= '/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae/LM2/Data/Measurements/ECT'
    # directory_path= 'C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Data/Measurements/ECT'
    directory_path = '/media/nas/GBIF_Downloads/Cornales/Loasaceae/LM2/Data/Measurements/ECT'
    test_ddr(directory_path)
    # plot_multiple_ddr(directory_path)