import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# The H5 file you want to inspect
H5_FILE_PATH = './datasets/lits17/test_vol_h5/case00002.npy.h5'

# The directory where the output images will be saved
OUTPUT_DIR = './lits17_case00001_slices'
# --- End Configuration ---

def visualize_and_save_slices(h5_path, output_dir):
    """
    Reads an H5 file containing 'image' and 'label' datasets,
    and saves each slice as a side-by-side PNG image.
    """
    # 1. Check if the file exists
    if not os.path.exists(h5_path):
        print(f"Error: File not found at '{h5_path}'")
        print("Please make sure the path is correct and you are running the script from the right directory.")
        return

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved to: '{output_dir}'")

    try:
        # 3. Read the H5 file
        with h5py.File(h5_path, 'r') as hf:
            # Load the entire volumes into memory as NumPy arrays
            image_volume = hf['image'][:]
            label_volume = hf['label'][:]

        # 4. Print information about the loaded data
        print(f"Successfully loaded '{h5_path}'")
        print(f"Image volume shape: {image_volume.shape} | Data type: {image_volume.dtype}")
        print(f"Label volume shape: {label_volume.shape} | Data type: {label_volume.dtype}")

        # CRITICAL: Check the unique values in the label volume.
        # This will tell you if you have more than just background (0).
        unique_labels = np.unique(label_volume)
        print(f"Unique labels found in the volume: {unique_labels}")

        if len(unique_labels) <= 1:
            print("\nWarning: Only one unique label value found. The ground truth may indeed be empty or all background.")

        # 5. Iterate through each slice and save a visualization
        num_slices = image_volume.shape[0]
        print(f"\nGenerating and saving {num_slices} slice images...")

        for i in range(num_slices):
            image_slice = image_volume[i, :, :]
            label_slice = label_volume[i, :, :]

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Case: {os.path.basename(h5_path)} - Slice {i + 1}/{num_slices}', fontsize=16)

            # Display the image slice
            ax1.imshow(image_slice, cmap='gray')
            ax1.set_title('Input Image')
            ax1.axis('off')

            # Display the label slice
            # Using a vibrant colormap like 'nipy_spectral' or 'jet' helps distinguish different integer labels.
            # Setting vmin and vmax ensures consistent coloring across all slices.
            im = ax2.imshow(label_slice, cmap='nipy_spectral', vmin=0, vmax=max(1, unique_labels.max()))
            ax2.set_title('Ground Truth Label')
            ax2.axis('off')

            # Add a colorbar to show what values the colors correspond to
            fig.colorbar(im, ax=ax2, ticks=unique_labels)

            # Save the figure to a file
            output_filename = os.path.join(output_dir, f"slice_{i:03d}.png")
            plt.savefig(output_filename)

            # Close the figure to free up memory
            plt.close(fig)

        print(f"\nDone! All {num_slices} slices have been saved.")

    except FileNotFoundError:
        print(f"Error: The file '{h5_path}' was not found.")
    except KeyError as e:
        print(f"Error: Could not find key {e} in the H5 file.")
        print("Please ensure the H5 file contains 'image' and 'label' datasets.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    visualize_and_save_slices(H5_FILE_PATH, OUTPUT_DIR)
