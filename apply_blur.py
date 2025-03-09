import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_blur_to_h5(input_dir, output_dir, sigma=1.0):
    """
    Applies Gaussian blur to 3D CT volumes in HDF5 files
    Args:
        input_dir: Input directory containing .npy.h5 files
        output_dir: Output directory for blurred files
        sigma: Standard deviation for Gaussian kernel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.h5'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
                    # Copy all attributes first
                    for key in f_in.attrs:
                        f_out.attrs[key] = f_in.attrs[key]

                    # Process each dataset in the file
                    for dataset_name in f_in:
                        # Read original data
                        data = f_in[dataset_name][:]

                        # Apply Gaussian blur to each slice (axis 0 is depth)
                        blurred_data = np.zeros_like(data)
                        for i in range(data.shape[0]):
                            blurred_data[i] = gaussian_filter(data[i], sigma=sigma)

                        # Create dataset with same properties
                        f_out.create_dataset(
                            dataset_name,
                            data=blurred_data,
                            compression=f_in[dataset_name].compression,
                            chunks=f_in[dataset_name].chunks
                        )

                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_directory = "./datasets/Synapse/test_vol_h5/"
    output_directory = "./datasets/Synapse_blurred/test_vol_h5/"
    apply_blur_to_h5(input_directory, output_directory, sigma=1.5)
