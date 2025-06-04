import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt # Added for saving images

def normalize_image(img):
    """ Normalizes image data to 0-1 range for visualization. """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        # Normalize to 0-1
        img_normalized = (img - min_val) / (max_val - min_val)
    else:
        # Handle constant image (avoid division by zero)
        img_normalized = np.zeros_like(img)
    # Optional: Clip extreme values if needed for CT scans before normalization
    # For example: img = np.clip(img, -1000, 1000)
    return img_normalized

def apply_blur_and_save_samples(input_dir, output_dir, sample_output_dir, sigma=1.0, num_samples=10, image_dataset_name='image', label_dataset_name='label'):
    """
    Applies Gaussian blur to 3D CT volumes in HDF5 files, saves blurred HDF5,
    and saves image samples (original, blurred, label) for a subset of cases.

    Args:
        input_dir (str): Input directory containing .h5 files (e.g., *.npy.h5).
        output_dir (str): Output directory for blurred HDF5 files.
        sample_output_dir (str): Output directory for PNG image samples.
        sigma (float): Standard deviation for Gaussian kernel.
        num_samples (int): Number of sample cases to save images from.
        image_dataset_name (str): Name of the image dataset within HDF5 files.
        label_dataset_name (str): Name of the label dataset within HDF5 files.
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sample_output_dir, exist_ok=True)

    samples_saved = 0
    processed_files = sorted(os.listdir(input_dir)) # Process in a consistent order

    # Process each file in input directory
    for filename in processed_files:
        # Adjust the check if your files have a different pattern like '.npy.h5'
        if not filename.endswith('.h5'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        # Extract a base name for sample images (remove extensions)
        case_name = filename.replace('.npy.h5', '').replace('.h5', '')

        try:
            with h5py.File(input_path, 'r') as f_in:
                # Check if required datasets exist before proceeding
                if image_dataset_name not in f_in:
                    print(f"Warning: Skipping {filename}, missing '{image_dataset_name}' dataset.")
                    continue
                if label_dataset_name not in f_in:
                    print(f"Warning: Skipping {filename}, missing '{label_dataset_name}' dataset.")
                    continue

                with h5py.File(output_path, 'w') as f_out:
                    # Copy all attributes first
                    for key in f_in.attrs:
                        f_out.attrs[key] = f_in.attrs[key]

                    original_image_data = None
                    blurred_image_data = None
                    label_data = None

                    # Process each dataset in the input file
                    for dataset_name in f_in:
                        data = f_in[dataset_name][:]

                        if dataset_name == image_dataset_name:
                            # Apply Gaussian blur slice by slice (axis 0 is depth)
                            print(f"Applying blur to '{dataset_name}' in {filename}...")
                            blurred_data = np.zeros_like(data)
                            for i in range(data.shape[0]):
                                blurred_data[i] = gaussian_filter(data[i], sigma=sigma)

                            # Store data for potential sampling
                            original_image_data = data
                            blurred_image_data = blurred_data

                            # Create blurred dataset in output HDF5
                            f_out.create_dataset(
                                dataset_name,
                                data=blurred_data,
                                compression=f_in[dataset_name].compression,
                                chunks=f_in[dataset_name].chunks
                            )
                        elif dataset_name == label_dataset_name:
                             # Store label data for potential sampling
                            label_data = data
                            # Copy label data unchanged to output HDF5
                            print(f"Copying '{dataset_name}' in {filename}...")
                            f_out.create_dataset(
                                dataset_name,
                                data=data,
                                compression=f_in[dataset_name].compression,
                                chunks=f_in[dataset_name].chunks
                            )
                        else:
                            # Copy any other datasets unchanged
                            print(f"Copying '{dataset_name}' in {filename}...")
                            f_out.create_dataset(
                                dataset_name,
                                data=data,
                                compression=f_in[dataset_name].compression,
                                chunks=f_in[dataset_name].chunks
                            )

                    # --- Save Image Samples ---
                    # Check if we still need to save samples and if data was loaded
                    if samples_saved < num_samples and original_image_data is not None and label_data is not None:
                        print(f"Saving image sample for {filename}...")
                        # Select a representative slice (e.g., middle one)
                        slice_idx = original_image_data.shape[0] // 2

                        # Extract the 2D slices
                        original_slice = original_image_data[slice_idx]
                        blurred_slice = blurred_image_data[slice_idx]
                        label_slice = label_data[slice_idx]

                        # Normalize images for saving as PNG
                        norm_original_slice = normalize_image(original_slice)
                        norm_blurred_slice = normalize_image(blurred_slice)

                        # Define output paths for the sample images
                        sample_basename = f"{case_name}_slice_{slice_idx}"
                        original_img_path = os.path.join(sample_output_dir, f"{sample_basename}_original_image.png")
                        blurred_img_path = os.path.join(sample_output_dir, f"{sample_basename}_blurred_image.png")
                        label_img_path = os.path.join(sample_output_dir, f"{sample_basename}_label.png")

                        # Save the images using matplotlib
                        # Use 'gray' colormap for medical images
                        # Use 'viridis' or 'jet' or specific mapping for labels if they are multi-class
                        # Use 'gray' for labels if they are binary (0/1)
                        plt.imsave(original_img_path, norm_original_slice, cmap='gray')
                        plt.imsave(blurred_img_path, norm_blurred_slice, cmap='gray')
                        # Adjust cmap for labels as needed. 'viridis' often shows integer labels well.
                        plt.imsave(label_img_path, label_slice, cmap='viridis')

                        print(f"Saved sample images to {sample_output_dir} for slice {slice_idx} of {filename}")
                        samples_saved += 1
                    # --- End Save Image Samples ---

            print(f"Processed HDF5: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print(f"\nFinished processing. Saved {samples_saved} image samples to {sample_output_dir}")

# Example usage
if __name__ == "__main__":
    # --- Configuration ---
    # !! Adjust these paths and names according to your setup !!
    input_directory = "./datasets/Synapse/test_vol_h5/"
    output_directory_blurred_h5 = "./datasets/Synapse_blurred/test_vol_h5/"
    output_directory_samples = "./datasets/Synapse_blurred/samples/" # Directory for PNG samples
    blur_sigma = 1.5
    num_image_samples = 10
    # !! Crucial: Ensure these dataset names match your HDF5 structure !!
    image_dset_name = 'image'
    label_dset_name = 'label'
    # --- End Configuration ---

    print("Starting script...")
    print(f"Input HDF5 directory: {input_directory}")
    print(f"Output blurred HDF5 directory: {output_directory_blurred_h5}")
    print(f"Output samples directory: {output_directory_samples}")
    print(f"Blur Sigma: {blur_sigma}")
    print(f"Number of samples to save: {num_image_samples}")
    print(f"Image dataset name in HDF5: '{image_dset_name}'")
    print(f"Label dataset name in HDF5: '{label_dset_name}'")

    apply_blur_and_save_samples(
        input_dir=input_directory,
        output_dir=output_directory_blurred_h5,
        sample_output_dir=output_directory_samples,
        sigma=blur_sigma,
        num_samples=num_image_samples,
        image_dataset_name=image_dset_name,
        label_dataset_name=label_dset_name
    )

    print("Script finished.")
