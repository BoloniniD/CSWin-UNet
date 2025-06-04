import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import math

# Configuration
INPUT_DIR = "./datasets/Synapse/train_npz" # Use the correctly processed NPZ files as input
OUTPUT_DIR = "./datasets/Synapse_blurred/train_npz"
VISUALIZATION_DIR = "./preprocessing_visualization_blurred" # Separate viz for blurred
LIST_OUTPUT_DIR = "./lists/lists_Synapse_blurred" # Directory for the train.txt file
SIGMA = 1.0
VISUALIZE_SAMPLES = 5 # Reduced samples for faster run, adjust as needed
MAX_ATTEMPTS = 100

def is_valid_label(label_array):
    """Check if label array contains meaningful segmentation data"""
    if label_array is None:
        return False
    unique_vals = np.unique(label_array)
    return len(unique_vals) > 1

def find_valid_samples(input_dir, npz_files, num_samples=3):
    """Find files that contain valid labels for visualization"""
    valid_samples = []
    attempts = 0
    print(f"Searching for {num_samples} valid samples (max attempts: {MAX_ATTEMPTS})...")
    # Shuffle to get potentially different samples each run
    np.random.shuffle(npz_files)
    for filename in npz_files:
        if len(valid_samples) >= num_samples or attempts >= MAX_ATTEMPTS:
            break

        input_path = os.path.join(input_dir, filename)
        try:
            # Use allow_pickle=True just in case, though unlikely needed now
            with np.load(input_path, allow_pickle=True) as data:
                label = None
                # Check standard key first
                if 'label' in data:
                    label = data['label']
                else: # Check alternatives if needed (less likely with corrected data)
                    for key in ['labels', 'segmentation', 'mask', 'ground_truth']:
                        if key in data:
                            label = data[key]
                            break

                if is_valid_label(label):
                    valid_samples.append(filename)
                    # print(f"Found valid sample: {filename}") # Less verbose

        except Exception as e:
            print(f"Error checking {filename}: {str(e)}")
        attempts += 1

    if not valid_samples:
        print(f"Warning: Could not find {num_samples} files with valid labels in first {MAX_ATTEMPTS} files. Proceeding with {len(valid_samples)} samples.")
    elif len(valid_samples) < num_samples:
         print(f"Warning: Found only {len(valid_samples)} valid samples out of {num_samples} requested.")

    return valid_samples

def debug_labels(label_array, filename):
    """Examine label contents and return unique non-zero labels"""
    # print(f"\nDebugging labels for {filename}:") # Less verbose
    # print(f"Shape: {label_array.shape}, Data type: {label_array.dtype}")
    # print(f"Min: {np.min(label_array)}, Max: {np.max(label_array)}")
    unique_labels = np.unique(label_array)
    # print(f"Unique values: {unique_labels}")
    return unique_labels[unique_labels != 0]

def save_sample_comparison(original, blurred, label, unique_labels, filename, sigma):
    """Save visualization including individual label masks"""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    if original.ndim == 3 and original.shape[0] in [1, 3]: # CHW
        original_vis = original.transpose(1, 2, 0).squeeze()
        blurred_vis = blurred.transpose(1, 2, 0).squeeze()
    else: # Assume HWC or HW
        original_vis = original.squeeze()
        blurred_vis = blurred.squeeze()
    label_vis = label.squeeze()

    num_labels_to_show = len(unique_labels)
    num_base_plots = 3
    total_plots = num_base_plots + num_labels_to_show
    cols = math.ceil(math.sqrt(total_plots))
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    # Plot Base Images
    axes[0].imshow(original_vis, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(blurred_vis, cmap='gray')
    axes[1].set_title(f"Blurred (σ={sigma})")
    axes[1].axis('off')
    im = axes[2].imshow(label_vis, cmap='jet', vmin=0, vmax=np.max(unique_labels) if len(unique_labels)>0 else 1)
    axes[2].set_title("Combined Labels")
    axes[2].axis('off')

    # Plot Individual Label Masks
    for i, label_id in enumerate(unique_labels):
        ax_idx = num_base_plots + i
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            binary_mask = (label_vis == label_id).astype(np.uint8)
            ax.imshow(binary_mask, cmap='gray')
            ax.set_title(f"Label Class {label_id}")
            ax.axis('off')

    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(VISUALIZATION_DIR, f"sample_{filename.replace('.npz', '')}_blurred_detailed.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    # print(f"Saved detailed visualization: {output_path}") # Less verbose
    return output_path

def process_npz_file(input_path, output_path, is_visualization=False):
    """
    Applies blur to the image in an npz file and saves it.
    Returns the base filename if successful and saved, otherwise None.
    """
    base_filename = os.path.basename(input_path).replace('.npz', '')
    try:
        with np.load(input_path) as data:
            # Ensure 'label' key exists and is valid
            if 'label' not in data or not is_valid_label(data['label']):
                if is_visualization: print(f"Skipping {base_filename} for viz: Invalid label")
                return None # Indicate failure or skip

            # Ensure 'image' key exists
            if 'image' not in data:
                 print(f"Skipping {base_filename}: Missing 'image' key")
                 return None

            image = data['image']
            label = data['label'] # Keep the original label

            # Apply Gaussian blur
            # Ensure image is float for blurring, handle potential non-float input
            if not np.issubdtype(image.dtype, np.floating):
                image = image.astype(np.float32)
            blurred_image = gaussian_filter(image, sigma=SIGMA)

            unique_labels = None
            if is_visualization:
                # Get unique labels for visualization function
                unique_labels = debug_labels(label, base_filename + ".npz")
                # Return necessary data for visualization call
                return image, blurred_image, label, unique_labels

            # Save the blurred image and original label to the output path
            if output_path:
                np.savez_compressed(output_path, image=blurred_image.astype(np.float32), label=label)
                return base_filename # Return base filename on successful save

            # Should not happen in batch mode, but return None if no output path
            return None

    except Exception as e:
        print(f"ERROR processing {input_path}: {str(e)}")
        return None # Indicate failure

def process_directory():
    """Applies blur, creates visualizations, and generates train.txt list."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(LIST_OUTPUT_DIR, exist_ok=True) # Create list directory

    all_npz_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npz')]
    if not all_npz_files:
        print(f"Error: No .npz files found in {INPUT_DIR}")
        return

    processed_filenames = [] # List to store names for train.txt
    skipped_files = 0

    # --- Visualization Phase ---
    print("\n--- Visualization Phase ---")
    # Find valid samples from the *input* directory
    viz_sample_filenames = find_valid_samples(INPUT_DIR, all_npz_files, VISUALIZE_SAMPLES)
    viz_processed_count = 0
    for filename in viz_sample_filenames:
        input_path = os.path.join(INPUT_DIR, filename)
        # Call process_npz_file in visualization mode
        viz_data = process_npz_file(input_path, None, is_visualization=True)

        if viz_data: # Check if it returned valid data tuple
            original, blurred, label, unique_labels = viz_data
            save_sample_comparison(original, blurred, label, unique_labels, filename, SIGMA)
            viz_processed_count += 1
        else:
            skipped_files += 1
            print(f"Skipped {filename} during detailed visualization check.")
    print(f"Visualization phase complete. {viz_processed_count} samples visualized.")

    # --- Batch Processing Phase ---
    print("\n--- Batch Processing Phase ---")
    # Process all files, including those visualized (to ensure they are saved blurred)
    for filename in tqdm(all_npz_files, desc="Applying blur"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename) # Save to blurred directory

        # Call process_npz_file in processing mode
        saved_base_filename = process_npz_file(input_path, output_path, is_visualization=False)

        if saved_base_filename:
            processed_filenames.append(saved_base_filename)
        # No else needed here, skipping is handled implicitly if None is returned

    # Calculate skipped count accurately
    total_files = len(all_npz_files)
    successfully_processed_count = len(processed_filenames)
    skipped_files = total_files - successfully_processed_count # More accurate count

    # --- Generate train.txt ---
    list_file_path = os.path.join(LIST_OUTPUT_DIR, "train.txt")
    print(f"\nGenerating training list: {list_file_path}")
    try:
        with open(list_file_path, 'w') as f:
            for name in sorted(processed_filenames): # Sort for consistency
                f.write(name + '\n')
        print(f"Successfully wrote {len(processed_filenames)} entries to train.txt")
    except Exception as e:
        print(f"Error writing train.txt: {e}")


    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"Processing complete.")
    print(f"Total input files checked: {total_files}")
    print(f"Valid files processed and saved: {successfully_processed_count}")
    print(f"Files skipped (invalid/missing labels or errors): {skipped_files}")
    print(f"Blurred results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Detailed visualizations saved to: {os.path.abspath(VISUALIZATION_DIR)}")
    print(f"Training list saved to: {os.path.abspath(list_file_path)}")


if __name__ == "__main__":
    print("Medical Image Blurring Processor with List Generation")
    # Ensure INPUT_DIR points to the *clean* NPZ files from the previous step
    print(f"Input (Clean NPZ): {os.path.abspath(INPUT_DIR)}")
    print(f"Output (Blurred NPZ): {os.path.abspath(OUTPUT_DIR)}")
    print(f"List Output: {os.path.abspath(LIST_OUTPUT_DIR)}")

    process_directory()
