import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def inspect_npz_file(filepath):
    """Completely analyze an .npz file's structure and contents"""
    print(f"\n{'='*50}\nInspecting: {os.path.basename(filepath)}\n{'='*50}")

    try:
        # Load the npz file
        data = np.load(filepath, allow_pickle=True)

        # 1. Show all available keys
        print("\n[1] All keys in file:")
        keys = list(data.keys())
        print(keys)

        # 2. Detailed inspection of each array
        for key in keys:
            print(f"\n[2] Contents of '{key}':")
            arr = data[key]

            # Basic info
            print(f"Shape: {arr.shape}")
            print(f"Data type: {arr.dtype}")
            print(f"Min value: {np.min(arr)}")
            print(f"Max value: {np.max(arr)}")

            # For small arrays, print actual values
            if arr.size < 100:
                print("Values:")
                print(arr)

            # Special handling for potential label arrays
            if arr.ndim in [2, 3] and arr.dtype in [np.int8, np.int16, np.int32, np.int64]:
                unique_vals = np.unique(arr)
                print(f"\n[3] Label Analysis for '{key}':")
                print(f"Unique values: {unique_vals}")
                print(f"Value counts: {dict(zip(*np.unique(arr, return_counts=True)))}")

                # Visualize if it might be a label
                plt.figure(figsize=(12, 6))
                plt.imshow(arr.squeeze(), cmap='jet')
                plt.colorbar()
                plt.title(f"Visualization of '{key}' (cmap='jet')")
                plt.savefig(f"{os.path.basename(filepath)}_{key}_visualization.png")
                plt.close()
                print(f"Saved visualization of '{key}' to disk")

        # 3. Check for nested structures
        print("\n[4] Checking for nested structures:")
        for key in keys:
            arr = data[key]
            if arr.dtype == object:
                print(f"\n'{key}' contains Python objects:")
                print(f"First element type: {type(arr.item(0))}")
                if isinstance(arr.item(0), dict):
                    print("First element keys:", list(arr.item(0).keys()))

        # 4. Save full file info to text file
        with open(f"{os.path.basename(filepath)}_inspection.txt", "w") as f:
            f.write(f"NPZ File Inspection Report\n{'='*30}\n")
            f.write(f"File: {filepath}\n\n")

            for key in keys:
                arr = data[key]
                f.write(f"\nKey: {key}\n")
                f.write(f"Shape: {arr.shape}\n")
                f.write(f"DataType: {arr.dtype}\n")
                f.write(f"Min/Max: {np.min(arr)}/{np.max(arr)}\n")

                if arr.size < 1000:  # Don't dump huge arrays
                    np.savetxt(f, arr.squeeze(), fmt='%s')

        print(f"\nSaved full inspection report to {os.path.basename(filepath)}_inspection.txt")

    except Exception as e:
        print(f"Error inspecting file: {str(e)}")

def process_directory(input_dir, num_samples=3):
    """Inspect multiple files in a directory"""
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    print(f"\nFound {len(npz_files)} .npz files. Inspecting first {num_samples}...")

    for filename in npz_files[:num_samples]:
        filepath = os.path.join(input_dir, filename)
        inspect_npz_file(filepath)

if __name__ == "__main__":
    INPUT_DIR = "./datasets/Synapse/train_npz"

    print("NPZ File Structure Inspector")
    print(f"Scanning directory: {os.path.abspath(INPUT_DIR)}")

    process_directory(INPUT_DIR)

    print("\nInspection complete. Check generated files for details:")
    print("- .png files for visualizations")
    print("- .txt files for full data reports")
