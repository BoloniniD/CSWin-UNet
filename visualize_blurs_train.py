import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from pathlib import Path

def visualize_slice_comparison(original_path, blurred_path):
    """Visualize comparison between original and blurred NPZ slices"""
    # Load both versions
    with np.load(original_path) as orig_data, np.load(blurred_path) as blur_data:
        original = orig_data['image']
        blurred = blur_data['image']
        label = orig_data['label']  # Not used but loaded for completeness

    # Print data statistics
    print(f"\nOriginal slice: {original.shape} | Min: {original.min():.2f} Max: {original.max():.2f}")
    print(f"Blurred slice: {blurred.shape} | Min: {blurred.min():.2f} Max: {blurred.max():.2f}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Slice Comparison\n{Path(original_path).name}", y=0.95)

    # Initial display parameters
    vmin = min(original.min(), blurred.min())
    vmax = max(original.max(), blurred.max())

    # Show images
    im1 = ax1.imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title("Original")
    ax1.axis('off')

    im2 = ax2.imshow(blurred, cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title("Blurred")
    ax2.axis('off')

    # Add windowing sliders
    axcolor = 'lightgoldenrodyellow'
    ax_vmin = plt.axes([0.2, 0.06, 0.6, 0.03], facecolor=axcolor)
    ax_vmax = plt.axes([0.2, 0.01, 0.6, 0.03], facecolor=axcolor)

    vmin_slider = Slider(ax_vmin, 'Min', vmin, vmax, valinit=vmin)
    vmax_slider = Slider(ax_vmax, 'Max', vmin, vmax, valinit=vmax)

    def update_window(val):
        im1.set_clim(vmin_slider.val, vmax_slider.val)
        im2.set_clim(vmin_slider.val, vmax_slider.val)
        fig.canvas.draw_idle()

    vmin_slider.on_changed(update_window)
    vmax_slider.on_changed(update_window)

    plt.show()

def visualize_random_slices(original_dir, blurred_dir, num_samples=3):
    """Visualize random slices from directory"""
    # Get all NPZ files
    all_files = [f for f in os.listdir(original_dir) if f.endswith('.npz')]

    # Select random samples
    if not all_files:
        print("No NPZ files found in the directory")
        return

    selected_files = random.sample(all_files, min(num_samples, len(all_files)))

    for fname in selected_files:
        orig_path = os.path.join(original_dir, fname)
        blur_path = os.path.join(blurred_dir, fname)

        if not os.path.exists(blur_path):
            print(f"Blurred version missing for {fname}")
            continue

        try:
            print(f"\nVisualizing: {fname}")
            visualize_slice_comparison(orig_path, blur_path)
        except Exception as e:
            print(f"Error visualizing {fname}: {str(e)}")

if __name__ == "__main__":
    # Configuration - modify these paths
    INPUT_DIR = "./datasets/Synapse/train_npz"  # Directory with original .npz files
    OUTPUT_DIR = "./datasets/Synapse_blurred/train_npz"

    # Visualize 3 random slices with comparisons
    visualize_random_slices(INPUT_DIR, OUTPUT_DIR, num_samples=3)
