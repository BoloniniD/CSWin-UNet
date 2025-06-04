import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_comparison(original_path, blurred_path):
    with h5py.File(original_path, 'r') as f_orig, h5py.File(blurred_path, 'r') as f_blur:
        orig_ds = list(f_orig.keys())[0]
        blur_ds = list(f_blur.keys())[0]

        original = f_orig[orig_ds][:]
        blurred = f_blur[blur_ds][:]

    # Print data statistics for debugging
    print(f"\nOriginal data: {original.shape} {original.dtype} | Min: {original.min()} Max: {original.max()}")
    print(f"Blurred data: {blurred.shape} {blurred.dtype} | Min: {blurred.min()} Max: {blurred.max()}")

    # Auto-adjust window levels if needed
    if original.dtype == np.float32 or original.dtype == np.float64:
        # Assuming normalized data between 0-1
        vmin, vmax = 0.0, 1.0
    else:
        # CT Hounsfield Units default window
        vmin, vmax = -1000, 2000  # Adjust these based on print output

    # Create figure with adjustable windowing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Original vs Blurred [vmin={vmin}, vmax={vmax}]")

    im1 = ax1.imshow(original[original.shape[0]//2], cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title("Original")

    im2 = ax2.imshow(blurred[blurred.shape[0]//2], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title("Blurred")

    # Add windowing sliders
    axcolor = 'lightgoldenrodyellow'
    ax_vmin = plt.axes([0.2, 0.06, 0.6, 0.03], facecolor=axcolor)
    ax_vmax = plt.axes([0.2, 0.01, 0.6, 0.03], facecolor=axcolor)

    vmin_slider = Slider(ax_vmin, 'Min', original.min(), original.max(), valinit=vmin)
    vmax_slider = Slider(ax_vmax, 'Max', original.min(), original.max(), valinit=vmax)

    def update_window(val):
        im1.set_clim(vmin_slider.val, vmax_slider.val)
        im2.set_clim(vmin_slider.val, vmax_slider.val)
        fig.canvas.draw_idle()

    vmin_slider.on_changed(update_window)
    vmax_slider.on_changed(update_window)

    # Show middle slice by default
    slice_idx = original.shape[0] // 2

    # Add slice slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slice_slider = Slider(
        ax=ax_slider,
        label='Slice',
        valmin=0,
        valmax=original.shape[0]-1,
        valinit=slice_idx,
        valstep=1
    )

    def update(val):
        idx = int(slice_slider.val)
        im1.set_data(original[idx])
        im2.set_data(blurred[idx])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()

def visualize_random_samples(original_dir, blurred_dir, num_samples=3):
    # Get matching file pairs
    files = [f for f in os.listdir(original_dir) if f.endswith('.h5')]
    samples = np.random.choice(files, min(num_samples, len(files)), replace=False)

    for fname in samples:
        orig_path = os.path.join(original_dir, fname)
        blur_path = os.path.join(blurred_dir, fname)

        if not os.path.exists(blur_path):
            print(f"Blurred version missing for {fname}")
            continue

        try:
            print(f"Visualizing {fname}")
            visualize_comparison(orig_path, blur_path)
        except Exception as e:
            print(f"Error visualizing {fname}: {str(e)}")
            with h5py.File(orig_path, 'r') as f:
                print(f"Available datasets in original: {list(f.keys())}")
            with h5py.File(blur_path, 'r') as f:
                print(f"Available datasets in blurred: {list(f.keys())}")

# Example usage
if __name__ == "__main__":
    input_directory = "./datasets/Synapse/test_vol_h5/"
    output_directory = "./datasets/Synapse_blurred/test_vol_h5/"

    # Visualize 3 random samples
    visualize_random_samples(input_directory, output_directory, num_samples=3)