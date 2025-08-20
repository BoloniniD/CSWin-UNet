import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_content):
    """Parse the log content and extract epoch data"""

    # Define the layer order based on your model structure
    layer_order = [
        'stem', 'encoder1', 'merge1', 'encoder2', 'merge2',
        'encoder3', 'merge3', 'encoder4', 'bottleneck',
        'decoder4', 'upsample4', 'concat4', 'decoder3', 'upsample3', 'concat3',
        'decoder2', 'upsample2', 'concat2', 'decoder1', 'upsample1',
        'norm_up', 'output'
    ]

    # Mapping from log names to our layer names
    log_to_layer = {
        'cswin_unet.stage1_conv_embed': 'stem',
        'cswin_unet.stage1': 'encoder1',
        'cswin_unet.merge1': 'merge1',
        'cswin_unet.stage2': 'encoder2',
        'cswin_unet.merge2': 'merge2',
        'cswin_unet.stage3': 'encoder3',
        'cswin_unet.merge3': 'merge3',
        'cswin_unet.stage4': 'encoder4',
        'cswin_unet.norm': 'bottleneck',
        'cswin_unet.stage_up4': 'decoder4',
        'cswin_unet.upsample4': 'upsample4',
        'cswin_unet.concat_linear4': 'concat4',
        'cswin_unet.stage_up3': 'decoder3',
        'cswin_unet.upsample3': 'upsample3',
        'cswin_unet.concat_linear3': 'concat3',
        'cswin_unet.stage_up2': 'decoder2',
        'cswin_unet.upsample2': 'upsample2',
        'cswin_unet.concat_linear2': 'concat2',
        'cswin_unet.stage_up1': 'decoder1',
        'cswin_unet.upsample1': 'upsample1',
        'cswin_unet.norm_up': 'norm_up',
        'cswin_unet.output': 'output'
    }

    epochs = []
    lines = log_content.strip().split('\n')

    current_epoch = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check for start of new epoch (BEFORE TPGM line)
        before_tpgm_match = re.search(r'BEFORE TPGM: Average parameter change from pretrained: ([\d.]+)', line)
        if before_tpgm_match:
            # Save previous epoch if exists
            if current_epoch is not None:
                epochs.append(current_epoch)

            # Start new epoch
            current_epoch = {
                'before_tpgm': float(before_tpgm_match.group(1)),
                'iterations': []
            }
            i += 1
            continue

        # Check for iteration within an epoch
        iteration_match = re.search(r'--- Iteration (\d+)/\d+ ---', line)
        if iteration_match and current_epoch is not None:
            iteration_num = int(iteration_match.group(1))

            # Get global projection ratios (next line)
            i += 1
            global_ratios = {'min': 0, 'max': 0, 'mean': 0}
            if i < len(lines):
                global_line = lines[i].strip()
                global_match = re.search(r'Global Projection Ratios - Min: ([\d.]+), Max: ([\d.]+), Mean: ([\d.]+)', global_line)
                if global_match:
                    global_ratios = {
                        'min': float(global_match.group(1)),
                        'max': float(global_match.group(2)),
                        'mean': float(global_match.group(3))
                    }

            # Get layer ratios
            layer_ratios = {}
            i += 2  # Skip "Mean Ratios per Model Block:" line

            # Read layer data until we hit end markers
            while i < len(lines):
                layer_line = lines[i].strip()
                if layer_line.startswith('---') or layer_line == "" or "Final constraint values" in layer_line:
                    break

                layer_match = re.search(r'-\s+(\S+)\s+:\s+([\d.]+)', layer_line)
                if layer_match:
                    log_layer_name = layer_match.group(1)
                    ratio = float(layer_match.group(2))
                    if log_layer_name in log_to_layer:
                        layer_name = log_to_layer[log_layer_name]
                        layer_ratios[layer_name] = ratio
                i += 1

            current_epoch['iterations'].append({
                'iteration': iteration_num,
                'global_ratios': global_ratios,
                'layer_ratios': layer_ratios
            })
            continue

        i += 1

    # Add the last epoch
    if current_epoch is not None:
        epochs.append(current_epoch)

    return epochs, layer_order

def create_plots(epochs, layer_order):
    """Create three separate plots for the data"""

    if not epochs:
        print("No epochs found in the log!")
        return None

    # Extract data for plotting
    epoch_nums = list(range(1, len(epochs) + 1))
    before_tpgm_values = [epoch['before_tpgm'] for epoch in epochs]

    # For global ratios and layer ratios, use the last iteration of each epoch
    global_mins = []
    global_maxs = []
    global_means = []

    layer_data = {layer: [] for layer in layer_order}

    for epoch in epochs:
        if epoch['iterations']:
            # Use the last iteration of the epoch
            last_iteration = epoch['iterations'][-1]
            global_mins.append(last_iteration['global_ratios']['min'])
            global_maxs.append(last_iteration['global_ratios']['max'])
            global_means.append(last_iteration['global_ratios']['mean'])

            for layer in layer_order:
                if layer in last_iteration['layer_ratios']:
                    layer_data[layer].append(last_iteration['layer_ratios'][layer])
                else:
                    layer_data[layer].append(np.nan)
        else:
            # No iterations in this epoch
            global_mins.append(np.nan)
            global_maxs.append(np.nan)
            global_means.append(np.nan)
            for layer in layer_order:
                layer_data[layer].append(np.nan)

    # Create the plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))

    # Plot 1: Average parameter change from pretrained
    axes[0].plot(epoch_nums, before_tpgm_values, 'bo-', linewidth=3, markersize=8)
    axes[0].set_title('Average Parameter Change from Pretrained', fontsize=18, fontweight='bold', pad=20)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Parameter Change', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    # Plot 2: Global Projection Ratios
    axes[1].plot(epoch_nums, global_mins, 'r.-', label='Min', linewidth=3, markersize=8)
    axes[1].plot(epoch_nums, global_maxs, 'g.-', label='Max', linewidth=3, markersize=8)
    axes[1].plot(epoch_nums, global_means, 'b.-', label='Mean', linewidth=3, markersize=8)
    axes[1].set_title('Global Projection Ratios', fontsize=18, fontweight='bold', pad=20)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Ratio', fontsize=14)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    # Plot 3: Layer Projection Ratios
    colors = plt.cm.tab20(np.linspace(0, 1, len(layer_order)))

    for i, layer in enumerate(layer_order):
        if not all(np.isnan(layer_data[layer])):
            axes[2].plot(epoch_nums, layer_data[layer], 'o-',
                        color=colors[i], label=layer, linewidth=2, markersize=6)

    axes[2].set_title('Layer Projection Ratios (Ordered by Model Flow)', fontsize=18, fontweight='bold', pad=20)
    axes[2].set_xlabel('Epoch', fontsize=14)
    axes[2].set_ylabel('Ratio', fontsize=14)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, ncol=1)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

    return fig

def analyze_log(log_input):
    """
    Main function to analyze log data

    Args:
        log_input: Either a file path (string) or log content (string)
    """

    # Determine if input is file path or content
    if log_input.count('\n') < 10 and len(log_input) < 500:
        # Likely a file path
        try:
            with open(log_input, 'r') as f:
                log_content = f.read()
            print(f"✓ Read log from file: {log_input}")
        except FileNotFoundError:
            print(f"✗ File {log_input} not found!")
            return None
    else:
        # Likely log content
        log_content = log_input
        print("✓ Using provided log content")

    # Parse the log
    print("📊 Parsing log data...")
    epochs, layer_order = parse_log(log_content)

    print(f"\n📈 Analysis Results:")
    print(f"   Found {len(epochs)} epochs")

    for i, epoch in enumerate(epochs):
        print(f"   Epoch {i+1}: Before TPGM = {epoch['before_tpgm']:.6f}, {len(epoch['iterations'])} iterations")

    # Create plots
    print("\n🎨 Creating plots...")
    fig = create_plots(epochs, layer_order)

    if fig:
        print("✓ Plots created successfully!")
        # Optionally save the figure
        # fig.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
        # print("✓ Plot saved as 'training_analysis.png'")

    return fig, epochs, layer_order

# Usage Examples:

fig, epochs, layer_order = analyze_log("tpgm_ratios.log")