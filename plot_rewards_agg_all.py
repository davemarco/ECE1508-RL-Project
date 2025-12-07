import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration for plots
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Blue, Green, Orange (Matplotlib defaults)

def get_aggregated_stats(input_dir_path):
    """
    Scans a directory for multiple seeds, aligns them, and computes 
    mean and std deviation.
    
    Returns:
        tuple: (x_axis, y_mean, y_std) or None if no data found.
    """
    path = Path(input_dir_path)
    if not path.exists():
        print(f"Warning: Path {path} does not exist. Skipping.")
        return None

    # Find all seed files recursively
    y_files = sorted(list(path.rglob("y_data_*.npy")))
    x_files = sorted(list(path.rglob("x_data_*.npy")))
    
    if not y_files:
        print(f"Warning: No .npy files found in {path}. Skipping.")
        return None

    # Load raw data
    raw_y = [np.load(f) for f in y_files]
    raw_x = [np.load(f) for f in x_files]
    
    # Align data to the shortest run (truncate)
    min_len = min(len(y) for y in raw_y)
    y_stack = np.vstack([y[:min_len] for y in raw_y])
    
    # Assume X axis is consistent, take first seed and truncate
    x_axis = raw_x[0][:min_len]

    # Compute stats
    y_mean = np.mean(y_stack, axis=0)
    y_std = np.std(y_stack, axis=0)
    
    return x_axis, y_mean, y_std

def plot_comparison(methods_data, output_dir):
    """
    methods_data: Dict of { "Method Name": (x_data, y_mean, y_std) }
    """
    plt.figure(figsize=(12, 7))
    
    # Iterate over methods and their assigned colors
    for i, (name, (x, mean, std)) in enumerate(methods_data.items()):
        # Cycle through colors if we have more methods than defined colors
        color = COLORS[i % len(COLORS)]
        
        # 1. Plot Mean Line
        plt.plot(x, mean, color=color, linewidth=2, label=name)
        
        # 2. Plot Standard Deviation Shading
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        
        # 3. Add text annotation for final value
        # Position: slightly above the final point
        last_x = x[-1]
        last_y = mean[-1]
        
        # Determine text offset based on y-axis scale
        # (Heuristic: 2% of the max height of the graph)
        offset = 1100 * 0.02 
        
        plt.text(
            last_x, 
            last_y + offset, 
            f"{last_y:.1f}", 
            color=color, 
            fontweight='bold',
            fontsize=9,
            ha='left' # Align text to the left of the point
        )

    plt.ylim([0, 1100]) # Fixed limit as requested previously
    
    # Find the maximum x value across all methods to set x-limit
    max_x = max(data[0][-1] for data in methods_data.values())
    plt.xlim([0, max_x * 1.1]) # Add 10% padding for the text labels

    plt.xlabel("# Environment Steps", fontsize=12)
    plt.ylabel("Reward per Episode", fontsize=12)
    plt.title("Method Comparison: Training Performance", fontsize=14)
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Save
    if output_dir:
        out_path = Path(output_dir) / "comparison_plot.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {out_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    # --- CONFIGURATION: DEFINE YOUR METHODS HERE ---
    # Map the Label you want in the legend to the folder path containing seeds
    methods_config = {
        "SAC": "/home/johnl/results/base/SAC",
        "PPO": "/home/johnl/results/base/PPO",
        "TD3": "/home/johnl/results/base/FastTD3",
    }
    # -----------------------------------------------

    processed_data = {}

    print("Processing data...")
    for method_name, path in methods_config.items():
        result = get_aggregated_stats(path)
        if result:
            processed_data[method_name] = result
            print(f"Loaded {method_name}")
    
    if processed_data:
        plot_comparison(processed_data, args.output_dir)
    else:
        print("No valid data found to plot.")