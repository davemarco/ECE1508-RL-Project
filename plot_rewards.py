import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot(x_data, y_data, y_dataerr, output_dir):
    plt.xlim([0, x_data[-1] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"Final Reward: {y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig(f"{output_dir}/rewards_per_episode.png")
    plt.close()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir
    
    x_data = list(Path(input_dir).glob("x_data_*.npy"))[-1]
    y_data = list(Path(input_dir).glob("y_data_*.npy"))[-1]
    y_dataerr = list(Path(input_dir).glob("y_dataerr_*.npy"))[-1]
    
    x_data = np.load(x_data)
    y_data = np.load(y_data)
    y_dataerr = np.load(y_dataerr)
    
    plot(x_data, y_data, y_dataerr, output_dir)