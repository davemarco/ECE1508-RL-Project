import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter

def get_args():
    parser = argparse.ArgumentParser(description="Generate uneven terrain heightfield for MuJoCo")
    parser.add_argument("--size", type=float, default=200.0, help="Terrain size in meters (default: 200m x 200m)")
    parser.add_argument("--resolution", type=float, default=0.2, help="Grid resolution in meters (default: 0.2m)")
    parser.add_argument("--num_bumps", type=int, default=None, help="Number of bumps to generate (default: auto-scaled with area)")
    parser.add_argument("--min_radius", type=int, default=1, help="Minimum bump radius in cells (default: 4)")
    parser.add_argument("--max_radius", type=int, default=5, help="Maximum bump radius in cells (default: 20)")
    parser.add_argument("--min_height", type=float, default=0.0, help="Minimum bump height fraction (default: 0.0)")
    parser.add_argument("--max_height", type=float, default=1.0, help="Maximum bump height fraction (default: 1.0)")
    parser.add_argument("--height_bias", type=float, default=2.0, help="Power law bias for height distribution (>1 favors smaller bumps, default: 2.0)")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of global noise (default: 0)")
    parser.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian smoothing sigma (default: 0.0, disabled)")
    parser.add_argument("--output", type=str, default="common/heightfields/terrain.png", help="Output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Auto-scale bump count to maintain density (8000 bumps over 100m × 100m = 0.8 bumps/m²).
    if args.num_bumps is None:
        bumps_per_m2 = 20000 / (100.0 * 100.0)
        args.num_bumps = max(1, int(bumps_per_m2 * args.size * args.size))
        print(f"Auto-scaled bump count for {args.size}m × {args.size}m terrain: {args.num_bumps}")

    # Grid: size / resolution
    N = int(args.size / args.resolution)
    rng = np.random.default_rng(args.seed)

    # Output path
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Base height map in [0,1]
    h = np.zeros((N, N), dtype=np.float32)

    # Precompute coordinate grid for radial bumps
    yy, xx = np.ogrid[:N, :N]

    # Add bumps with specified parameters
    for _ in range(args.num_bumps):
        cx, cy = rng.integers(0, N, size=2)
        r = int(rng.integers(args.min_radius, args.max_radius))
        r = max(r, 1)

        # Squared distance field
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist2 <= (r * r)

        # Non-uniform height distribution using power law
        # height_bias > 1: more small bumps, fewer tall bumps
        # height_bias < 1: more tall bumps, fewer small bumps
        # height_bias = 1: uniform distribution (original behavior)
        random_val = rng.random() ** args.height_bias
        bump_amp = args.min_height + (args.max_height - args.min_height) * random_val
        bump = bump_amp * (1.0 - dist2 / float(r * r))
        bump = np.clip(bump, 0.0, 1.0)

        h[mask] = np.maximum(h[mask], bump[mask])

    # # Apply Gaussian smoothing to remove sharp transitions
    # if args.smooth_sigma > 0:
    #     h = gaussian_filter(h, sigma=args.smooth_sigma)
    #
    # # Add gentle global roughness
    # if args.noise_std > 0:
    #     h += rng.normal(0.0, args.noise_std, size=h.shape).astype(np.float32)

    h = np.clip(h, 0.0, 1.0)

    # Save PNG
    img = Image.fromarray((h * 255).astype(np.uint8))
    img.save(out_path.as_posix())

    print(f"Saved heightfield to: {out_path.resolve()}")
    print(f"Grid size: {N}x{N}, Bumps: {args.num_bumps}, Radius: [{args.min_radius}, {args.max_radius}]")
    print(f"Height range: [{args.min_height}, {args.max_height}], Height bias: {args.height_bias}, Smoothing: {args.smooth_sigma}")
