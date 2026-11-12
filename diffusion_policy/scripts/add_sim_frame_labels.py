"""Per-frame sim labels -> data arrays of a sim_to_real zarr, for TrainAndVal's weighted sampling (label_bins on a
replay-buffer key): what openpi's cost_bins / view_weights read from meta/frame_start_cost.npy / meta/frame_views.npz.

  data/start_cost    float32 [T]  each frame's state's task cost (cost-balanced sampling: label_bins over it)
  data/fine_align    uint8   [T]  1 = fine alignment (a pad within 2 cm of the pellet, before the lift)
  data/flush_corner  uint8   [T]  1 = an episode that started with the pellet pressed against the tray

Both inputs come from avatar_intelligence tree_exploration/sim_views.py, beside the zarr:
<zarr without .zarr>_start_cost.npy and <zarr without .zarr>_views.npz (the zarr's frame order). Rewrites the arrays.

    python3 diffusion_policy/scripts/add_sim_frame_labels.py /media/dexnex_ssd/datasets/aiet_grip_sim_to_real.zarr
"""

import argparse

import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr")
    parser.add_argument("--start-cost", default=None, help="default: <zarr without .zarr>_start_cost.npy")
    parser.add_argument("--views", default=None, help="default: <zarr without .zarr>_views.npz")
    args = parser.parse_args()
    stem = args.zarr.rstrip("/").removesuffix(".zarr")
    root = zarr.open(args.zarr, mode="a")
    n = int(root["meta/episode_ends"][-1])
    cost = np.load(args.start_cost or stem + "_start_cost.npy").astype(np.float32)
    views = np.load(args.views or stem + "_views.npz")
    arrays = {"start_cost": cost, "fine_align": views["fine_align"].astype(np.uint8), "flush_corner": views["flush_corner"].astype(np.uint8)}
    for key, values in arrays.items():
        if len(values) != n:
            raise SystemExit(f"{key}: {len(values)} values for {n} frames in {args.zarr}")
    if not np.isfinite(cost).all():
        raise SystemExit(f"start_cost: {int((~np.isfinite(cost)).sum())} frames without a cost (sim_views.py couldn't tag them)")
    for key, values in arrays.items():
        root["data"].array(key, values, chunks=(min(n, 65536),), overwrite=True)
        summary = f"median {np.median(values):.2f}" if key == "start_cost" else f"{int(values.sum())} frames ({100 * values.mean():.1f}%)"
        print(f"{args.zarr}: data/{key} {values.dtype} [{n}] {summary}")


if __name__ == "__main__":
    main()
