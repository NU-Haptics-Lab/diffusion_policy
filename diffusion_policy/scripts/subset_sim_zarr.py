"""A random subset of a sim_to_real zarr's episodes -> a new zarr: whole episodes, drawn at random until --rows frames.
Image arrays (one-row chunks) are copied chunk file by chunk file, still compressed; the other arrays are read and
indexed. meta/episode_ends, and the attrs (sim_episodes subset, subset_of, subset_seed) follow. If the source has
sim_views.py's sidecars beside it (<zarr without .zarr>_views.npz / _start_cost.npy), their subsets are written beside
the output, for add_sim_frame_labels.py.

    python3 diffusion_policy/scripts/subset_sim_zarr.py /media/dexnex_ssd/datasets/layout/aiet_grip_sim_to_real.zarr \\
        /media/dexnex_ssd/datasets/aiet_grip_layout50k_sim_to_real.zarr --rows 50000
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--rows", type=int, default=50000, help="frames to keep (whole episodes: stops at the first episode past it)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    register_codecs()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    src = zarr.open(args.src, mode="r")
    ends = np.asarray(src["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    order = np.random.default_rng(args.seed).permutation(len(ends))
    keep = order[:int(np.searchsorted(np.cumsum((ends - starts)[order]), args.rows)) + 1]
    keep = np.sort(keep)  # (the source's episode order)
    rows = np.concatenate([np.arange(starts[k], ends[k]) for k in keep])
    lengths = ends[keep] - starts[keep]
    print(f"{args.src}: {len(keep)} of {len(ends)} episodes, {len(rows)} of {int(ends[-1])} frames (seed {args.seed})")

    out = zarr.open(args.out, mode="w")
    out.create_group("meta").array("episode_ends", np.cumsum(lengths).astype(ends.dtype), chunks=src["meta/episode_ends"].chunks,
                                   compressor=src["meta/episode_ends"].compressor)
    data = out.create_group("data")
    for key in sorted(src["data"].keys()):
        a = src["data"][key]
        if a.chunks[0] == 1 and a.ndim > 1:  # (images: one compressed file a frame -- copied as they are)
            dst = data.create(key, shape=(len(rows),) + a.shape[1:], chunks=a.chunks, dtype=a.dtype, compressor=a.compressor,
                              filters=a.filters, fill_value=a.fill_value, dimension_separator=a._dimension_separator)
            sep = a._dimension_separator or "."
            tail = sep.join(["0"] * (a.ndim - 1))
            sdir, ddir = os.path.join(args.src, "data", key), os.path.join(args.out, "data", key)

            def copy(j_r):
                j, r = j_r
                name = f"{r}{sep}{tail}"
                if os.path.exists(os.path.join(sdir, name)):  # (an all-fill chunk has no file)
                    shutil.copyfile(os.path.join(sdir, name), os.path.join(ddir, f"{j}{sep}{tail}"))

            with ThreadPoolExecutor(32) as pool:
                list(tqdm(pool.map(copy, enumerate(rows.tolist())), total=len(rows), desc=key, unit="img"))
            assert dst.shape[0] == len(rows)
        else:
            data.array(key, a[:][rows], chunks=a.chunks, compressor=a.compressor, filters=a.filters)
            print(f"  {key} {a.dtype} {(len(rows),) + a.shape[1:]}")
    attrs = dict(src.attrs)
    if "sim_episodes" in attrs:
        attrs["sim_episodes"] = [attrs["sim_episodes"][k] for k in keep]
    attrs.update(subset_of=os.path.abspath(args.src), subset_seed=args.seed, subset_episodes=keep.tolist())
    out.attrs.update(attrs)

    stem, out_stem = args.src.rstrip("/").removesuffix(".zarr"), args.out.rstrip("/").removesuffix(".zarr")
    if os.path.exists(stem + "_views.npz"):
        views = np.load(stem + "_views.npz")
        np.savez(out_stem + "_views.npz", **{k: views[k][rows] for k in views.files})
        print(f"  -> {out_stem}_views.npz")
    if os.path.exists(stem + "_start_cost.npy"):
        np.save(out_stem + "_start_cost.npy", np.load(stem + "_start_cost.npy")[rows])
        print(f"  -> {out_stem}_start_cost.npy")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
