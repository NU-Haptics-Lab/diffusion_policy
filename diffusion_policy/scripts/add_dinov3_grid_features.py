"""DINOv3-L 3x3-grid image features for the diffusion policy: a few hundred numbers per camera that keep fine position.

The pipeline the token-compression probe picked (avatar_intelligence analysis/dinov3_token_compression_probe.py,
2026-11-10: 448 px, PCA k = 32 and a fixed 3x3 average-pooled grid beat 224, smaller k, attention pooling and mean
pooling on palm-position error):

    the zarr's 224 x 224 frame -> bicubic to 448 -> DINOv3 ViT-L/16 (bf16) -> 28 x 28 patch tokens (1024-d)
    -> PCA to 32 channels (one basis per camera, fit once over the training zarrs, scaled by the first component's std)
    -> adaptive average pool to 3 x 3 cells -> data/<camera>_grid3x3_dinov3l  [T, 9, 32] float16  (cells row-major)

A cell's position is fixed by its index (the policy's per-key Linear sees it), and each patch token still carries its
own position (DINOv3's RoPE: linearly decodable to ~1 patch), so the cell averages keep some within-cell position.

    fit-pca  sample frames from every training zarr, fit the per-camera PCA -> one .npz (the same file is needed at
             inference: encode() below)
    add      write the three feature arrays into a zarr (incremental and resumable: rows past attrs["rows_done"];
             refuses a different PCA file than the one the arrays were written with unless --restart)

    python diffusion_policy/scripts/add_dinov3_grid_features.py fit-pca --zarrs A.zarr B.zarr --out pca.npz
    python diffusion_policy/scripts/add_dinov3_grid_features.py add --zarr A.zarr --pca pca.npz

Only the env camera of real task_id 27 is valid (other real task_ids record no env camera): the PCA fit samples its
env frames from task_id 27 rows only; `add` encodes every row (the config masks the env key for other task_ids).
"""

import argparse
import hashlib
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs(verbose=False)

CAMERAS = {  # zarr image key -> feature key
    "wrist_image": "wrist_cam_grid3x3_dinov3l",
    "overhead_image": "overhead_cam_grid3x3_dinov3l",
    "env_image": "env_cam_grid3x3_dinov3l",
}
ENV_CAMERA, ENV_TASK_IDS = "env_image", (27,)
IMG_SIZE = 448
GRID = 3
K = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHUNK = 2048  # rows a block (and the feature arrays' zarr chunk)
DECODE_THREADS = 16


def load_dinov3(repo: str, weights: str, device: torch.device):
    model = torch.hub.load(os.path.expanduser(repo), "dinov3_vitl16", source="local", weights=os.path.expanduser(weights))
    return model.eval().to(device)


@torch.no_grad()
def dino_tokens(model, images: np.ndarray, device: torch.device, batch: int) -> torch.Tensor:
    """[N, h, w, 3] uint8 -> [N, (IMG_SIZE/16)^2, 1024] fp32 patch tokens (bicubic to IMG_SIZE, bf16 autocast)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model.forward_features((x - mean) / std)["x_norm_patchtokens"].float())
    return torch.cat(out)


class GridPCA:
    """The fitted per-camera PCA (fit-pca's .npz) and the token -> grid reduction."""

    def __init__(self, path: str, device: torch.device):
        d = np.load(path)
        self.path, self.sha1 = path, hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]
        if int(d["img_size"]) != IMG_SIZE or int(d["k"]) != K or int(d["grid"]) != GRID:
            raise SystemExit(f"{path}: fit for img_size {int(d['img_size'])} / k {int(d['k'])} / grid {int(d['grid'])}, "
                             f"this script uses {IMG_SIZE} / {K} / {GRID}")
        self.mean = {cam: torch.tensor(d[f"{cam}/mean"], device=device) for cam in CAMERAS}
        self.basis = {cam: torch.tensor(d[f"{cam}/basis"], device=device) for cam in CAMERAS}  # [1024, K], scaled

    def grid(self, cam: str, tokens: torch.Tensor) -> torch.Tensor:
        """[N, G*G, 1024] tokens -> [N, GRID*GRID, K] cell means of the PCA scores (row-major cells)."""
        n, g = len(tokens), int(round(tokens.shape[1] ** 0.5))
        scores = (tokens - self.mean[cam]) @ self.basis[cam]  # [N, G*G, K]
        cells = F.adaptive_avg_pool2d(scores.transpose(1, 2).reshape(n, K, g, g), GRID)  # [N, K, GRID, GRID]
        return cells.reshape(n, K, GRID * GRID).transpose(1, 2)


@torch.no_grad()
def encode(model, pca: GridPCA, images: dict[str, np.ndarray], device: torch.device, batch: int = 64) -> dict[str, np.ndarray]:
    """{zarr image key: [N, 224, 224, 3] uint8} -> {feature key: [N, 9, 32] float32}: the training features' exact
    pipeline (for inference: the same 224 x 224 frames the zarrs store, the same PCA file)."""
    return {CAMERAS[cam]: pca.grid(cam, dino_tokens(model, imgs, device, batch)).cpu().numpy() for cam, imgs in images.items()}


def read_rows(pool: ThreadPoolExecutor, array, rows: np.ndarray) -> np.ndarray:
    """array[rows] for one-row-chunk image arrays, decoded in parallel."""
    return np.stack(list(pool.map(lambda r: np.asarray(array[int(r)]), rows)))


# ----------------------------------------------------------------------------- fit-pca


def fit_pca(args) -> None:
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    model = load_dinov3(args.dinov3_repo, args.weights, device)
    stores = [zarr.open(os.path.expanduser(p), "r") for p in args.zarrs]
    per_zarr = args.frames // len(stores)
    out = {"img_size": IMG_SIZE, "k": K, "grid": GRID, "zarrs": np.array([os.path.abspath(os.path.expanduser(p)) for p in args.zarrs]),
           "frames_per_zarr": per_zarr, "weights": os.path.basename(args.weights)}
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        for cam in CAMERAS:
            s = torch.zeros(1024, device=device, dtype=torch.float64)
            ss = torch.zeros(1024, 1024, device=device, dtype=torch.float64)
            count = 0
            for path, z in zip(args.zarrs, stores):
                n = z["data"][cam].shape[0]
                rows = np.arange(n)
                if cam == ENV_CAMERA and "task_id" in z["data"]:
                    rows = rows[np.isin(z["data"]["task_id"][:], ENV_TASK_IDS)]
                rows = np.sort(rng.choice(rows, min(per_zarr, len(rows)), replace=False))
                for i in tqdm(range(0, len(rows), args.batch), desc=f"PCA fit {cam} <- {os.path.basename(path)}", unit="batch"):
                    t = dino_tokens(model, read_rows(pool, z["data"][cam], rows[i:i + args.batch]), device, args.batch)
                    t = t.reshape(-1, 1024).double()
                    s += t.sum(0)
                    ss += t.T @ t
                    count += len(t)
            mu = s / count
            evals, evecs = torch.linalg.eigh(ss / count - torch.outer(mu, mu))
            evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)
            explained = (evals.cumsum(0) / evals.sum()).cpu().numpy()
            out[f"{cam}/mean"] = mu.float().cpu().numpy()
            out[f"{cam}/basis"] = (evecs[:, :K] / evals[0].sqrt()).float().cpu().numpy()
            out[f"{cam}/explained"] = explained
            print(f"{cam}: {count} tokens; PCA explained variance k=8 {100 * explained[7]:.0f}%, k=16 {100 * explained[15]:.0f}%, "
                  f"k={K} {100 * explained[K - 1]:.0f}%, k=64 {100 * explained[63]:.0f}%", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"-> {args.out}")


# ----------------------------------------------------------------------------- add


def add(args) -> None:
    device = torch.device("cuda")
    pca = GridPCA(os.path.expanduser(args.pca), device)
    z = zarr.open(os.path.expanduser(args.zarr), "r+")
    data = z["data"]
    n = data[next(iter(CAMERAS))].shape[0]
    arrays = {}
    for cam, key in CAMERAS.items():
        if key in data and args.restart:
            del data[key]
        if key not in data:
            data.create_dataset(key, shape=(n, GRID * GRID, K), chunks=(CHUNK, GRID * GRID, K), dtype="f2",
                                compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE))
            data[key].attrs.update(rows_done=0, pca_file=os.path.abspath(pca.path), pca_sha1=pca.sha1, img_size=IMG_SIZE,
                                   k=K, grid=GRID, source=cam)
        a = data[key]
        if a.attrs.get("pca_sha1") != pca.sha1:
            raise SystemExit(f"{key}: written with PCA {a.attrs.get('pca_sha1')} ({a.attrs.get('pca_file')}), not {pca.sha1} "
                             f"({pca.path}): mixing them would be wrong -- --restart to rewrite it from row 0")
        if a.shape[0] != n:
            a.resize(n, GRID * GRID, K)  # (the zarr grew: incremental)
        arrays[cam] = a
    start = min(int(a.attrs["rows_done"]) for a in arrays.values())
    if start >= n:
        print(f"{args.zarr}: all {n} rows done")
        return
    print(f"{args.zarr}: rows {start}..{n} ({n - start} frames x {len(CAMERAS)} cameras), PCA {pca.sha1}", flush=True)
    model = load_dinov3(args.dinov3_repo, args.weights, device)

    blocks = [(a, min(a + CHUNK, n)) for a in range(start - start % CHUNK, n, CHUNK)]
    ready: queue.Queue = queue.Queue(maxsize=2)  # (the next blocks' images decoded while the GPU works)

    def reader():
        with ThreadPoolExecutor(DECODE_THREADS) as pool:
            for a, b in blocks:
                ready.put((a, b, {cam: read_rows(pool, data[cam], np.arange(a, b)) for cam in CAMERAS}))
        ready.put(None)

    threading.Thread(target=reader, daemon=True).start()
    bar = tqdm(total=n - start, initial=0, desc=f"grid features {os.path.basename(args.zarr.rstrip('/'))}", unit="frame")
    t0 = time.time()
    while (item := ready.get()) is not None:
        a, b, images = item
        feats = encode(model, pca, images, device, args.batch)
        for cam, arr in arrays.items():
            arr[a:b] = feats[CAMERAS[cam]].astype(np.float16)
            arr.attrs["rows_done"] = b
        bar.update(b - max(a, start))
        bar.set_postfix(img_s=f"{len(CAMERAS) * (b - start) / (time.time() - t0):.0f}")
    bar.close()
    sample = arrays[next(iter(CAMERAS))][max(0, n - 4):n].astype(np.float32)
    print(f"-> {', '.join(CAMERAS.values())}: {n} rows; last rows' feature std {sample.std():.3f}, "
          f"finite {np.isfinite(sample).all()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov3-repo", default="~/libraries/dinov3")
    parser.add_argument("--weights", default="~/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--batch", type=int, default=64, help="images a DINOv3 forward")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit-pca")
    p.add_argument("--zarrs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=4000, help="frames per camera, split evenly over the zarrs")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("add")
    p.add_argument("--zarr", required=True)
    p.add_argument("--pca", required=True)
    p.add_argument("--restart", action="store_true", help="delete the feature arrays and rewrite them from row 0")
    args = parser.parse_args()
    {"fit-pca": fit_pca, "add": add}[args.cmd](args)


if __name__ == "__main__":
    main()
