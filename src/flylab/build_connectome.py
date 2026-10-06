"""Build compact connectome artifacts for the simulator and the 3D brain view.

Reads:
  upstream/Drosophila_brain_model/Completeness_783.csv   (neuron roster, FlyWire 783)
  upstream/Drosophila_brain_model/Connectivity_783.parquet (15.1M synaptic edges)
  data/raw/neuron_annotations.tsv                        (soma coords, classes, NT)

Writes into data/derived/:
  neurons.npz     positions (float32, normalised mm-ish), class/nt/side codes
  edges.npz       CSR-style int32 arrays for fast spike propagation
  meta.json       label vocabularies + region groupings for the web UI
  brain_points.bin  Float32Array [x,y,z, class, nt] per neuron for WebGL
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UP = ROOT / "upstream" / "Drosophila_brain_model"
OUT = ROOT / "data" / "derived"

# Coarse neuropil groups used for the glowing region indicators. FlyWire
# annotations give us super_class + cell_class; we fold them into functional
# blocks that a non-specialist can reason about.
REGION_RULES = [
    ("optic_lobe", lambda sc, cc: sc in {"optic"}),
    ("visual_projection", lambda sc, cc: sc in {"visual_projection", "visual_centrifugal"}),
    ("sensory", lambda sc, cc: sc in {"sensory", "sensory_ascending"}),
    ("descending", lambda sc, cc: sc == "descending"),
    ("ascending", lambda sc, cc: sc == "ascending"),
    ("motor", lambda sc, cc: sc == "motor"),
    ("endocrine", lambda sc, cc: sc == "endocrine"),
    ("central", lambda sc, cc: True),
]


def classify_region(super_class: str, cell_class: str) -> str:
    for name, rule in REGION_RULES:
        if rule(super_class, cell_class):
            return name
    return "central"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading neuron roster")
    comp = pd.read_csv(UP / "Completeness_783.csv", index_col=0)
    comp.index = comp.index.astype("int64")
    comp.index.name = "root_id"
    n = len(comp)
    # Brian2 index == row order in Completeness_783.csv (see upstream model.py)
    order = pd.Series(np.arange(n, dtype=np.int32), index=comp.index)

    print("[2/5] loading FlyWire annotations")
    ann = pd.read_csv(
        RAW / "neuron_annotations.tsv",
        sep="\t",
        low_memory=False,
        usecols=[
            "root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
            "super_class", "cell_class", "cell_type", "top_nt", "side",
        ],
    )
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(comp.index)

    # Prefer soma position (cell body); fall back to the annotation point.
    # FlyWire coords are in 4x4x40 nm voxels -> convert to micrometres.
    sx = ann["soma_x"].to_numpy(dtype="float64")
    sy = ann["soma_y"].to_numpy(dtype="float64")
    sz = ann["soma_z"].to_numpy(dtype="float64")
    px = ann["pos_x"].to_numpy(dtype="float64")
    py = ann["pos_y"].to_numpy(dtype="float64")
    pz = ann["pos_z"].to_numpy(dtype="float64")
    x = np.where(np.isnan(sx), px, sx) * 0.004  # 4 nm -> um
    y = np.where(np.isnan(sy), py, sy) * 0.004
    z = np.where(np.isnan(sz), pz, sz) * 0.040  # 40 nm sections -> um

    missing = np.isnan(x) | np.isnan(y) | np.isnan(z)
    print(f"      neurons without coordinates: {int(missing.sum())} (jittered near centroid)")
    cx, cy, cz = np.nanmedian(x), np.nanmedian(y), np.nanmedian(z)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 12.0, size=(int(missing.sum()), 3))
    x[missing] = cx + jitter[:, 0]
    y[missing] = cy + jitter[:, 1]
    z[missing] = cz + jitter[:, 2]

    # Centre on the brain centroid, keep micrometres so the web view can scale.
    pos = np.stack([x, y, z], axis=1).astype(np.float32)
    pos -= pos.mean(axis=0, keepdims=True)

    print("[3/5] encoding labels")
    super_class = ann["super_class"].fillna("unknown").astype(str)
    cell_class = ann["cell_class"].fillna("unknown").astype(str)
    region = [classify_region(a, b) for a, b in zip(super_class, cell_class)]
    region_names = [r[0] for r in REGION_RULES]
    region_code = np.array([region_names.index(r) for r in region], dtype=np.int8)

    nt_names = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
                "octopamine", "unknown"]
    nt = ann["top_nt"].fillna("unknown").astype(str).str.lower()
    nt_code = np.array([nt_names.index(v) if v in nt_names else len(nt_names) - 1
                        for v in nt], dtype=np.int8)

    side_names = ["left", "right", "center", "unknown"]
    side = ann["side"].fillna("unknown").astype(str).str.lower()
    side_code = np.array([side_names.index(v) if v in side_names else 3
                          for v in side], dtype=np.int8)

    cell_type = ann["cell_type"].fillna("").astype(str).to_numpy()

    print("[4/5] loading connectivity (15M edges)")
    con = pd.read_parquet(
        UP / "Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy(dtype=np.int32)
    post = con["Postsynaptic_Index"].to_numpy(dtype=np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)
    del con

    # Sort by presynaptic index -> CSR layout for cache-friendly propagation.
    print("      sorting into CSR")
    sort_idx = np.argsort(pre, kind="stable")
    pre = pre[sort_idx]
    post = post[sort_idx]
    w = w[sort_idx]
    indptr = np.zeros(n + 1, dtype=np.int64)
    counts = np.bincount(pre, minlength=n)
    indptr[1:] = np.cumsum(counts)

    print("[5/5] writing artifacts")
    np.savez_compressed(
        OUT / "neurons.npz",
        root_id=comp.index.to_numpy(dtype="int64"),
        pos=pos,
        region=region_code,
        nt=nt_code,
        side=side_code,
        cell_type=cell_type,
    )
    np.savez(
        OUT / "edges.npz",
        indptr=indptr,
        indices=post,
        weight=w,
    )

    # Interleaved buffer for the WebGL point cloud: x,y,z,region,nt
    buf = np.zeros((n, 5), dtype=np.float32)
    buf[:, 0:3] = pos
    buf[:, 3] = region_code
    buf[:, 4] = nt_code
    buf.tofile(OUT / "brain_points.bin")

    meta = {
        "n_neurons": int(n),
        "n_edges": int(len(post)),
        "regions": region_names,
        "region_counts": {r: int((region_code == i).sum())
                          for i, r in enumerate(region_names)},
        "neurotransmitters": nt_names,
        "sides": side_names,
        "bounds": {
            "min": pos.min(axis=0).tolist(),
            "max": pos.max(axis=0).tolist(),
        },
        "source": {
            "connectome": "FlyWire FAFB v783 (Dorkenwald et al. 2024)",
            "model": "Shiu et al. 2024 leaky integrate-and-fire brain model",
            "annotations": "Schlegel et al. 2024 flywire_annotations",
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"done: {n:,} neurons, {len(post):,} edges")
    print(f"      {OUT / 'brain_points.bin'} ({buf.nbytes / 1e6:.1f} MB)")
    for k, v in meta["region_counts"].items():
        print(f"      {k:20s} {v:>7,}")


if __name__ == "__main__":
    main()

