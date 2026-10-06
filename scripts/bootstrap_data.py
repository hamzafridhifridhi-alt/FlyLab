"""Cross-platform Python script to fetch FlyLab external data and vendored libraries."""
import os
import re
import subprocess
import urllib.request
from pathlib import Path

FLYLAB_ROOT = Path(__file__).resolve().parents[1]
THREE_VERSION = "0.169.0"
ANNOTATIONS_URL = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)

def main():
    print(f"==> FlyLab data bootstrap in {FLYLAB_ROOT}")
    upstream = FLYLAB_ROOT / "upstream"
    data_raw = FLYLAB_ROOT / "data" / "raw"
    web_vendor = FLYLAB_ROOT / "web" / "vendor"
    
    upstream.mkdir(parents=True, exist_ok=True)
    data_raw.mkdir(parents=True, exist_ok=True)
    web_vendor.mkdir(parents=True, exist_ok=True)
    
    # 1. Drosophila brain model
    brain_model_dir = upstream / "Drosophila_brain_model"
    if brain_model_dir.exists() and (brain_model_dir / ".git").exists():
        print("[1/4] Brain model repository already present, skipping")
    else:
        print("[1/4] Cloning Shiu et al. brain model (~380 MB)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/philshiu/Drosophila_brain_model.git", str(brain_model_dir)],
            check=True, cwd=str(FLYLAB_ROOT)
        )
        
    # 2. FlyGym source
    flygym_dir = upstream / "flygym"
    if flygym_dir.exists() and (flygym_dir / ".git").exists():
        print("[2/4] FlyGym source repository already present, skipping")
    else:
        print("[2/4] Cloning FlyGym / NeuroMechFly v2 source...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NeLy-EPFL/flygym.git", str(flygym_dir)],
            check=True, cwd=str(FLYLAB_ROOT)
        )
        
    # 3. FlyWire annotations
    ann_path = data_raw / "neuron_annotations.tsv"
    if ann_path.exists() and ann_path.stat().st_size > 0:
        print("[3/4] FlyWire annotations already present, skipping")
    else:
        print("[3/4] Downloading FlyWire neuron annotations (~30 MB)...")
        download_file(ANNOTATIONS_URL, ann_path)
        
    # 4. Three.js
    three_module = web_vendor / "three.module.js"
    bloom_pass = web_vendor / "UnrealBloomPass.js"
    if three_module.exists() and bloom_pass.exists():
        print("[4/4] Three.js already vendored, skipping")
    else:
        print(f"[4/4] Downloading Three.js {THREE_VERSION}...")
        download_file(f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.js", three_module)
        download_file(f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js", web_vendor / "OrbitControls.js")
        
        modules = [
            "postprocessing/EffectComposer.js",
            "postprocessing/RenderPass.js",
            "postprocessing/ShaderPass.js",
            "postprocessing/MaskPass.js",
            "postprocessing/UnrealBloomPass.js",
            "postprocessing/OutputPass.js",
            "postprocessing/Pass.js",
            "shaders/CopyShader.js",
            "shaders/LuminosityHighPassShader.js",
            "shaders/OutputShader.js",
        ]
        for m in modules:
            fname = Path(m).name
            download_file(f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/{m}", web_vendor / fname)
            
        targets = [p for p in web_vendor.glob("*.js") if p.name != "three.module.js"]
        for path in targets:
            source = path.read_text(encoding="utf-8")
            patched = source.replace("from 'three'", "from '/vendor/three.module.js'")
            patched = re.sub(r"from '\.\./[a-z]+/([A-Za-z0-9_]+\.js)'", r"from '/vendor/\1'", patched)
            patched = re.sub(r"from '\./([A-Za-z0-9_]+\.js)'", r"from '/vendor/\1'", patched)
            if patched != source:
                path.write_text(patched, encoding="utf-8")
                
        unresolved = [
            (path.name, match.group(1))
            for path in targets
            for match in re.finditer(r"from '([^']+)'", path.read_text(encoding="utf-8"))
            if not match.group(1).startswith("/vendor/")
        ]
        if unresolved:
            raise RuntimeError(f"vendor imports could not be rewritten: {unresolved}")
        print(f"      rewrote imports in {len(targets)} vendored modules")

    print("\nBootstrap complete! Next run connectome build.")

if __name__ == "__main__":
    main()
