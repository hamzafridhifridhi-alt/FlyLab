#!/usr/bin/env bash
# Fetch everything FlyLab needs that is too large to keep in git:
#   upstream/   the Shiu et al. brain model (connectome parquet) and FlyGym source
#   data/raw/   FlyWire neuron annotations (soma coordinates, cell types)
#   web/vendor/ pinned Three.js build for the browser point cloud
#
# Safe to re-run: anything already present is left alone.
set -euo pipefail

cd "$(dirname "$0")/.."
FLYLAB_ROOT="$(pwd)"
THREE_VERSION="0.169.0"
ANNOTATIONS_URL="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

echo "==> FlyLab data bootstrap in ${FLYLAB_ROOT}"
mkdir -p upstream data/raw web/vendor

# 1. Brain model + connectome (about 380 MB, contains Connectivity_783.parquet)
if [ -d upstream/Drosophila_brain_model/.git ]; then
  echo "[1/4] brain model already present, skipping"
else
  echo "[1/4] cloning Shiu et al. brain model (~380 MB)"
  git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git \
    upstream/Drosophila_brain_model
fi

# 2. FlyGym source. The pip package supplies the runtime; this clone is kept so
#    the reference tutorials and MJCF assets can be inspected alongside the code.
if [ -d upstream/flygym/.git ]; then
  echo "[2/4] flygym source already present, skipping"
else
  echo "[2/4] cloning FlyGym / NeuroMechFly v2 source"
  git clone --depth 1 https://github.com/NeLy-EPFL/flygym.git upstream/flygym
fi

# 3. FlyWire annotations: soma coordinates and cell types for the 3D brain view.
if [ -s data/raw/neuron_annotations.tsv ]; then
  echo "[3/4] FlyWire annotations already present, skipping"
else
  echo "[3/4] downloading FlyWire neuron annotations (~30 MB)"
  curl -fsSL -o data/raw/neuron_annotations.tsv "${ANNOTATIONS_URL}"
fi

# 4. Three.js core plus the post-processing chain used for the neuron bloom.
#    Vendored rather than loaded from a CDN so the UI works offline. These modules
#    import the bare specifier "three" and sibling relative paths, neither of
#    which a browser can resolve, so the imports are rewritten to served paths.
if [ -s web/vendor/three.module.js ] && [ -s web/vendor/UnrealBloomPass.js ]; then
  echo "[4/4] Three.js already vendored, skipping"
else
  echo "[4/4] downloading Three.js ${THREE_VERSION}"
  curl -fsSL -o web/vendor/three.module.js \
    "https://unpkg.com/three@${THREE_VERSION}/build/three.module.js"
  curl -fsSL -o web/vendor/OrbitControls.js \
    "https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls.js"
  for module in \
    postprocessing/EffectComposer.js \
    postprocessing/RenderPass.js \
    postprocessing/ShaderPass.js \
    postprocessing/MaskPass.js \
    postprocessing/UnrealBloomPass.js \
    postprocessing/OutputPass.js \
    postprocessing/Pass.js \
    shaders/CopyShader.js \
    shaders/LuminosityHighPassShader.js \
    shaders/OutputShader.js
  do
    curl -fsSL -o "web/vendor/$(basename "${module}")" \
      "https://unpkg.com/three@${THREE_VERSION}/examples/jsm/${module}"
  done
  python3 - <<'PYEOF'
import re
from pathlib import Path

# The server exposes web/vendor as a flat /vendor/ directory, so rewrite the
# bare "three" specifier and any ../shaders/ or ./ relative imports to match.
vendor = Path("web/vendor")
targets = [p for p in vendor.glob("*.js") if p.name != "three.module.js"]
for path in targets:
    source = path.read_text()
    patched = source.replace("from 'three'", "from '/vendor/three.module.js'")
    patched = re.sub(r"from '\.\./[a-z]+/([A-Za-z0-9_]+\.js)'",
                     r"from '/vendor/\1'", patched)
    patched = re.sub(r"from '\./([A-Za-z0-9_]+\.js)'",
                     r"from '/vendor/\1'", patched)
    if patched != source:
        path.write_text(patched)

unresolved = [
    (path.name, m.group(1))
    for path in targets
    for m in re.finditer(r"from '([^']+)'", path.read_text())
    if not m.group(1).startswith("/vendor/")
]
if unresolved:
    raise SystemExit(f"vendor imports could not be rewritten: {unresolved}")
print(f"      rewrote imports in {len(targets)} vendored modules")
PYEOF
fi

echo
echo "Fetched:"
du -sh upstream data/raw web/vendor 2>/dev/null || true
echo
echo "Next: uv run python src/flylab/build_connectome.py"
