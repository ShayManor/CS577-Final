"""Write a yellow copy of habitat's hab_spot_arm robot (its meshes are teal) to data/robots/spot_yellow.

Usage: python scripts/make_yellow_spot.py
Recolors every embedded texture and material color to Spot's yellow, keeping each texel's brightness, and leaves
dark parts dark. The URDF is copied unchanged, so it points at the recolored meshes in the new folder.
"""
import io
import os
import shutil

import numpy as np
from PIL import Image
from pygltflib import GLTF2

SRC = "data/robots/hab_spot_arm"
DST = "data/robots/spot_yellow"
YELLOW = np.array([250, 200, 20], dtype=float) / 255.0
DARK_LUM = 0.12
LUM_REF = 0.45


def _recolor(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0, 1], shape (..., 3)."""
    lum = rgb.mean(axis=-1, keepdims=True)
    yellow = YELLOW * np.clip(lum / LUM_REF, 0.85, 1.0)
    return np.where(lum < DARK_LUM, rgb, yellow)


def recolor_glb(src: str, dst: str) -> None:
    g = GLTF2().load(src)
    blob = bytearray(g.binary_blob())
    for img in g.images:
        view = g.bufferViews[img.bufferView]
        data = bytes(blob[view.byteOffset or 0:(view.byteOffset or 0) + view.byteLength])
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        a = np.asarray(im).astype(float) / 255.0
        a[..., :3] = _recolor(a[..., :3])
        out = io.BytesIO()
        Image.fromarray((a * 255).astype(np.uint8)).save(out, format="PNG")
        new = out.getvalue()
        # Append the new image to the end of the buffer and repoint the view at it.
        while len(blob) % 4:
            blob.append(0)
        view.byteOffset, view.byteLength = len(blob), len(new)
        blob += new
        img.mimeType = "image/png"
    for m in g.materials:
        pbr = m.pbrMetallicRoughness
        if pbr is not None and pbr.baseColorFactor is not None and pbr.baseColorTexture is None:
            c = np.array(pbr.baseColorFactor[:3])
            pbr.baseColorFactor = [*_recolor(c[None])[0].tolist(), pbr.baseColorFactor[3]]
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(dst)


def main() -> None:
    for root, _, files in os.walk(SRC):
        out_dir = os.path.join(DST, os.path.relpath(root, SRC))
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            src, dst = os.path.join(root, name), os.path.join(out_dir, name)
            if name.endswith(".glb"):
                recolor_glb(src, dst)
            else:
                shutil.copy(src, dst)
    print("wrote", DST)


if __name__ == "__main__":
    main()
