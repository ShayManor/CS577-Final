"""Download the HSSD files needed by the scenes in a scene list (stage, objects, configs), not the whole repo.

Usage: python scripts/fetch_hssd.py configs/pilot_scenes.txt
Needs a Hugging Face login with the HSSD license accepted. Safe to re-run: finished files are skipped.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

REPO = "hssd/hssd-hab"
OUT = "data/scene_datasets/hssd-hab"
TOP_LEVEL = ("hssd-hab.scene_dataset_config.json", "scene_splits.yaml", "semantics/hssd-hab_semantic_lexicon.json")


def fetch(path: str) -> str:
    # Skip files already on disk without asking the Hub, which rate-limits re-checks of thousands of files.
    if os.path.exists(os.path.join(OUT, path)):
        return os.path.join(OUT, path)
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=OUT)


def fetch_all(paths: list, label: str) -> None:
    with ThreadPoolExecutor(16) as pool:
        futures = [pool.submit(fetch, p) for p in paths]
        for k, f in enumerate(as_completed(futures), 1):
            f.result()
            if k % 200 == 0 or k == len(paths):
                print(f"{label}: {k}/{len(paths)}", flush=True)


def main(scene_file: str) -> None:
    with open(scene_file) as f:
        scenes = [line.strip() for line in f if line.strip()]
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    by_stem = {}
    for p in files:
        if p.startswith("objects/"):
            by_stem.setdefault(os.path.basename(p).split(".")[0], []).append(p)

    per_scene = [f"scenes/{s}.scene_instance.json" for s in scenes]
    per_scene += [f"semantics/scenes/{s}.semantic_config.json" for s in scenes]
    fetch_all(list(TOP_LEVEL) + [p for p in per_scene if p in set(files)], "configs")

    need = set()
    for s in scenes:
        d = json.load(open(f"{OUT}/scenes/{s}.scene_instance.json"))
        stage = d["stage_instance"]["template_name"]
        need |= {p for p in files if p.startswith(stage + ".")}
        for o in d["object_instances"]:
            stem = o["template_name"].split("/")[-1]
            need |= set(by_stem.get(stem, []))
    fetch_all(sorted(need), "assets")
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
