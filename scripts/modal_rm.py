"""Train the reward model (nuance.reward_model) on Modal L4s, one container per --train-frac of the learning curve.

The rated clips are shrunk to the model input on the laptop first (10 fps, 384 x 160 into data/rm-input), so the GPU
never decodes full-size video. From the repo root, with the project venv:
    .venv/bin/modal run scripts/modal_rm.py::upload
    .venv/bin/modal run --detach scripts/modal_rm.py::main --fracs 1.0,0.75,0.5,0.25 --tag rm-v1
    .venv/bin/modal volume get nuance-rm runs/rm-v1 data/runs/
"""
import modal

MODEL = "Qwen/Qwen3.5-2B"
app = modal.App("nuance-rm")
vol = modal.Volume.from_name("nuance-rm", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "torchvision", "transformers==5.21.0", "peft==0.21.2", "accelerate", "flash-linear-attention",
                      "huggingface_hub", "imageio", "imageio-ffmpeg", "pillow", "scipy", "numpy")
         .env({"HF_HOME": "/root/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\"")
         .add_local_python_source("nuance"))


@app.function(gpu="L4", image=image, volumes={"/vol": vol}, timeout=10 * 3600, memory=32768)
def train(frac: float, tag: str, epochs: int) -> None:
    import threading
    from nuance import reward_model
    stop = threading.Event()

    def commit_every_5_min():
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_every_5_min, daemon=True).start()
    try:
        reward_model.train(["--ratings", "/vol/data/ratings.db", "--batches", "/vol/data/ctx-rate-02",
                            "--out", f"/vol/runs/{tag}/frac{frac}", "--train-frac", str(frac),
                            "--epochs", str(epochs), "--batch-size", "1", "--grad-accum", "8"])
    finally:
        stop.set()
        vol.commit()


@app.local_entrypoint()
def upload() -> None:
    with vol.batch_upload(force=True) as b:
        b.put_file("data/ratings/ratings.db", "/data/ratings.db")
        for f in ("clips.jsonl", "sources.json"):
            b.put_file(f"data/batches/ctx-rate-02/{f}", f"/data/ctx-rate-02/{f}")
        b.put_directory("data/rm-input/ctx-rate-02/videos", "/data/ctx-rate-02/videos")


@app.local_entrypoint()
def main(fracs: str = "1.0,0.75,0.5,0.25", tag: str = "rm-v1", epochs: int = 4) -> None:
    # spawned calls outlive this client (run with --detach); progress is read from the volume
    for f in fracs.split(","):
        print(f, train.spawn(float(f), tag, epochs).object_id)
