"""Reward model R_phi: a VLM (LoRA on Qwen3.5) that predicts the human 1-5 score of a clip under an instruction.

Input: N frames of the clip as the raters saw it (chase view and top-down; the border turns magenta at the decision
point), each labelled with its time, and the instruction. Score: the expected value of the next token over "1".."5",
so it is continuous for GRPO. Training is cross-entropy on the rated digit. Ratings are split into train and val by
decision point (a clip rated under several instructions, and the other candidates of its decision, stay together).

Runs in the training env on ideas8 (/mnt/data/nuance/envs/train, see scripts/pull_ratings.sh for the ratings):
    python -m nuance.reward_model train --ratings data/ratings/ratings.db \
        --batches data/batches/ctx-rate-01 data/batches/ctx-rate-02 --out data/runs/rm-001
    python -m nuance.reward_model plot --run data/runs/rm-001 --out figures/rm-001.png   (matplotlib)
"""
import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np

from nuance.config import T_DECISION_S

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DIGITS = ("1", "2", "3", "4", "5")
N_FRAMES = 8
FRAME_W = 768  # clips are 2400 x 1024; 768 x 328 is about 240 vision tokens per frame

RM_PROMPT = (
    "These are frames from a video of a Spot robot, with its time above each frame. Each frame shows a camera "
    "following Spot (left) and a top-down map (right) with Spot's route: white before the decision point, magenta "
    "after it. At t={t:.1f} s the border turns magenta and Spot is told: \"{instruction}\". How well does Spot's "
    "motion after that follow the instruction? Answer with one digit from 1 (not at all) to 5 (exactly as asked).")


# ---- data ---------------------------------------------------------------------------------------------------------

def load_ratings(db: Path) -> list:
    with sqlite3.connect(db) as c:
        return [dict(zip(("rater", "clip_id", "instruction", "score"), row)) for row in c.execute(
            "SELECT rater, clip_id, instruction, score FROM ratings WHERE score IS NOT NULL AND flagged = 0")]


def load_clips(batches: list) -> dict:
    """{clip_id: record} over the batches' clips.jsonl, each with "video" made absolute and "group" its decision
    point (a compiled batch's sources.json maps its decisions to the batches they were rendered in)."""
    clips = {}
    for b in map(Path, batches):
        src = json.loads((b / "sources.json").read_text()) if (b / "sources.json").exists() else {}
        for line in open(b / "clips.jsonl"):
            r = json.loads(line)
            r["video"] = str(b / r["video"])
            r["front_frame"] = str(b / r["front_frame"])
            r["group"] = src.get(r["decision_id"], f"{r['batch_id']}/{r['decision_id']}")
            clips[r["clip_id"]] = r
    return clips


def examples(ratings: list, clips: dict) -> list:
    missing = {x["clip_id"] for x in ratings} - clips.keys()
    if missing:
        raise ValueError(f"{len(missing)} rated clips are in none of the batches, e.g. {sorted(missing)[:3]}")
    return [{"clip_id": x["clip_id"], "video": clips[x["clip_id"]]["video"], "instruction": x["instruction"],
             "score": int(x["score"]), "group": clips[x["clip_id"]]["group"], "rater": x["rater"]} for x in ratings]


def split(exs: list, val_frac: float, seed: int) -> tuple:
    groups = sorted({e["group"] for e in exs})
    random.Random(seed).shuffle(groups)
    val = set(groups[:max(1, round(val_frac * len(groups)))])
    return [e for e in exs if e["group"] not in val], [e for e in exs if e["group"] in val]


def clip_frames(video: str, n: int = N_FRAMES, width: int = FRAME_W) -> list:
    """n frames evenly spread over the clip, first to last, as [(t, PIL image)]."""
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(video)
    try:
        fps = rd.get_meta_data()["fps"]
        count = rd.count_frames()
        out = []
        for i in np.linspace(0, count - 1, n).round().astype(int):
            img = Image.fromarray(rd.get_data(int(i)))
            out.append((i / fps, img.resize((width, round(img.height * width / img.width)), Image.BICUBIC)))
        return out
    finally:
        rd.close()


def messages(frames: list, instruction: str) -> list:
    content = []
    for t, img in frames:
        content += [{"type": "text", "text": f"t={t:.1f} s"}, {"type": "image", "image": img}]
    content.append({"type": "text", "text": RM_PROMPT.format(t=T_DECISION_S, instruction=instruction)})
    return [{"role": "user", "content": content}]


# ---- model --------------------------------------------------------------------------------------------------------

def _lora_targets(model) -> list:
    """Every linear layer of the language model (not the vision tower or the output head), by full name."""
    import torch.nn as nn
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and "language_model" in n
            and "lm_head" not in n]


class RewardModel:
    def __init__(self, model: str = DEFAULT_MODEL, adapter=None, lora_r: int = 0, load_4bit: bool = False,
                 device=None):
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        self.torch = torch
        meta = json.loads((Path(adapter) / "rm_meta.json").read_text()) if adapter else {}
        model = meta.get("model", model)
        self.n_frames, self.width = meta.get("n_frames", N_FRAMES), meta.get("frame_w", FRAME_W)
        self.processor = AutoProcessor.from_pretrained(model)
        self.processor.tokenizer.padding_side = "left"  # the score is read at the last position
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if load_4bit else None
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, quantization_config=q,
                                                                 device_map=device)
        tok = self.processor.tokenizer
        self.digit_ids = [tok.convert_tokens_to_ids(d) for d in DIGITS]
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        elif lora_r:
            from peft import LoraConfig, get_peft_model
            if load_4bit:
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                               target_modules=_lora_targets(self.model)))
        self.model_name = model

    def inputs(self, items: list):
        """items: [(frames, instruction)] -> processor tensors on the model's device."""
        msgs = [messages(f, ins) for f, ins in items]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for m in msgs]
        imgs = [[img for _, img in f] for f, _ in items]
        return self.processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(self.model.device)

    def digit_logits(self, items: list):
        out = self.model(**self.inputs(items), logits_to_keep=1)
        return out.logits[:, -1, self.digit_ids].float()  # (B, 5)

    def expected(self, logits):
        p = logits.softmax(-1)
        return (p * self.torch.arange(1, 6, device=p.device, dtype=p.dtype)).sum(-1)

    def score(self, videos: list, instructions: list, batch_size: int = 4) -> list:
        """Expected human score (1 to 5) per (clip, instruction)."""
        self.model.eval()
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(videos), batch_size):
                items = [(clip_frames(v, self.n_frames, self.width), ins) for v, ins in
                         zip(videos[i:i + batch_size], instructions[i:i + batch_size])]
                out += self.expected(self.digit_logits(items)).tolist()
        return out


# ---- metrics ------------------------------------------------------------------------------------------------------

def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def metrics(exs: list, pred: list) -> dict:
    """Spearman and MAE against the human score, and pairwise accuracy: over pairs of ratings that a rater scored
    differently, either one clip under two instructions or two clips of one decision under one instruction, the
    fraction the model orders the same way."""
    y = np.array([e["score"] for e in exs], dtype=float)
    p = np.array(pred, dtype=float)
    rho = float(np.corrcoef(_rank(y), _rank(p))[0, 1]) if len(set(y)) > 1 and len(set(p)) > 1 else float("nan")
    right = total = 0
    for i in range(len(exs)):
        for j in range(i + 1, len(exs)):
            a, b = exs[i], exs[j]
            same = a["clip_id"] == b["clip_id"] or (a["group"] == b["group"] and a["instruction"] == b["instruction"])
            if same and a["rater"] == b["rater"] and y[i] != y[j]:
                total += 1
                right += (p[i] - p[j]) * (y[i] - y[j]) > 0
    return {"n": len(exs), "spearman": rho, "mae": float(np.abs(y - p).mean()),
            "pair_acc": right / total if total else float("nan"), "pairs": total}


# ---- train --------------------------------------------------------------------------------------------------------

def train(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="the rater server's sqlite db")
    ap.add_argument("--batches", nargs="+", required=True, help="batch dirs holding every rated clip")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--frame-w", type=int, default=FRAME_W)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default=None, help="default: cuda if there is a GPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exs = examples(load_ratings(Path(args.ratings)), load_clips(args.batches))
    tr, va = split(exs, args.val_frac, args.seed)
    print(f"{len(exs)} ratings: {len(tr)} train, {len(va)} val ({len({e['group'] for e in va})} decisions)", flush=True)
    t0 = time.time()
    frames = {v: clip_frames(v, args.n_frames, args.frame_w) for v in {e["video"] for e in exs}}
    print(f"decoded {len(frames)} clips ({time.time() - t0:.0f} s)", flush=True)

    rm = RewardModel(args.model, lora_r=args.lora_r, load_4bit=args.load_4bit, device=args.device)
    rm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rm.model.enable_input_require_grads()
    rm.model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in rm.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    meta = {"model": rm.model_name, "n_frames": args.n_frames, "frame_w": args.frame_w, "args": vars(args),
            "train_groups": sorted({e["group"] for e in tr}), "val_groups": sorted({e["group"] for e in va})}

    def evaluate(subset):
        rm.model.eval()
        pred = []
        with torch.inference_mode():
            for i in range(0, len(subset), args.batch_size):
                b = subset[i:i + args.batch_size]
                pred += rm.expected(rm.digit_logits([(frames[e["video"]], e["instruction"]) for e in b])).tolist()
        return pred

    history, best = [], -np.inf
    pred = evaluate(va)
    history.append({"epoch": 0, "train_loss": None, "val": metrics(va, pred)})
    print(json.dumps(history[-1]), flush=True)
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rm.model.train()
        order = list(tr)
        rng.shuffle(order)
        losses = []
        for k, i in enumerate(range(0, len(order), args.batch_size)):
            b = order[i:i + args.batch_size]
            logits = rm.digit_logits([(frames[e["video"]], e["instruction"]) for e in b])
            y = torch.tensor([e["score"] - 1 for e in b], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, y)
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            if (k + 1) % args.grad_accum == 0 or i + args.batch_size >= len(order):
                opt.step()
                opt.zero_grad()
        pred = evaluate(va)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics(va, pred),
                        "s": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        rho = history[-1]["val"]["spearman"]
        if not np.isnan(rho) and rho > best:  # keep the adapter with the best val Spearman
            best = rho
            rm.model.save_pretrained(out / "adapter")
            (out / "adapter" / "rm_meta.json").write_text(json.dumps({**meta, "epoch": epoch}, indent=2))
            with open(out / "val_pred.jsonl", "w") as f:
                for e, p in zip(va, pred):
                    f.write(json.dumps({**e, "pred": p}) + "\n")
    if best == -np.inf:  # no epoch had a defined Spearman: keep the last
        rm.model.save_pretrained(out / "adapter")
        (out / "adapter" / "rm_meta.json").write_text(json.dumps({**meta, "epoch": args.epochs}, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"done: best val spearman {best:.3f} ({time.time() - t0:.0f} s)", flush=True)


def plot(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(args.run)
    hist = json.loads((run / "history.json").read_text())
    val = [json.loads(line) for line in open(run / "val_pred.jsonl")] if (run / "val_pred.jsonl").exists() else []
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4))
    ep = [h["epoch"] for h in hist]
    a.plot(ep, [h["val"]["spearman"] for h in hist], "-o", color="#3d5a80", label="Spearman")
    a.plot(ep, [h["val"]["pair_acc"] for h in hist], "-o", color="#e07a5f", label="Pairwise acc.")
    a.set_xlabel("Epoch")
    a.set_ylim(-0.2, 1.0)
    a.legend(frameon=False)
    if val:
        jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(val))
        b.scatter([v["score"] + j for v, j in zip(val, jit)], [v["pred"] for v in val], s=18, color="#3d5a80")
        b.plot([1, 5], [1, 5], color="#bbbbbb", lw=1)
        b.set_xlabel("Human score")
        b.set_ylabel("Predicted")
        b.set_xlim(0.5, 5.5)
        b.set_ylim(0.5, 5.5)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    import sys
    {"train": train, "plot": plot}[sys.argv[1]](sys.argv[2:])
