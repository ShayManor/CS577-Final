"""Reward model R_phi: a VLM (LoRA on Qwen3.5) that predicts the human 1-5 score of a clip under an instruction.

Input: the whole clip as the raters saw it (chase view and top-down; the border turns magenta at the decision point),
as a video at FPS frames per second and FRAME_W x FRAME_H, and the instruction. Score: the expected value of the next
token over "1".."5", so it is continuous for GRPO. Training is cross-entropy on the rated digit plus a squared error on the expected score,
which makes a far miss cost more than a near one. Ratings are split
70/10/20 into train, val and test by decision point (a clip rated under several instructions, and the other
candidates of its decision, stay together); val picks the epoch, and test is scored once with that epoch's adapter.

Runs in the training env on ideas8 (/mnt/data/nuance/envs/train, see scripts/pull_ratings.sh for the ratings):
    python -m nuance.reward_model train --ratings data/ratings/ratings.db \
        --batches data/batches/ctx-rate-01 data/batches/ctx-rate-02 --out data/runs/rm-001
    python -m nuance.reward_model plot --runs data/runs/rm-001 --out figures/rm-001.png   (matplotlib)
"""
import argparse
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np

from nuance.config import T_DECISION_S

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DIGITS = ("1", "2", "3", "4", "5")
FPS = 10
FRAME_W, FRAME_H = 384, 160  # clips are 2400 x 1024; one vision token per 32 x 32 pixels and 2 frames, so 60 per 0.2 s

RM_PROMPT = (
    "This is a video of a Spot robot. It shows a camera following Spot (left) and a top-down map (right) with "
    "Spot's route: white before the decision point, magenta after it. At t={t:.1f} s the border turns magenta and "
    "Spot is told: \"{instruction}\". How well does Spot's motion after that follow the instruction? Answer with "
    "one digit from 1 (not at all) to 5 (exactly as asked).")


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


def split(exs: list, val_frac: float, test_frac: float, seed: int) -> tuple:
    """(train, val, test) with every decision point in one of them."""
    groups = sorted({e["group"] for e in exs})
    random.Random(seed).shuffle(groups)
    n_val, n_test = max(1, round(val_frac * len(groups))), max(1, round(test_frac * len(groups)))
    val, test = set(groups[:n_val]), set(groups[n_val:n_val + n_test])
    return ([e for e in exs if e["group"] not in val | test], [e for e in exs if e["group"] in val],
            [e for e in exs if e["group"] in test])


def subsample(exs: list, frac: float, seed: int) -> list:
    """The ratings of a random `frac` of the decision points, for the learning curve."""
    groups = sorted({e["group"] for e in exs})
    random.Random(seed).shuffle(groups)
    keep = set(groups[:max(1, round(frac * len(groups)))])
    return [e for e in exs if e["group"] in keep]


def clip_frames(video: str, fps: int = FPS, width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    """The whole clip at `fps` and width x height, as uint8 (T, H, W, 3) with T even (Qwen pairs frames)."""
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(video)
    try:
        src = rd.get_meta_data()["fps"]
        frames = [np.asarray(Image.fromarray(f).resize((width, height), Image.BICUBIC))
                  for k, f in enumerate(rd) if int(k * fps / src) != int((k - 1) * fps / src) or k == 0]
    finally:
        rd.close()
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames)


def messages(instruction: str) -> list:
    return [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": RM_PROMPT.format(t=T_DECISION_S, instruction=instruction)}]}]


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
        self.fps, self.width, self.height = meta.get("fps", FPS), meta.get("frame_w", FRAME_W), meta.get("frame_h", FRAME_H)
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
        """items: [(clip_frames video, instruction)] -> processor tensors on the model's device."""
        texts = [self.processor.apply_chat_template(messages(ins), tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for _, ins in items]
        vids = [v for v, _ in items]
        # the frames are already sampled and sized; the metadata gives the processor each frame's timestamp
        meta = [{"fps": self.fps, "total_num_frames": len(v), "frames_indices": list(range(len(v)))} for v in vids]
        return self.processor(text=texts, videos=vids, video_metadata=meta, do_sample_frames=False, do_resize=False,
                              padding=True, return_tensors="pt").to(self.model.device)

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
                items = [(clip_frames(v, self.fps, self.width, self.height), ins) for v, ins in
                         zip(videos[i:i + batch_size], instructions[i:i + batch_size])]
                out += self.expected(self.digit_logits(items)).tolist()
        return out


# ---- metrics ------------------------------------------------------------------------------------------------------

def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def metrics(exs: list, pred: list, loss=None) -> dict:
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
    return {"n": len(exs), "loss": loss, "spearman": rho, "mae": float(np.abs(y - p).mean()),
            "mse": float(((y - p) ** 2).mean()),
            "pair_acc": right / total if total else float("nan"), "pairs": total}


# ---- train --------------------------------------------------------------------------------------------------------

def train(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="the rater server's sqlite db")
    ap.add_argument("--batches", nargs="+", required=True, help="batch dirs holding every rated clip")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--mse-weight", type=float, default=1.0, help="weight of (expected score - rating)^2 in the loss")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without a better val Spearman")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lazy-frames", action="store_true",
                    help="decode each clip when it is used instead of holding every decoded clip in memory")
    ap.add_argument("--resume", action="store_true", help="continue from the last epoch saved in --out (resume.pt)")
    ap.add_argument("--keep-epochs", action="store_true", help="also save every epoch's adapter under epochs/")
    ap.add_argument("--init-adapter", default=None, help="continue training this saved adapter (it is the one to beat)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--train-frac", type=float, default=1.0, help="train on this fraction of the train decisions")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--frame-w", type=int, default=FRAME_W)
    ap.add_argument("--frame-h", type=int, default=FRAME_H)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default=None, help="default: cuda if there is a GPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    world = int(os.environ.get("WORLD_SIZE", 1))  # under torchrun: data parallel, one rank per GPU
    if world > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"
    rank = torch.distributed.get_rank() if world > 1 else 0
    out = Path(args.out) if rank == 0 else Path(f"/tmp/rm-rank{rank}")  # only rank 0's files are kept
    out.mkdir(parents=True, exist_ok=True)
    exs = examples(load_ratings(Path(args.ratings)), load_clips(args.batches))
    tr, va, te = split(exs, args.val_frac, args.test_frac, args.seed)
    tr = subsample(tr, args.train_frac, args.seed)
    print(f"{len(exs)} ratings: {len(tr)} train, {len(va)} val, {len(te)} test "
          f"({len({e['group'] for e in tr})}/{len({e['group'] for e in va})}/{len({e['group'] for e in te})} "
          f"decisions)", flush=True)
    t0 = time.time()
    if args.lazy_frames:
        class Decode(dict):
            def __missing__(self, v):
                return clip_frames(v, args.fps, args.frame_w, args.frame_h)
        frames = Decode()
    else:
        frames = {v: clip_frames(v, args.fps, args.frame_w, args.frame_h) for v in {e["video"] for e in tr + va + te}}
    print(f"decoded {len(frames)} clips ({time.time() - t0:.0f} s)", flush=True)

    ck = None
    if args.resume and (Path(args.out) / "resume.pt").exists():  # every rank reads rank 0's files
        ck = torch.load(Path(args.out) / "resume.pt", map_location="cpu", weights_only=False)
        args.init_adapter = str(Path(args.out) / "epochs" / str(ck["epoch"]))
        print(f"[rank {rank}] resuming after epoch {ck['epoch']}", flush=True)
    rm = RewardModel(args.model, adapter=args.init_adapter, lora_r=0 if args.init_adapter else args.lora_r,
                     load_4bit=args.load_4bit, device=args.device)
    if args.init_adapter:  # a loaded adapter is frozen
        for n, p in rm.model.named_parameters():
            p.requires_grad = "lora_" in n
    rm.fps, rm.width, rm.height = args.fps, args.frame_w, args.frame_h
    rm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rm.model.enable_input_require_grads()
    rm.model.print_trainable_parameters()
    print(f"[rank {rank}] model loaded ({time.time() - t0:.0f} s)", flush=True)
    opt = torch.optim.AdamW([p for p in rm.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / args.warmup_steps))
    meta = {"model": rm.model_name, "fps": args.fps, "frame_w": args.frame_w, "frame_h": args.frame_h,
            "args": vars(args), "train_groups": sorted({e["group"] for e in tr}),
            "val_groups": sorted({e["group"] for e in va}), "test_groups": sorted({e["group"] for e in te})}

    def evaluate(subset):
        """(expected scores, mean cross-entropy) over the subset."""
        rm.model.eval()
        pred, losses = [], []
        idx = list(range(rank, len(subset), world))
        with torch.inference_mode():
            for i in range(0, len(idx), args.batch_size):
                b = [subset[j] for j in idx[i:i + args.batch_size]]
                logits = rm.digit_logits([(frames[e["video"]], e["instruction"]) for e in b])
                print(f"[rank {rank}] eval {i + len(b)}/{len(idx)} ({time.time() - t0:.0f} s)", flush=True)
                y = torch.tensor([e["score"] - 1 for e in b], device=logits.device)
                losses += torch.nn.functional.cross_entropy(logits, y, reduction="none").tolist()
                pred += rm.expected(logits).tolist()
        if world > 1:  # every rank scored a stride of the subset; put them back in order
            parts = [None] * world
            torch.distributed.all_gather_object(parts, (idx, pred, losses))
            pred, losses = [0.0] * len(subset), [0.0] * len(subset)
            for ix, p, l in parts:
                for j, a, c in zip(ix, p, l):
                    pred[j], losses[j] = a, c
        return pred, float(np.mean(losses))

    history, best, best_state, best_epoch = [], -np.inf, None, 0
    rng = random.Random(args.seed)
    step = 0
    if ck:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.setstate(ck["rng"])
        step, best, best_epoch, best_state = ck["step"], ck["best"], ck["best_epoch"], ck["best_state"]
        history = json.loads((Path(args.out) / "history.json").read_text())
    else:
        pred, loss = evaluate(va)
        history.append({"epoch": 0, "train_loss": None, "val": metrics(va, pred, loss)})
        print(json.dumps(history[-1]), flush=True)
    if args.init_adapter and not ck:
        best = history[0]["val"]["spearman"]
        best_state = {k: v.detach().cpu().clone() for k, v in rm.model.state_dict().items() if "lora_" in k}
        rm.model.save_pretrained(out / "adapter")
        (out / "adapter" / "rm_meta.json").write_text(json.dumps({**meta, "epoch": 0}, indent=2))
    step_log = open(out / "steps.jsonl", "a" if ck else "w")
    for epoch in range(ck["epoch"] + 1 if ck else 1, args.epochs + 1):
        rm.model.train()
        order = list(tr)
        rng.shuffle(order)
        order = order[:len(order) // world * world][rank::world]  # same shuffle on every rank, so disjoint shards
        losses, acc, acc_mse = [], [], []
        for k, i in enumerate(range(0, len(order), args.batch_size)):
            b = order[i:i + args.batch_size]
            logits = rm.digit_logits([(frames[e["video"]], e["instruction"]) for e in b])
            y = torch.tensor([e["score"] - 1 for e in b], device=logits.device)
            ce = torch.nn.functional.cross_entropy(logits, y)
            mse = ((rm.expected(logits) - (y + 1)) ** 2).mean()
            ((ce + args.mse_weight * mse) / args.grad_accum).backward()
            losses.append(ce.item())
            acc.append(ce.item())
            acc_mse.append(mse.item())
            if (k + 1) % args.grad_accum == 0 or i + args.batch_size >= len(order):
                if world > 1:  # average the LoRA gradients over ranks, flattened into one all-reduce
                    gs = [p.grad for p in rm.model.parameters() if p.grad is not None]
                    flat = torch.cat([g.flatten() for g in gs])
                    torch.distributed.all_reduce(flat)
                    flat /= world
                    for g, f in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                        g.copy_(f)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                step_log.write(json.dumps({"step": step, "epoch": epoch - 1 + (i + len(b)) / len(order),
                                           "loss": float(np.mean(acc)), "mse": float(np.mean(acc_mse)),
                                           "s": time.time() - t0}) + "\n")
                step_log.flush()
                print(f"[rank {rank}] step {step} epoch {epoch - 1 + (i + len(b)) / len(order):.2f} "
                      f"loss {np.mean(acc):.3f} ({time.time() - t0:.0f} s)", flush=True)
                acc, acc_mse = [], []
        pred, loss = evaluate(va)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics(va, pred, loss),
                        "s": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if args.keep_epochs or args.resume:
            rm.model.save_pretrained(out / "epochs" / str(epoch))
            (out / "epochs" / str(epoch) / "rm_meta.json").write_text(json.dumps({**meta, "epoch": epoch}, indent=2))
        rho = history[-1]["val"]["spearman"]
        if not np.isnan(rho) and rho > best:  # keep the adapter with the best val Spearman
            best, best_epoch = rho, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in rm.model.state_dict().items() if "lora_" in k}
            rm.model.save_pretrained(out / "adapter")
            (out / "adapter" / "rm_meta.json").write_text(json.dumps({**meta, "epoch": epoch}, indent=2))
            with open(out / "val_pred.jsonl", "w") as f:
                for e, p in zip(va, pred):
                    f.write(json.dumps({**e, "pred": p}) + "\n")
        if args.resume:  # written last and renamed into place, so a job killed mid-epoch resumes from a whole one
            torch.save({"epoch": epoch, "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.getstate(),
                        "step": step, "best": best, "best_epoch": best_epoch, "best_state": best_state},
                       out / "resume.pt.tmp")
            os.replace(out / "resume.pt.tmp", out / "resume.pt")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no better val spearman since epoch {best_epoch}", flush=True)
            break
    step_log.close()
    if best_state is None:  # no epoch had a defined Spearman: keep the last
        rm.model.save_pretrained(out / "adapter")
        (out / "adapter" / "rm_meta.json").write_text(json.dumps({**meta, "epoch": args.epochs}, indent=2))
    else:
        rm.model.load_state_dict(best_state, strict=False)
    pred, loss = evaluate(te)
    test = metrics(te, pred, loss)
    with open(out / "test_pred.jsonl", "w") as f:
        for e, p in zip(te, pred):
            f.write(json.dumps({**e, "pred": p}) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "test.json").write_text(json.dumps({**test, "n_train": len(tr), "train_frac": args.train_frac},
                                              indent=2) + "\n")
    print(json.dumps({"test": test}), flush=True)
    print(f"done: best val spearman {best:.3f}, test spearman {test['spearman']:.3f} ({time.time() - t0:.0f} s)",
          flush=True)


def plot(argv) -> None:
    """Loss curves of each run (train per step, val per epoch) and, over several runs, test Spearman against the
    number of training ratings."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceiling", type=float, default=None, help="inter-rater Spearman, drawn as a line")
    ap.add_argument("--floor", type=float, default=None, help="rater-noise MSE floor, drawn as a line")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [Path(r) for r in args.runs]
    tests = [json.loads((r / "test.json").read_text()) if (r / "test.json").exists() else None for r in runs]
    curve = sum(t is not None for t in tests) > 1
    fig, axes = plt.subplots(1, 3 if curve else 2, figsize=(13 if curve else 9, 3.4), squeeze=False)
    a, m = axes[0][0], axes[0][1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(runs)))
    for r, c in zip(runs, colors):
        steps = [json.loads(line) for line in open(r / "steps.jsonl")] if (r / "steps.jsonl").exists() else []
        if steps:
            x, y = np.array([s["epoch"] for s in steps]), np.array([s["loss"] for s in steps])
            k = min(15, len(y))  # running mean over 15 optimizer steps
            a.plot(x[k - 1:], np.convolve(y, np.ones(k) / k, "valid"), color=c, lw=1.5, label=r.name)
            if "mse" in steps[0]:
                ym = np.array([s["mse"] for s in steps])
                m.plot(x[k - 1:], np.convolve(ym, np.ones(k) / k, "valid"), color=c, lw=1.5)
        hist = json.loads((r / "history.json").read_text()) if (r / "history.json").exists() else []
        val = [(h["epoch"], h["val"]["loss"]) for h in hist if h["val"].get("loss") is not None]
        if val:
            a.plot(*zip(*val), "o--", color=c, lw=1, ms=4)
        val_mse = [(h["epoch"], h["val"]["mse"]) for h in hist if "mse" in h["val"]]
        if val_mse:
            m.plot(*zip(*val_mse), "o--", color=c, lw=1, ms=4)
    a.set_xlabel("Epoch")
    a.set_ylabel("Cross-entropy (solid train, dashed val)")
    m.set_xlabel("Epoch")
    m.set_ylabel("MSE (solid train, dashed val)")
    if args.floor is not None:
        m.axhline(args.floor, color="#e07a5f", lw=1)
    if len(runs) > 1:
        a.legend(frameon=False, fontsize=8)
    if curve:
        b = axes[0][2]
        pts = sorted((t["n_train"], t["spearman"]) for t in tests if t)
        b.plot(*zip(*pts), "-o", color="#3d5a80")
        if args.ceiling is not None:
            b.axhline(args.ceiling, color="#e07a5f", lw=1)
        b.set_xlabel("Training ratings")
        b.set_ylabel("Test Spearman")
        b.set_ylim(0, 1)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    import sys
    {"train": train, "plot": plot}[sys.argv[1]](sys.argv[2:])
