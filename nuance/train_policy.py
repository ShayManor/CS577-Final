"""The NUANCE policy: QLoRA on Qwen3.5-4B, an SFT warm-up on highly rated actions, then GRPO against the reward model.

The policy sees the front frame at a decision point, the instruction and its current speed, and replies with an action
(nuance.actions). In GRPO every sampled action is walked and rendered as a clip (nuance.rollout), and the reward model
(nuance.reward_model) scores the clip under the instruction. An action that does not parse, or that Spot cannot walk
(off the floor, an invalid line, a fall), gets FAIL_REWARD, below every score.

Runs in the training env on ideas8, from the repo root:
    python -m nuance.train_policy sft --ratings data/ratings/ratings.db \
        --batches data/batches/ctx-rate-01 data/batches/ctx-rate-02 --out data/runs/sft-001
    python -m nuance.train_policy grpo --sft data/runs/sft-001 --rm data/runs/rm-001/adapter \
        --batches data/batches/ctx-final-01 --out data/runs/grpo-001 [--resume]
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from nuance.actions import POLICY_PROMPT, action_of, decision_of, format_action, parse_action

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
FAIL_REWARD = 0.0
SFT_MIN_SCORE = 4.0  # mean rating at which a (clip, instruction) is an SFT target


def _prompt(r: dict, instruction: str) -> str:
    return POLICY_PROMPT.format(v0=r["start_speed"], instruction=instruction)


def _quant():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def _scenes(path) -> set:
    return set(Path(path).read_text().split()) if path else set()


def sft_rows(ratings: list, clips: dict, exclude: set) -> list:
    """One row per (clip, instruction) whose mean rating is at least SFT_MIN_SCORE: the clip's action as the reply."""
    scores = defaultdict(list)
    for x in ratings:
        scores[(x["clip_id"], x["instruction"])].append(x["score"])
    rows = []
    for (cid, ins), s in sorted(scores.items()):
        r = clips[cid]
        if np.mean(s) >= SFT_MIN_SCORE and r["scene_id"] not in exclude:
            rows.append({"prompt": [{"role": "user", "content": _prompt(r, ins)}],
                         "completion": [{"role": "assistant", "content": format_action(action_of(r))}],
                         "images": [r["front_frame"]]})
    return rows


def prompt_rows(clips: dict, exclude: set) -> list:
    """One GRPO prompt per decision point and instruction it was rated (or is to be rated) under."""
    seen, rows = set(), []
    for r in clips.values():
        key = r["group"]
        if key in seen or r["scene_id"] in exclude:
            continue
        seen.add(key)
        for ins in [r["instruction"], *r["alt_instructions"]]:
            rows.append({"prompt": [{"role": "user", "content": _prompt(r, ins)}], "image": r["front_frame"],
                         "decision": json.dumps(decision_of(r)), "instruction": ins})
    return rows


def _dataset(rows: list, image_cols: tuple):
    from datasets import Dataset, Image, Sequence
    ds = Dataset.from_list(rows)
    for c in image_cols:
        ds = ds.cast_column(c, Sequence(Image()) if c == "images" else Image())
    return ds


def sft(argv) -> None:
    from nuance.reward_model import load_clips, load_ratings
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--exclude-scenes", default=None, help="file of held-out scene ids, one per line")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="bf16 weights (smoke tests on CPU)")
    args = ap.parse_args(argv)

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    rows = sft_rows(load_ratings(Path(args.ratings)), load_clips(args.batches), _scenes(args.exclude_scenes))
    print(f"{len(rows)} SFT examples", flush=True)
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                    per_device_train_batch_size=2, gradient_accumulation_steps=4, bf16=True,
                    gradient_checkpointing=True, logging_steps=1, save_strategy="no", report_to="none",
                    completion_only_loss=True,  # Qwen3.5's template renders no thinking by default, as in grpo
                    model_init_kwargs={"quantization_config": None if args.no_4bit else _quant(),
                                       "dtype": "bfloat16"})
    peft = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                                     r"|in_proj_qkv|in_proj_z|out_proj)")
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=_dataset(rows, ("images",)), peft_config=peft)
    trainer.train()
    trainer.save_model(args.out)
    (Path(args.out) / "policy_meta.json").write_text(json.dumps({"model": args.model, "args": vars(args)}, indent=2))


class SimReward:
    """GRPO reward: walk and render each completion's action, score the clip with the reward model."""

    def __init__(self, pool, rm, out: Path, keep_groups: int):
        self.pool, self.rm, self.out, self.keep = pool, rm, out, keep_groups
        self.__name__ = "sim_reward"

    def __call__(self, prompts, completions, decision, instruction, trainer_state=None, log_metric=None, **_):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        actions = [parse_action(t) for t in texts]
        step = trainer_state.global_step if trainer_state is not None else 0
        groups = defaultdict(list)  # one rollout request per decision point and instruction
        for i, a in enumerate(actions):
            if a is not None:
                groups[(decision[i], instruction[i])].append(i)
        keys = list(groups)
        reqs = [(json.loads(d), [actions[i] for i in groups[(d, ins)]], self.out / f"step{step:05d}" / f"g{k:03d}")
                for k, (d, ins) in enumerate(keys)]
        rewards = [FAIL_REWARD] * len(texts)
        why = defaultdict(int)
        videos, idx = [], []
        for (d, ins), res in zip(keys, self.pool.run(reqs)):
            for i, x in zip(groups[(d, ins)], res["results"]):
                if x["video"] is None:
                    why[x["why"].split(":")[0]] += 1
                else:
                    videos.append(x["video"])
                    idx.append(i)
        if videos:
            for i, s in zip(idx, self.rm.score(videos, [instruction[i] for i in idx])):
                rewards[i] = float(s)
        for k in range(self.keep, len(keys)):  # keep the first groups' clips of each step to look at
            shutil.rmtree(self.out / f"step{step:05d}" / f"g{k:03d}", ignore_errors=True)
        if log_metric is not None:
            n = len(texts)
            log_metric("unparsed_frac", sum(a is None for a in actions) / n)
            log_metric("unwalkable_frac", sum(why.values()) / n)
            if idx:
                log_metric("rm_score_mean", float(np.mean([rewards[i] for i in idx])))
        with open(self.out / "rollouts.jsonl", "a") as f:
            for i, t in enumerate(texts):
                f.write(json.dumps({"step": step, "instruction": instruction[i], "completion": t,
                                    "reward": rewards[i]}) + "\n")
        return rewards


def grpo(argv) -> None:
    from nuance.reward_model import RewardModel, load_clips
    from nuance.rollout import RolloutPool
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT run dir (its LoRA adapter is trained on)")
    ap.add_argument("--rm", required=True, help="reward model adapter dir")
    ap.add_argument("--batches", nargs="+", required=True, help="batch dirs whose decision points are the prompts")
    ap.add_argument("--exclude-scenes", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=8, help="rollout workers")
    ap.add_argument("--rm-4bit", action="store_true")
    ap.add_argument("--keep-groups", type=int, default=1, help="rollout groups per step whose clips are kept")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-4bit", action="store_true", help="bf16 policy weights (smoke tests on CPU)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    out = Path(args.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    rows = prompt_rows(load_clips(args.batches), _scenes(args.exclude_scenes))
    print(f"{len(rows)} prompts", flush=True)
    base = json.loads((Path(args.sft) / "policy_meta.json").read_text())["model"]
    model = AutoModelForImageTextToText.from_pretrained(base, quantization_config=None if args.no_4bit else _quant(),
                                                        dtype=torch.bfloat16, device_map="auto")
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, args.sft, is_trainable=True)
    gen = args.prompts_per_step * args.group
    cfg = GRPOConfig(output_dir=str(out), max_steps=args.steps, learning_rate=args.lr,
                     per_device_train_batch_size=args.micro_batch, steps_per_generation=gen // args.micro_batch,
                     gradient_accumulation_steps=gen // args.micro_batch, num_generations=args.group,
                     max_completion_length=96, temperature=1.0, bf16=True, gradient_checkpointing=True,
                     logging_steps=1, save_steps=25, report_to="none", log_completions=True,
                     chat_template_kwargs={"enable_thinking": False})
    rm = RewardModel(adapter=args.rm, load_4bit=args.rm_4bit)
    pool = RolloutPool(args.workers)
    try:
        trainer = GRPOTrainer(model=model, reward_funcs=SimReward(pool, rm, out / "rollouts", args.keep_groups),
                              args=cfg, train_dataset=_dataset(rows, ("image",)),
                              processing_class=AutoProcessor.from_pretrained(base))
        trainer.train(resume_from_checkpoint=True if args.resume else None)
        trainer.save_model(str(out / "final"))
    finally:
        pool.close()


if __name__ == "__main__":
    import sys
    {"sft": sft, "grpo": grpo}[sys.argv[1]](sys.argv[2:])
