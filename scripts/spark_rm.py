"""Train the 27B reward model on Scholar's Spark (GB10) nodes, data parallel under torchrun; launch with spark_rm.sh.

ARM picks the input: "full" trains on every instruction with the speed graph beside every clip
(scripts/render_side_graph.py); "side", "trace", "color" and "ratio" are the speed-instruction-only input tests.
    python scripts/spark_rm.py ARM OUT [reward_model train args...]"""
import faulthandler, json, os, sys
faulthandler.dump_traceback_later(120, repeat=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuance import reward_model as rm
import transformers.core_model_loading as cml

# On the GB10, copying to the GPU straight from a memory-mapped safetensors file takes ~30 s per tensor; from a
# plain in-memory copy it takes milliseconds. Clone each tensor out of the mmap before transformers moves it.
_copy = cml._materialize_copy
cml._materialize_copy = lambda tensor, device=None, dtype=None: _copy(tensor[...].clone(), device, dtype)

D = "/scratch/scholar/manors/nuance/data"
SEP = " || "
arm, out, *rest = sys.argv[1:]
S = json.load(open(f"{D}/speed_set.json"))
speed = set(S["instructions"])
split0, examples0, messages0 = rm.split, rm.examples, rm.messages


def examples(ratings, clips):
    res = []
    for e in examples0(ratings, clips):
        if arm == "full":  # every instruction, the speed graph beside every clip
            e["video"] = f"{D}/ctx-rate-02/videos_side/{e['clip_id']}.mp4"
        if arm in ("trace", "color", "side") and e["clip_id"] in S["clips"]:
            e["video"] = f"{D}/ctx-rate-02/videos_{arm}/{e['clip_id']}.mp4"
        if arm == "ratio" and e["clip_id"] in S["clips"]:
            s0, s1 = S["clips"][e["clip_id"]]
            e["instruction"] += SEP + f"{s1 / s0:.2f}"
        res.append(e)
    return res


def split(exs, *a):
    if arm == "full":
        return split0(exs, *a)
    return tuple([e for e in s if e["instruction"].split(SEP)[0] in speed] for s in split0(exs, *a))


def messages(instruction):
    ins, *r = instruction.split(SEP)
    m = messages0(ins)
    t = m[0]["content"][1]["text"]
    if arm == "trace":
        t = t.replace(" How well", " The box at the bottom left plots Spot's speed over time, in the same colors. How well")
    if arm == "color":
        t = t.replace("magenta after it.", "after it colored by Spot's speed, from blue (slow) to red (fast).")
    if arm in ("side", "full"):
        t = t.replace(" How well", " The panel on the left plots Spot's speed over time, in the same colors. How well")
    if arm == "ratio":
        t = t.replace(" How well", f" Spot's speed after the decision is {r[0]}x its speed before. How well")
    m[0]["content"][1]["text"] = t
    return m


rm.examples, rm.split, rm.messages = examples, split, messages
rm.train(["--ratings", f"{D}/ratings.db", "--batches", f"{D}/ctx-rate-02", "--out", out, "--keep-epochs", "--lazy-frames",
          "--batch-size", "1"] + (["--frame-w", "512", "--frame-h", "192"] if arm in ("side", "full") else []) + rest)
