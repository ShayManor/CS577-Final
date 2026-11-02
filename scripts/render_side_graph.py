"""Put a speed-over-time graph in a panel left of the clip (white before the decision, magenta after) on a 512x192
canvas; the clip is scaled to 416x178 (from 384x160) and letterboxed on the right. Sampled like clip_frames."""
import json, os, sys, numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor
IN, OUT = "data/batches/ctx-rate-02/videos", "data/rm-input/ctx-rate-02/videos_side"
CW, CH, PW, VW, VH = 512, 192, 96, 416, 178
VMAX, TD, RAMP = 1.3, 2.0, 0.5
SET = {c["clip_id"]: [c["start_speed"], c["achieved_speed"]] for c in map(json.loads, open("data/batches/ctx-rate-02/clips.jsonl"))
       if os.path.exists(f"{IN}/{c['clip_id']}.mp4")}

def speed(t, s0, s1):
    return s0 + (s1 - s0) * np.clip((t - TD) / RAMP, 0, 1)

def run(c):
    s0, s1 = SET[c]
    rd = imageio.get_reader(f"{IN}/{c}.mp4"); src = rd.get_meta_data()["fps"]
    fr = [Image.fromarray(f).resize((VW, VH), Image.BICUBIC)
          for k, f in enumerate(rd) if int(k * 10 / src) != int((k - 1) * 10 / src) or k == 0]
    rd.close()
    if len(fr) % 2: fr.append(fr[-1])
    T = len(fr) / 10
    px = lambda t: (6 + t / T * (PW - 12), CH - 8 - speed(t, s0, s1) / VMAX * (CH - 16))
    w = imageio.get_writer(f"{OUT}/{c}.mp4.tmp.mp4", fps=10, codec="libx264", pixelformat="yuv444p",
                           output_params=["-crf", "12"], macro_block_size=16)
    for k, f in enumerate(fr):
        im = Image.new("RGB", (CW, CH)); im.paste(f, (PW, (CH - VH) // 2)); d = ImageDraw.Draw(im)
        t = k / 10
        d.line([px(x) for x in np.linspace(0, min(t, TD), 20)], fill=(255, 255, 255), width=3)
        if t > TD:
            d.line([px(x) for x in np.linspace(TD, t, 40)], fill=(255, 40, 220), width=3)
        w.append_data(np.asarray(im))
    w.close(); os.replace(f"{OUT}/{c}.mp4.tmp.mp4", f"{OUT}/{c}.mp4")
    return len(fr)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ids = sys.argv[1:] or sorted(SET)
    with ProcessPoolExecutor(10) as ex:
        n = list(ex.map(run, ids))
    print(len(n), min(n), max(n))
