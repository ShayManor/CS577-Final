"""Approve or reject candidate instructions in a browser.

Usage: python scripts/review_instructions.py  (then open http://localhost:5057)
Reads configs/instruction_candidates.json, saves every decision to configs/instruction_review.json.
Keys: n = approve, m = reject, s = clear, left/right arrows move between instructions.
"""
import json
import os

from flask import Flask, jsonify, request

CANDIDATES = "configs/instruction_candidates.json"
REVIEW = "configs/instruction_review.json"

app = Flask(__name__)


def load_review() -> dict:
    if not os.path.exists(REVIEW):
        return {}
    with open(REVIEW) as f:
        return json.load(f)


@app.get("/api/state")
def state():
    with open(CANDIDATES) as f:
        return jsonify(candidates=json.load(f), review=load_review())


@app.post("/api/decide")
def decide():
    body = request.get_json()
    review = load_review()
    if body["decision"] is None:
        review.pop(body["text"], None)
    else:
        review[body["text"]] = body["decision"]
    with open(REVIEW, "w") as f:
        json.dump(review, f, indent=2)
    return jsonify(ok=True)


@app.post("/api/add")
def add():
    body = request.get_json()
    with open(CANDIDATES) as f:
        candidates = json.load(f)
    if body["text"] not in {c["text"] for c in candidates}:
        candidates.append(body)
        with open(CANDIDATES, "w") as f:
            f.write("[\n" + ",\n".join("  " + json.dumps(c) for c in candidates) + "\n]\n")
    return jsonify(ok=True)


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Instruction review</title><style>
body{font:15px system-ui;max-width:760px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
.row{display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:6px;cursor:pointer}
.row.cur{outline:2px solid #4a7bd0;background:#fff}
.text{flex:1;font-size:17px}.tag{font-size:12px;color:#777;width:150px}
.approve{background:#dff3e2}.reject{background:#f6dede;color:#999;text-decoration:line-through}
button{border:1px solid #ccc;background:#fff;border-radius:5px;padding:3px 9px;cursor:pointer}
h3{margin:22px 0 6px;color:#555}#counts{color:#666}
</style></head><body>
<h2>Instruction review</h2><div id="counts"></div>
<p style="color:#888">n approve · m reject · s clear · arrows move</p>
<div><input id="new" placeholder="add instruction" size="30">
<select id="dir"><option>faster</option><option>slower</option><option>hold</option><option>absolute</option><option>clearance</option><option>goal</option></select>
<select id="size"><option>small</option><option>large</option><option>unspecified</option><option>none</option></select>
<button onclick="addNew()">add</button></div><div id="list"></div>
<script>
let cands=[],review={},cur=0;
async function load(){const s=await (await fetch('/api/state')).json();cands=s.candidates;review=s.review;render()}
function render(){
  const L=document.getElementById('list');L.innerHTML='';let last='';
  cands.forEach((c,i)=>{
    if(c.direction!==last){last=c.direction;L.insertAdjacentHTML('beforeend',`<h3>${c.direction}</h3>`)}
    const d=review[c.text]||'';
    const el=document.createElement('div');el.className=`row ${d} ${i===cur?'cur':''}`;
    el.innerHTML=`<span class="text"></span><span class="tag">${c.size}</span>
      <button data-d="approve">✓</button><button data-d="reject">✗</button>`;
    el.querySelector('.text').textContent=c.text;
    el.onclick=e=>{cur=i;if(e.target.dataset.d)decide(e.target.dataset.d);else render()};
    L.appendChild(el)});
  const v=Object.values(review);
  document.getElementById('counts').textContent=
    `${v.filter(x=>x==='approve').length} approved, ${v.filter(x=>x==='reject').length} rejected, ${cands.length-v.length} left`;
  document.querySelector('.cur')?.scrollIntoView({block:'nearest'})}
async function decide(d){
  const t=cands[cur].text;
  if(d===null)delete review[t];else review[t]=d;
  await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t,decision:d})});
  if(d&&cur<cands.length-1)cur++;render()}
async function addNew(){
  const text=document.getElementById('new').value.trim();if(!text)return;
  await fetch('/api/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text,direction:dir.value,size:size.value})});
  document.getElementById('new').value='';load()}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='n')decide('approve');else if(e.key==='m')decide('reject');else if(e.key==='s')decide(null);
  else if(e.key==='ArrowDown'||e.key==='ArrowRight'){cur=Math.min(cur+1,cands.length-1);render();e.preventDefault()}
  else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){cur=Math.max(cur-1,0);render();e.preventDefault()}});
load().then(()=>{cur=Math.max(0,cands.findIndex(c=>!review[c.text]));render()});
</script></body></html>"""

if __name__ == "__main__":
    app.run(port=5057)
