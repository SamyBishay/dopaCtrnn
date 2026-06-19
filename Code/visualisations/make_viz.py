"""Build a self-contained interactive trajectory viewer.

  python make_viz.py --root results --out results/maze_viz.html

Scans results/seed*/trajectories.json, inlines everything into ONE HTML file (no server,
no fetch) so you can just open it in Firefox. Controls let you pick the experiment (seed),
the mode (training progression / eval), and the trial. Two views:
  - selected trial: the agent's path, coloured per step by w_GD (which subnetwork is in
    control: teal = habitual, crimson = goal-directed), with a w_GD-vs-step timeline;
  - overlay: every trajectory of the chosen mode, thin lines, white = earliest -> black =
    latest, on a mid-grey background.
"""
import argparse
import glob
import json
import os

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dopaCTRNN maze viewer</title>
<style>
  :root{ --bg:#1f2227; --panel:#2a2e35; --ink:#e8e8e8; --muted:#9aa0a6; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
  header{padding:14px 18px;border-bottom:1px solid #000}
  h1{font-size:16px;margin:0 0 2px} .sub{color:var(--muted);font-size:12px}
  .controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:12px 18px;background:var(--panel);border-bottom:1px solid #000}
  .controls label{color:var(--muted);font-size:12px;margin-right:6px}
  select,button{background:#3a3f48;color:var(--ink);border:1px solid #11141a;border-radius:6px;padding:6px 9px;font:13px system-ui}
  button{cursor:pointer} button:hover{background:#454b55}
  input[type=range]{vertical-align:middle;width:240px}
  .wrap{display:flex;flex-wrap:wrap;gap:18px;padding:18px;align-items:flex-start}
  .card{background:var(--panel);border:1px solid #000;border-radius:10px;padding:14px}
  .card h2{font-size:13px;margin:0 0 10px;color:var(--muted);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
  canvas{display:block;border-radius:6px}
  .legend{display:flex;gap:16px;align-items:center;margin-top:10px;font-size:12px;color:var(--muted)}
  .sw{display:inline-block;width:34px;height:10px;border-radius:3px;vertical-align:middle;margin-right:6px}
  .info{margin-top:10px;font-size:13px}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
  .ok{background:#16432a;color:#7ee2a8} .no{background:#4a1f22;color:#ff9aa0}
  .bar{height:10px;border-radius:3px;background:linear-gradient(90deg,#fff,#000);width:160px;display:inline-block;vertical-align:middle;border:1px solid #555}
</style></head>
<body>
<header><h1>dopaCTRNN &mdash; T-maze trajectory viewer</h1>
<div class="sub">path colour = w<sub>GD</sub> (subnetwork in control) &middot; overlay shades trials white&rarr;black over training</div></header>

<div class="controls">
  <span><label>experiment</label><select id="exp"></select></span>
  <span><label>mode</label><select id="mode">
    <option value="train">training (progression)</option>
    <option value="eval">eval (final policy)</option></select></span>
  <span><label>trial</label><input type="range" id="trial" min="0" max="0" value="0">
    <button id="prev">&#9664;</button><button id="next">&#9654;</button>
    <span id="trialLabel" style="margin-left:8px;color:var(--muted)"></span></span>
</div>

<div class="wrap">
  <div class="card"><h2>selected trial</h2>
    <canvas id="single"></canvas>
    <div class="legend">
      <span><span class="sw" style="background:rgb(0,150,160)"></span>habitual</span>
      <span><span class="sw" style="background:rgb(120,95,110)"></span>mixed</span>
      <span><span class="sw" style="background:rgb(200,40,60)"></span>goal-directed</span>
      <span>&#9899; start &middot; &#11044; end (filled = correct)</span>
    </div>
    <div class="info" id="info"></div>
    <canvas id="timeline" style="margin-top:12px"></canvas>
  </div>
  <div class="card"><h2>all trajectories (this mode)</h2>
    <canvas id="overlay"></canvas>
    <div class="legend"><span>earliest <span class="bar"></span> latest</span>
      <span id="ntrials"></span></div>
  </div>
</div>

<script>
const VIZ = __VIZ_JSON__;
const M = VIZ.maze, S = 74, PAD = 26;
const Wd = M.cols*S + PAD*2, Ht = M.rows*S + PAD*2;
const passSet = new Set(M.passable.map(p=>p[0]+","+p[1]));
const eq = (a,b)=>a[0]===b[0]&&a[1]===b[1];
const cx = c => PAD + c*S + S/2, cy = r => PAD + r*S + S/2;

function setup(cv,w,h){const dpr=window.devicePixelRatio||1;cv.width=w*dpr;cv.height=h*dpr;
  cv.style.width=w+"px";cv.style.height=h+"px";const x=cv.getContext("2d");x.scale(dpr,dpr);return x;}
function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function respColor(w){const g=[200,40,60],h=[0,150,160];
  return `rgb(${lerp(h[0],g[0],w)},${lerp(h[1],g[1],w)},${lerp(h[2],g[2],w)})`;}
function jit(i){let x=Math.sin((i+1)*12.9898)*43758.5453;x-=Math.floor(x);return (x-0.5)*S*0.26;}

function drawMaze(ctx,bg,corridor){
  ctx.clearRect(0,0,Wd,Ht); ctx.fillStyle=bg; ctx.fillRect(0,0,Wd,Ht);
  for(let r=0;r<M.rows;r++)for(let c=0;c<M.cols;c++){
    if(passSet.has(r+","+c)){ctx.fillStyle=corridor;
      ctx.fillRect(PAD+c*S+3,PAD+r*S+3,S-6,S-6);}
  }
  // markers
  const dot=(p,col,fill)=>{ctx.beginPath();ctx.arc(cx(p[1]),cy(p[0]),9,0,7);
    ctx.lineWidth=2.5;ctx.strokeStyle=col;if(fill){ctx.fillStyle=col;ctx.fill();}ctx.stroke();};
  ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillStyle="#777";
  ctx.fillText("L",cx(M.l_end[1]),cy(M.l_end[0])-S/2+13);
  ctx.fillText("R",cx(M.r_end[1]),cy(M.r_end[0])-S/2+13);
}

function drawSingle(traj){
  const ctx=ctxSingle; drawMaze(ctx,"#f4f3ef","#e2e0d8");
  // highlight the correct (non-match) target arm
  const tgt = traj.blocked==="L"?M.l_end:M.r_end;
  ctx.strokeStyle="#c8a23a";ctx.lineWidth=3;ctx.setLineDash([5,4]);
  ctx.strokeRect(PAD+tgt[1]*S+3,PAD+tgt[0]*S+3,S-6,S-6);ctx.setLineDash([]);
  // path coloured by w_GD per step
  ctx.lineWidth=4;ctx.lineCap="round";ctx.lineJoin="round";
  for(let i=0;i<traj.pos.length-1;i++){const a=traj.pos[i],b=traj.pos[i+1];
    if(eq(a,b))continue;
    ctx.strokeStyle=respColor(traj.w[i]!==undefined?traj.w[i]:0.5);
    ctx.beginPath();ctx.moveTo(cx(a[1]),cy(a[0]));ctx.lineTo(cx(b[1]),cy(b[0]));ctx.stroke();}
  // start / end dots
  const s=traj.pos[0],e=traj.pos[traj.pos.length-1];
  ctx.beginPath();ctx.arc(cx(s[1]),cy(s[0]),7,0,7);ctx.fillStyle="#2e7d32";ctx.fill();
  ctx.beginPath();ctx.arc(cx(e[1]),cy(e[0]),8,0,7);ctx.lineWidth=2.5;
  ctx.strokeStyle=traj.correct?"#1b5e20":"#c62828";
  if(traj.correct){ctx.fillStyle="#1b5e20";ctx.fill();}ctx.stroke();
}

function drawTimeline(traj){
  const h=92,w=Wd; const ctx=ctxTime; ctx.clearRect(0,0,w,h);
  ctx.fillStyle="#23262c";ctx.fillRect(0,0,w,h);
  const x0=PAD,x1=w-PAD,y0=16,y1=h-20,n=traj.w.length;
  ctx.strokeStyle="#444";ctx.setLineDash([3,3]);
  const ymid=y0+(y1-y0)*0.5;ctx.beginPath();ctx.moveTo(x0,ymid);ctx.lineTo(x1,ymid);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#8a8f96";ctx.font="11px system-ui";ctx.textAlign="left";
  ctx.fillText("w_GD per step (1 = goal-directed, 0 = habitual)",x0,11);
  if(n>0){const dx=(x1-x0)/Math.max(1,n-1);
    for(let i=0;i<n;i++){const x=x0+dx*i,y=y1-(y1-y0)*traj.w[i];
      ctx.fillStyle=respColor(traj.w[i]);ctx.fillRect(x-2,y-2,4,4);
      if(i>0){ctx.strokeStyle="#5b6068";ctx.lineWidth=1;ctx.beginPath();
        ctx.moveTo(x0+dx*(i-1),y1-(y1-y0)*traj.w[i-1]);ctx.lineTo(x,y);ctx.stroke();}}}
}

function drawOverlay(list){
  const ctx=ctxOver; drawMaze(ctx,"#8c9197","#7c8187");
  const n=list.length; ctx.lineWidth=1; ctx.lineCap="round";
  for(let j=0;j<n;j++){const t=list[j],g=Math.round(255*(1-(n<2?0:j/(n-1))));
    const jx=jit(j),jy=jit(j+131);
    ctx.strokeStyle=`rgba(${g},${g},${g},0.72)`;ctx.beginPath();let started=false;
    for(let i=0;i<t.pos.length;i++){const p=t.pos[i],X=cx(p[1])+jx,Y=cy(p[0])+jy;
      if(!started){ctx.moveTo(X,Y);started=true;}else ctx.lineTo(X,Y);}
    ctx.stroke();}
}

// ---- state + wiring ----
const ctxSingle=setup(document.getElementById("single"),Wd,Ht);
const ctxTime=setup(document.getElementById("timeline"),Wd,92);
const ctxOver=setup(document.getElementById("overlay"),Wd,Ht);
const expSel=document.getElementById("exp"),modeSel=document.getElementById("mode"),
      trialR=document.getElementById("trial"),trialLab=document.getElementById("trialLabel"),
      infoD=document.getElementById("info"),nD=document.getElementById("ntrials");

VIZ.experiments.forEach((e,i)=>{const o=document.createElement("option");
  o.value=i;o.textContent="seed "+e.seed+(e.untrained?" (untrained)":"");expSel.appendChild(o);});

function curList(){const e=VIZ.experiments[+expSel.value];
  const l=modeSel.value==="train"?e.train:e.eval;return l||[];}

function refreshList(){const l=curList();
  trialR.max=Math.max(0,l.length-1);if(+trialR.value>trialR.max)trialR.value=trialR.max;
  nD.textContent=l.length+" trials";drawOverlay(l);showTrial();}

function showTrial(){const l=curList();if(!l.length){infoD.textContent="(no trajectories in this mode)";
    drawMaze(ctxSingle,"#f4f3ef","#e2e0d8");ctxTime.clearRect(0,0,Wd,92);trialLab.textContent="";return;}
  const i=Math.min(+trialR.value,l.length-1),t=l[i];
  drawSingle(t);drawTimeline(t);
  const ep=t.episode!==undefined?("episode "+t.episode):("eval trial "+(i+1));
  trialLab.textContent=(i+1)+" / "+l.length;
  infoD.innerHTML=ep+" &middot; blocked arm = <b>"+t.blocked+"</b> (correct target) &middot; "+
    "<span class='pill "+(t.correct?"ok":"no")+"'>"+(t.correct?"correct":"incorrect")+"</span> &middot; "+
    t.w.length+" steps";
}

expSel.onchange=refreshList; modeSel.onchange=refreshList;
trialR.oninput=showTrial;
document.getElementById("prev").onclick=()=>{trialR.value=Math.max(0,+trialR.value-1);showTrial();};
document.getElementById("next").onclick=()=>{trialR.value=Math.min(+trialR.max,+trialR.value+1);showTrial();};
refreshList();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/maze_viz.html")
    args = ap.parse_args()

    maze, exps = None, []
    for p in sorted(glob.glob(os.path.join(args.root, "seed*", "trajectories.json"))):
        d = json.load(open(p))
        maze = maze or d.get("maze")
        exps.append({"seed": d["seed"], "untrained": d.get("untrained", False),
                     "train": d.get("train", []), "eval": d.get("eval", [])})
    if not exps:
        print("no trajectories.json found under", args.root,
              "\n(run run_experiment.py first)")
        return

    data = {"maze": maze, "experiments": exps}
    html = TEMPLATE.replace("__VIZ_JSON__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({len(exps)} experiment(s), {kb:.0f} KB)")
    print("open it in Firefox: file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
