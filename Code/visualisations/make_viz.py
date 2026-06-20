"""Build a self-contained interactive trajectory viewer.

  python make_viz.py --root results --out results/maze_viz.html

Two panels side-by-side: left = sampled LEFT arm (blocked=R),
right = sampled RIGHT arm (blocked=L).

Each panel has:
  - Step-by-step playback slider (see every step including WAITs)
  - w_GD timeline synced to the step slider
  - Heatmap overlay (visit frequency across all same-side trials)

Trivial trajectories (agent never leaves START) are excluded.
"""
import argparse
import base64
import glob
import json
import os

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dopaCTRNN maze viewer</title>
<style>
  :root{--bg:#16191f;--panel:#1e2229;--ink:#e0e0e0;--mut:#788094;
        --hab:rgb(0,172,180);--gd:rgb(218,48,56);}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font:13px/1.5 system-ui,sans-serif}
  header{padding:10px 18px;border-bottom:1px solid #000}
  h1{font-size:14px;font-weight:700;margin-bottom:2px}
  .sub{color:#707888;font-size:11px}
  .gc{display:flex;gap:14px;align-items:center;padding:8px 18px;
      background:var(--panel);border-bottom:1px solid #000;flex-wrap:wrap}
  .gc label,.gc .lbl{color:#707888;font-size:11px;margin-right:3px}
  select,button{background:#272c36;color:var(--ink);border:1px solid #0d1018;
    border-radius:5px;padding:4px 8px;font:12px system-ui;cursor:pointer}
  button:hover{background:#343a46}
  .panels{display:flex;gap:18px;padding:14px;align-items:flex-start}
  .panel{flex:1;min-width:0}
  .ph{display:flex;align-items:baseline;gap:8px;margin-bottom:9px;
      padding-bottom:6px;border-bottom:1px solid #252930}
  .ph h2{font-size:13px;font-weight:600}
  .tag{font-size:10px;padding:1px 7px;border-radius:10px;font-weight:700}
  .tagL{background:#0e2030;color:#44c8e4}.tagR{background:#280e20;color:#e44878}
  .tr-row{display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap}
  input[type=range]{vertical-align:middle}
  .rng-trial{width:130px}.rng-step{width:180px;accent-color:#5090b8}
  .tl{color:#707888;font-size:11px}
  .card{background:var(--panel);border:1px solid #0a0d14;border-radius:7px;
        padding:9px;margin-bottom:10px}
  .card h3{font-size:10px;color:#606878;font-weight:700;text-transform:uppercase;
           letter-spacing:.05em;margin-bottom:7px}
  canvas{display:block;border-radius:3px}
  .step-info{margin-top:6px;font-size:11px;color:#606878;min-height:16px}
  .pill{display:inline-block;padding:1px 7px;border-radius:999px;
        font-size:10px;font-weight:700}
  .ok{background:#0a2818;color:#44d878}.no{background:#281010;color:#f06868}
  .leg{display:flex;gap:10px;margin-top:7px;font-size:10px;
       color:#606878;align-items:center;flex-wrap:wrap}
  .sw{display:inline-block;width:22px;height:7px;border-radius:2px;
      vertical-align:middle;margin-right:3px}
  .bar{width:80px;height:7px;border-radius:2px;vertical-align:middle;
       background:linear-gradient(90deg,#ddd,#222);display:inline-block;
       border:1px solid #444;margin:0 3px}
  .ph-legend{display:flex;gap:8px;font-size:10px;color:#606878;
             margin-top:4px;flex-wrap:wrap}
  .phs{display:inline-block;width:16px;height:4px;vertical-align:middle;
       margin-right:3px;border-radius:1px}
</style></head>
<body>
<header>
  <h1>dopaCTRNN &mdash; T-maze step viewer</h1>
  <div class="sub">
    use the step slider or &#9654;/&#9646; to step through each timestep &nbsp;&middot;&nbsp;
    <span style="color:var(--gd)">&#9632; goal-directed</span>
    <span style="color:var(--hab)"> &#9632; habitual</span> &nbsp;
    (path colour = w<sub>GD</sub> during navigation; choice segment = green/red outcome)
    &nbsp;&middot;&nbsp; <span style="color:#9090b0">w<sub>GD</sub>: expression weight — gates
    W<sub>eff</sub>=f(DA)&middot;W at each step; not a learned parameter</span>
  </div>
</header>
<div id="untrainedBanner" style="display:none;background:#332200;color:#cc9900;
  padding:6px 18px;font-size:12px;font-weight:700;border-bottom:1px solid #554400">
  Untrained baseline &mdash; weights are random initialisation; no learned policy.
  Trajectories illustrate the architectural structure, not task competence.
</div>
<div class="gc">
  <span><label>experiment</label><select id="expSel"></select></span>
  <span><label>mode</label>
    <select id="modeSel">
      <option value="train">training progression</option>
      <option value="eval">eval (final policy)</option>
    </select></span>
  <span id="epochRow"><label>training window</label>
    <select id="epochSel">
      <option value="all">all trials</option>
      <option value="early">early (first 30%)</option>
      <option value="late">late (last 30%)</option>
    </select></span>
  <span id="gcInfo" class="lbl"></span>
</div>

<div class="panels">

<!-- LEFT sample panel -->
<div class="panel">
  <div class="ph">
    <h2>Sampled LEFT arm</h2><span class="tag tagL">blocked = R</span>
  </div>
  <div class="tr-row">
    <span class="lbl">trial</span>
    <input type="range" class="rng-trial" id="trialL" min="0" max="0" value="0">
    <button id="prevL">&#9664;</button><button id="nextL">&#9654;</button>
    <span id="lblL" class="tl"></span>
  </div>
  <div class="card">
    <h3>Step-by-step playback</h3>
    <canvas id="singleL"></canvas>
    <div class="tr-row" style="margin-top:7px;margin-bottom:0">
      <span class="lbl">step</span>
      <input type="range" class="rng-step" id="stepL" min="0" max="0" value="0">
      <button id="playL">&#9654;</button><button id="pauseL">&#9646;&#9646;</button>
      <span id="stpLblL" class="tl"></span>
    </div>
    <div class="step-info" id="stepInfoL"></div>
    <canvas id="timeL" style="margin-top:7px"></canvas>
    <div class="ph-legend" id="phlegL"></div>
    <div class="leg">
      <span><span class="sw" style="background:var(--hab)"></span>habitual</span>
      <span><span class="sw" style="background:#888"></span>mixed</span>
      <span><span class="sw" style="background:var(--gd)"></span>goal-directed</span>
      <span style="color:#e8c840">&#9632;</span> agent &nbsp;
      <span>&#9679; start</span>
    </div>
  </div>
  <div class="card">
    <h3>Visit-frequency heatmap (all trials)</h3>
    <canvas id="overlayL"></canvas>
    <div class="leg">low <span class="bar" style="background:linear-gradient(90deg,#1a2a3a,#44aaff)"></span> high
      <span id="nL" class="tl"></span></div>
  </div>
</div>

<!-- RIGHT sample panel -->
<div class="panel">
  <div class="ph">
    <h2>Sampled RIGHT arm</h2><span class="tag tagR">blocked = L</span>
  </div>
  <div class="tr-row">
    <span class="lbl">trial</span>
    <input type="range" class="rng-trial" id="trialR" min="0" max="0" value="0">
    <button id="prevR">&#9664;</button><button id="nextR">&#9654;</button>
    <span id="lblR" class="tl"></span>
  </div>
  <div class="card">
    <h3>Step-by-step playback</h3>
    <canvas id="singleR"></canvas>
    <div class="tr-row" style="margin-top:7px;margin-bottom:0">
      <span class="lbl">step</span>
      <input type="range" class="rng-step" id="stepR" min="0" max="0" value="0">
      <button id="playR">&#9654;</button><button id="pauseR">&#9646;&#9646;</button>
      <span id="stpLblR" class="tl"></span>
    </div>
    <div class="step-info" id="stepInfoR"></div>
    <canvas id="timeR" style="margin-top:7px"></canvas>
    <div class="ph-legend" id="phlegR"></div>
    <div class="leg">
      <span><span class="sw" style="background:var(--hab)"></span>habitual</span>
      <span><span class="sw" style="background:#888"></span>mixed</span>
      <span><span class="sw" style="background:var(--gd)"></span>goal-directed</span>
      <span style="color:#e8c840">&#9632;</span> agent &nbsp;
      <span>&#9679; start</span>
    </div>
  </div>
  <div class="card">
    <h3>Visit-frequency heatmap (all trials)</h3>
    <canvas id="overlayR"></canvas>
    <div class="leg">low <span class="bar" style="background:linear-gradient(90deg,#1a2a3a,#44aaff)"></span> high
      <span id="nR" class="tl"></span></div>
  </div>
</div>

</div><!-- .panels -->

<div id="figsWrap" style="padding:0 14px 24px;border-top:1px solid #0a0d14;margin-top:4px">
  <div style="font:700 11px system-ui;color:#606878;text-transform:uppercase;
              letter-spacing:.05em;padding:12px 0 10px">Analysis figures</div>
  <div id="figsGrid" style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start"></div>
  <div id="figsEmpty" style="font-size:12px;color:#444c5c;display:none">
    No figures found for this experiment.</div>
</div>

<script>
const VIZ = __VIZ_JSON__;
const M = VIZ.maze;

// ── geometry ──────────────────────────────────────────────────────────────
const S = 60, PAD = 30;
const MW = M.cols*S + PAD*2, MH = M.rows*S + PAD*2, TLH = 74;
const cx = c => PAD + c*S + S/2;
const cy = r => PAD + r*S + S/2;
const eq = (a,b) => a[0]===b[0] && a[1]===b[1];

function dpr(){ return window.devicePixelRatio||1; }
function setup(cv,w,h){
  const d=dpr(); cv.width=w*d; cv.height=h*d;
  cv.style.width=w+'px'; cv.style.height=h+'px';
  const x=cv.getContext('2d'); x.scale(d,d); return x;
}
function lerp(a,b,t){ return Math.round(a+(b-a)*t); }
function wCol(w){
  const h=[0,172,180],g=[218,48,56];
  return `rgb(${lerp(h[0],g[0],w)},${lerp(h[1],g[1],w)},${lerp(h[2],g[2],w)})`;
}
const PHASE_COL={pre_sample:'#5577aa',sample:'#3399dd',delay:'#cc9922',choice:null};

// ── maze background ───────────────────────────────────────────────────────
function drawMaze(ctx,opts){
  const{bg='#e8e4da',floor='#d0ccc0',blocked=null,dark=false}=opts;
  ctx.clearRect(0,0,MW,MH); ctx.fillStyle=bg; ctx.fillRect(0,0,MW,MH);
  ctx.save();
  ctx.strokeStyle=floor; ctx.lineWidth=S*0.70; ctx.lineCap='square';
  ctx.beginPath(); ctx.moveTo(cx(0),cy(0)); ctx.lineTo(cx(M.cols-1),cy(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx(2),cy(0)); ctx.lineTo(cx(2),cy(M.rows-1)); ctx.stroke();
  ctx.restore();
  if(blocked){
    ctx.save(); ctx.fillStyle='rgba(200,55,55,0.28)';
    const h2=S*0.35, c0=blocked==='L'?0:3, c1=blocked==='L'?1:4;
    ctx.fillRect(cx(c0)-h2,cy(0)-h2,cx(c1)-cx(c0)+h2*2,h2*2);
    ctx.restore();
  }
  ctx.save(); ctx.font='bold 10px system-ui'; ctx.textAlign='center';
  ctx.fillStyle=dark?'#556':'#999';
  ctx.fillText('L',cx(0),PAD-9);
  ctx.fillText('R',cx(M.cols-1),PAD-9);
  ctx.fillText('start',cx(2),cy(M.rows-1)+S*0.42+13);
  ctx.restore();
}

// ── single trial, drawn up to `step` ─────────────────────────────────────
function drawSingleAtStep(ctx,traj,step){
  drawMaze(ctx,{blocked:traj.blocked});

  // target arm dashed outline
  const tgt=traj.blocked==='L'?M.l_end:M.r_end;
  const h2=S*0.35;
  ctx.save(); ctx.strokeStyle='#c8a030'; ctx.lineWidth=2.5; ctx.setLineDash([5,4]);
  ctx.strokeRect(cx(tgt[1])-h2+1,cy(tgt[0])-h2+1,h2*2-2,h2*2-2);
  ctx.setLineDash([]); ctx.restore();

  const phases=traj.phase||null;
  const n=traj.pos.length;
  const end=Math.min(step,n-1); // draw segments 0..end-1 (segment i arrives at pos[i+1])

  ctx.save(); ctx.lineWidth=4; ctx.lineCap='round'; ctx.lineJoin='round';
  for(let i=0;i<end;i++){
    const a=traj.pos[i],b=traj.pos[i+1];
    const ph=phases?phases[i]:null;
    let col;
    if(ph){
      col=ph==='choice'?(traj.correct?'#22c860':'#ee4040'):(PHASE_COL[ph]||wCol(traj.w[i]||.5));
    } else {
      col=wCol(traj.w[i]!==undefined?traj.w[i]:.5);
    }
    if(eq(a,b)){
      // WAIT — draw a small tick mark at position
      ctx.fillStyle=col;
      ctx.fillRect(cx(a[1])-2.5,cy(a[0])-2.5,5,5);
    } else {
      ctx.strokeStyle=col;
      ctx.beginPath(); ctx.moveTo(cx(a[1]),cy(a[0])); ctx.lineTo(cx(b[1]),cy(b[0]));
      ctx.stroke();
    }
  }
  ctx.restore();

  // Start dot
  const s=traj.pos[0];
  ctx.beginPath(); ctx.arc(cx(s[1]),cy(s[0]),7,0,Math.PI*2);
  ctx.fillStyle='#2a7830'; ctx.fill();

  // End dot if we've reached it
  if(step>=n-1){
    const e=traj.pos[n-1];
    ctx.beginPath(); ctx.arc(cx(e[1]),cy(e[0]),8,0,Math.PI*2);
    ctx.lineWidth=2.5;
    ctx.strokeStyle=traj.correct?'#2a7830':'#b82020';
    if(traj.correct){ctx.fillStyle='#2a7830';ctx.fill();}
    ctx.stroke();
  }

  // Agent marker at current position (bright yellow square)
  const cur=traj.pos[Math.min(step,n-1)];
  ctx.save();
  ctx.fillStyle='#f0d840';
  ctx.shadowColor='#f0d840'; ctx.shadowBlur=8;
  ctx.fillRect(cx(cur[1])-6,cy(cur[0])-6,12,12);
  ctx.restore();
}

// ── timeline, with needle at current step ─────────────────────────────────
function drawTimeline(ctx,traj,step){
  const W=MW,H=TLH;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#181c22'; ctx.fillRect(0,0,W,H);
  const x0=PAD,x1=W-PAD,y0=14,y1=H-12,n=traj.w.length;
  if(!n) return;
  const ym=y0+(y1-y0)*.5;
  ctx.save(); ctx.strokeStyle='#333a46'; ctx.lineWidth=1; ctx.setLineDash([3,4]);
  ctx.beginPath(); ctx.moveTo(x0,ym); ctx.lineTo(x1,ym); ctx.stroke();
  ctx.setLineDash([]); ctx.restore();
  ctx.save(); ctx.fillStyle='#4a5060'; ctx.font='9px system-ui';
  ctx.textAlign='right'; ctx.fillText('1',x1,y0+4); ctx.fillText('0',x1,y1+4);
  ctx.textAlign='left'; ctx.fillStyle='#5080a0';
  ctx.fillText('w_GD (expression weight: 0=habitual, 1=goal-directed)',x0,10);
  ctx.restore();
  const dx=(x1-x0)/Math.max(1,n-1);
  // Phase bands
  if(traj.phase){
    let pi=0;
    while(pi<traj.phase.length){
      let pj=pi+1;
      while(pj<traj.phase.length&&traj.phase[pj]===traj.phase[pi]) pj++;
      const pcol=PHASE_COL[traj.phase[pi]];
      if(pcol){
        ctx.save(); ctx.globalAlpha=.10; ctx.fillStyle=pcol;
        ctx.fillRect(x0+dx*pi,y0,dx*(pj-pi),y1-y0); ctx.restore();
      }
      pi=pj;
    }
  }
  // Area fill
  ctx.save(); ctx.beginPath(); ctx.moveTo(x0,y1);
  for(let i=0;i<n;i++) ctx.lineTo(x0+dx*i,y1-(y1-y0)*traj.w[i]);
  ctx.lineTo(x0+dx*(n-1),y1); ctx.closePath();
  ctx.fillStyle='rgba(60,110,150,.15)'; ctx.fill(); ctx.restore();
  // Line + dots
  ctx.save(); ctx.beginPath();
  for(let i=0;i<n;i++){
    const x=x0+dx*i,y=y1-(y1-y0)*traj.w[i];
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.strokeStyle='#406888'; ctx.lineWidth=1.5; ctx.stroke(); ctx.restore();
  for(let i=0;i<n;i++){
    const x=x0+dx*i,y=y1-(y1-y0)*traj.w[i];
    ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2);
    ctx.fillStyle=wCol(traj.w[i]); ctx.fill();
  }
  // Needle at current step
  const ns=Math.min(step,n-1), nx=x0+dx*ns;
  ctx.save(); ctx.strokeStyle='rgba(240,220,64,.8)'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(nx,y0-2); ctx.lineTo(nx,y1+2); ctx.stroke();
  const ny=y1-(y1-y0)*traj.w[ns];
  ctx.beginPath(); ctx.arc(nx,ny,4,0,Math.PI*2);
  ctx.fillStyle='#f0dc40'; ctx.fill(); ctx.restore();
}

// ── heatmap overlay ───────────────────────────────────────────────────────
function drawHeatmap(ctx,list){
  drawMaze(ctx,{bg:'#1c2028',floor:'#2a2f3a',dark:true});
  if(!list.length) return;
  // Count visits per (row,col) summed across all trajectories
  const visits={};
  for(const t of list){
    for(const p of t.pos){
      const k=`${p[0]},${p[1]}`;
      visits[k]=(visits[k]||0)+1;
    }
  }
  const maxV=Math.max(...Object.values(visits));
  for(const[k,v] of Object.entries(visits)){
    const[r,c]=k.split(',').map(Number);
    const f=v/maxV;
    // blue tint, brighter = more visits
    ctx.save();
    ctx.fillStyle=`rgba(${lerp(20,80,f)},${lerp(40,170,f)},${lerp(60,255,f)},${0.3+f*0.65})`;
    const h2=S*0.35;
    ctx.fillRect(cx(c)-h2,cy(r)-h2,h2*2,h2*2);
    ctx.restore();
  }
  // Count label in each cell
  ctx.save(); ctx.font=`bold ${Math.max(9,Math.round(S*0.22))}px system-ui`;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  for(const[k,v] of Object.entries(visits)){
    const[r,c]=k.split(',').map(Number);
    const f=v/maxV;
    ctx.fillStyle=f>0.5?'rgba(255,255,255,.7)':'rgba(180,210,255,.6)';
    ctx.fillText(v,cx(c),cy(r));
  }
  ctx.restore();
}

// ── data helpers ──────────────────────────────────────────────────────────
function curExp(){ return VIZ.experiments[+document.getElementById('expSel').value]; }
function curMode(){ return document.getElementById('modeSel').value; }
function curEpoch(){ return document.getElementById('epochSel').value; }
function isNonTrivial(t){ const p0=t.pos[0]; return t.pos.some(p=>p[0]!==p0[0]||p[1]!==p0[1]); }
function epochFilter(list){
  const ep=curEpoch();
  if(ep==='early'){ const n=Math.max(1,Math.ceil(list.length*0.3)); return list.slice(0,n); }
  if(ep==='late'){  const n=Math.floor(list.length*0.7); return list.slice(n); }
  return list;
}
function curLists(){
  const e=curExp(),mode=curMode();
  let all=(mode==='train'?e.train:e.eval)||[];
  if(mode==='train') all=epochFilter(all);
  const nt=all.filter(isNonTrivial);
  return{L:nt.filter(t=>t.blocked==='R'), R:nt.filter(t=>t.blocked==='L')};
}

// ── canvas setup ──────────────────────────────────────────────────────────
const ctxSL=setup(document.getElementById('singleL'),MW,MH);
const ctxTL=setup(document.getElementById('timeL'),MW,TLH);
const ctxOL=setup(document.getElementById('overlayL'),MW,MH);
const ctxSR=setup(document.getElementById('singleR'),MW,MH);
const ctxTR=setup(document.getElementById('timeR'),MW,TLH);
const ctxOR=setup(document.getElementById('overlayR'),MW,MH);

// ── animation state ───────────────────────────────────────────────────────
const timers={L:null,R:null};

function stopAnim(side){ if(timers[side]){clearInterval(timers[side]);timers[side]=null;} }

// ── render ────────────────────────────────────────────────────────────────
function renderStep(side){
  const{L,R}=curLists();
  const list=side==='L'?L:R;
  const ctxS=side==='L'?ctxSL:ctxSR;
  const ctxT=side==='L'?ctxTL:ctxTR;
  const trEl=document.getElementById('trial'+side);
  const stEl=document.getElementById('step'+side);
  const stLbl=document.getElementById('stpLbl'+side);
  const stepInfo=document.getElementById('stepInfo'+side);

  if(!list.length) return;
  const i=Math.min(+trEl.value,list.length-1);
  const t=list[i];
  const step=+stEl.value;
  const n=t.pos.length;

  drawSingleAtStep(ctxS,t,step);
  drawTimeline(ctxT,t,step);

  stLbl.textContent=(step+1)+' / '+n;
  const ph=t.phase?t.phase[Math.min(step,n-2)]:'';
  const w=t.w[Math.min(step,n-1)];
  const pos=t.pos[Math.min(step,n-1)];
  const aNames=['N','S','E','W','WAIT'];
  const act=t.a?aNames[t.a[Math.min(step,n-2)]]:'';
  stepInfo.textContent=[
    ph?`phase: ${ph.replace('_',' ')}`:'',
    `w_GD: ${w!=null?w.toFixed(3):'?'}`,
    `pos: (${pos[0]},${pos[1]})`,
    act?`action: ${act}`:''
  ].filter(Boolean).join('  ·  ');
}

function showSide(side){
  stopAnim(side);
  const{L,R}=curLists();
  const list=side==='L'?L:R;
  const ctxS=side==='L'?ctxSL:ctxSR;
  const ctxT=side==='L'?ctxTL:ctxTR;
  const ctxO=side==='L'?ctxOL:ctxOR;
  const trEl=document.getElementById('trial'+side);
  const stEl=document.getElementById('step'+side);
  const lblEl=document.getElementById('lbl'+side);
  const nEl=document.getElementById('n'+side);
  const phleg=document.getElementById('phleg'+side);
  const stepInfo=document.getElementById('stepInfo'+side);

  nEl.textContent=list.length+' trials';
  drawHeatmap(ctxO,list);

  if(!list.length){
    drawMaze(ctxS,{}); ctxT.clearRect(0,0,MW,TLH);
    lblEl.textContent=''; phleg.innerHTML=''; stepInfo.textContent='(no non-trivial trials)'; return;
  }

  const i=Math.min(+trEl.value,list.length-1);
  const t=list[i];
  lblEl.textContent=(i+1)+' / '+list.length;

  stEl.max=Math.max(0,t.pos.length-1);
  stEl.value=0;

  renderStep(side);

  // Phase legend
  if(t.phase){
    const seen=[],choiceCol=t.correct?'#22c860':'#ee4040';
    ['pre_sample','sample','delay','choice'].forEach(ph=>{
      if(t.phase.includes(ph)) seen.push(ph);
    });
    phleg.innerHTML=seen.map(ph=>{
      const col=ph==='choice'?choiceCol:(PHASE_COL[ph]||'#888');
      return`<span><span class="phs" style="background:${col}"></span>${ph.replace('_',' ')}</span>`;
    }).join('');
  } else phleg.innerHTML='';
}

function refreshFigs(){
  const e=curExp();
  const grid=document.getElementById('figsGrid');
  const empty=document.getElementById('figsEmpty');
  grid.innerHTML='';
  const figs=e.figs||[];
  if(!figs.length){ empty.style.display=''; return; }
  empty.style.display='none';
  figs.forEach(fig=>{
    const wrap=document.createElement('div');
    wrap.style.cssText='display:flex;flex-direction:column;align-items:center;gap:5px';
    const lbl=document.createElement('div');
    lbl.style.cssText='font-size:10px;color:#606878;font-weight:700;text-transform:uppercase;letter-spacing:.04em';
    lbl.textContent=fig.label;
    const img=document.createElement('img');
    img.src=fig.src;
    img.style.cssText='max-width:420px;width:100%;border-radius:5px;border:1px solid #252930;background:#1e2229';
    img.title=fig.label;
    wrap.appendChild(lbl); wrap.appendChild(img);
    grid.appendChild(wrap);
  });
}

function refreshAll(){
  const isTrain=curMode()==='train';
  document.getElementById('epochRow').style.display=isTrain?'':'none';
  document.getElementById('untrainedBanner').style.display=curExp().untrained?'':'none';
  const lists=curLists();
  ['L','R'].forEach(side=>{
    const list=lists[side];
    const trEl=document.getElementById('trial'+side);
    trEl.max=Math.max(0,list.length-1);
    if(+trEl.value>+trEl.max) trEl.value=trEl.max;
  });
  showSide('L'); showSide('R');
  const epochNote=isTrain&&curEpoch()!=='all'?` [${curEpoch()}]`:'';
  document.getElementById('gcInfo').textContent=
    (lists.L.length+lists.R.length)+' non-trivial trials'+epochNote;
  refreshFigs();
}

// ── wire up ───────────────────────────────────────────────────────────────
VIZ.experiments.forEach((e,i)=>{
  const o=document.createElement('option');
  o.value=i; o.textContent='seed '+e.seed+(e.untrained?' (untrained)':'');
  document.getElementById('expSel').appendChild(o);
});
document.getElementById('expSel').onchange=()=>{
  ['L','R'].forEach(s=>{ document.getElementById('trial'+s).value=0; });
  refreshAll();
};
document.getElementById('modeSel').onchange=()=>{
  ['L','R'].forEach(s=>{ document.getElementById('trial'+s).value=0; });
  refreshAll();
};
document.getElementById('epochSel').onchange=()=>{
  ['L','R'].forEach(s=>{ document.getElementById('trial'+s).value=0; });
  refreshAll();
};

['L','R'].forEach(side=>{
  document.getElementById('trial'+side).oninput=()=>showSide(side);
  document.getElementById('prev'+side).onclick=()=>{
    const tr=document.getElementById('trial'+side);
    tr.value=Math.max(0,+tr.value-1); showSide(side);
  };
  document.getElementById('next'+side).onclick=()=>{
    const tr=document.getElementById('trial'+side);
    tr.value=Math.min(+tr.max,+tr.value+1); showSide(side);
  };
  document.getElementById('step'+side).oninput=()=>renderStep(side);

  document.getElementById('play'+side).onclick=()=>{
    stopAnim(side);
    const stEl=document.getElementById('step'+side);
    timers[side]=setInterval(()=>{
      if(+stEl.value>=+stEl.max){ stopAnim(side); return; }
      stEl.value=+stEl.value+1; renderStep(side);
    },120);
  };
  document.getElementById('pause'+side).onclick=()=>stopAnim(side);
});

refreshAll();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out",  default="results/maze_viz.html")
    args = ap.parse_args()

    FIG_SLOTS = [
        ("fig1_handoff.png",     "H1/H2 – Handoff"),
        ("fig2_devaluation.png", "H3 – Devaluation"),
        ("fig3_lesions.png",     "H4 – Lesions"),
        ("fig4_reactivation.png","H5 – Reactivation"),
        ("fig5_attractor.png",   "H6 – Attractor"),
    ]

    maze, exps = None, []
    for p in sorted(glob.glob(os.path.join(args.root, "seed*", "trajectories.json"))):
        d = json.load(open(p))
        maze = maze or d.get("maze")
        seed_dir = os.path.dirname(p)
        figs = []
        for fname, label in FIG_SLOTS:
            img_path = os.path.join(seed_dir, fname)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                figs.append({"label": label, "src": f"data:image/png;base64,{b64}"})
        exps.append({"seed": d["seed"], "untrained": d.get("untrained", False),
                     "train": d.get("train", []), "eval": d.get("eval", []),
                     "figs": figs})

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
