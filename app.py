from flask import Flask, request, jsonify, render_template_string
import requests
import json
import uuid
import time
import base64
import random

app = Flask(__name__)

import os
COMFYUI_URL = os.environ.get("COMFYUI_URL", "")

def queue_prompt(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}).encode('utf-8')
    response = requests.post(f"{COMFYUI_URL}/prompt", data=data, headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"})
    return response.json()

def get_history(prompt_id):
    response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", headers={"ngrok-skip-browser-warning": "true"})
    return response.json()

def get_image(filename, subfolder, folder_type):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFYUI_URL}/view", params=params, headers={"ngrok-skip-browser-warning": "true"})
    return response.content

def build_workflow(prompt, negative_prompt, width, height, steps, cfg, seed):
    if seed == -1:
        seed = random.randint(0, 999999999)
    
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "z_image_turbo-Q5_K_S.gguf",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader", 
            "inputs": {
                "clip_name": "Qwen3-4B-Q5_K_M.gguf",
                "type": "lumina2",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "split_files/vae/ae.safetensors"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": negative_prompt
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "zimage_output"
            }
        }
    }
    return workflow

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Z-Image Turbo</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a0f;--bg2:#111118;--bg3:#1a1a26;--bg4:#22223a;
  --border:#2a2a3d;--border2:#3a3a55;
  --text:#e8e8f0;--text2:#9090b0;--text3:#5a5a7a;
  --accent:#6c63ff;--accent2:#4a44cc;--accent-glow:rgba(108,99,255,0.3);
  --cyan:#00d4ff;--cyan-dim:rgba(0,212,255,0.12);
  --success:#22c55e;--danger:#ef4444;--warn:#f59e0b;
}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh}
.app{display:grid;grid-template-columns:380px 1fr;grid-template-rows:52px 1fr;min-height:100vh}
.topbar{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;padding:0 20px;background:var(--bg2);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:800;letter-spacing:0.05em}
.logo-icon{width:28px;height:28px;background:var(--accent);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#fff}
.status-row{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text2)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--success)}
.dot.busy{background:var(--accent);animation:pulse 1s infinite}
.dot.error{background:var(--danger)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}

.left-panel{background:var(--bg2);border-right:1px solid var(--border);overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.right-panel{background:var(--bg);display:flex;flex-direction:column}

.card{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px}
.label{font-size:10px;font-weight:700;letter-spacing:0.15em;color:var(--text3);text-transform:uppercase;margin-bottom:8px}

textarea,input,select{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'Syne',sans-serif;font-size:13px;padding:10px 12px;outline:none;transition:border-color 0.2s,box-shadow 0.2s;resize:none}
textarea:focus,input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow)}
select option{background:var(--bg2)}
#promptInput{min-height:100px;line-height:1.5}
#negPrompt{min-height:56px;font-size:12px;color:var(--text2)}

.prompt-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.char-count{font-size:11px;color:var(--text3);font-family:'JetBrains Mono',monospace}

.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;border:1px solid var(--border2);background:transparent;color:var(--text2);border-radius:8px;padding:6px 12px;font-size:12px;font-family:'Syne',sans-serif;cursor:pointer;transition:all 0.15s;white-space:nowrap}
.btn:hover{background:var(--bg4);color:var(--text)}
.btn:active{transform:scale(0.97)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff;font-size:14px;font-weight:700;padding:13px 24px;border-radius:12px;width:100%;letter-spacing:0.03em}
.btn-primary:hover{background:var(--accent2);transform:translateY(-1px);box-shadow:0 6px 24px var(--accent-glow)}
.btn-primary:active{transform:translateY(0)}
.btn-primary:disabled{opacity:0.45;cursor:not-allowed;transform:none;box-shadow:none}
.btn-sm{padding:4px 10px;font-size:11px}
.btn-danger{color:var(--danger);border-color:rgba(239,68,68,0.25)}
.btn-danger:hover{background:rgba(239,68,68,0.08)}
.btn-cyan{color:var(--cyan);border-color:rgba(0,212,255,0.25)}
.btn-cyan:hover{background:var(--cyan-dim);color:var(--cyan)}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}

.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.chip{background:var(--bg);border:1px solid var(--border);border-radius:20px;padding:4px 10px;font-size:11px;color:var(--text2);cursor:pointer;transition:all 0.15s}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.chip.active{background:var(--accent-glow);border-color:var(--accent);color:var(--accent)}

.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.res-btn{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px;cursor:pointer;transition:all 0.15s;text-align:center}
.res-btn:hover{border-color:var(--border2)}
.res-btn.active{border-color:var(--accent);background:var(--accent-glow)}
.res-name{font-size:12px;font-weight:700;display:block}
.res-sub{font-size:10px;color:var(--text3);font-family:'JetBrains Mono',monospace}

.slider-row{margin-top:10px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:6px}
.slider-name{font-size:12px;color:var(--text2)}
.slider-val{font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--accent)}
input[type=range]{height:4px;-webkit-appearance:none;appearance:none;background:var(--border2);border-radius:2px;outline:none;padding:0;border:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);border:2px solid var(--bg);cursor:pointer;transition:box-shadow 0.2s}
input[type=range]::-webkit-slider-thumb:hover{box-shadow:0 0 0 4px var(--accent-glow)}

.seed-row{display:flex;gap:8px;margin-top:10px}
.seed-row input{flex:1;font-family:'JetBrains Mono',monospace}

.output-top{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border)}
.output-title{font-size:13px;font-weight:700}
.output-actions{display:flex;gap:8px}

.preview-area{flex:1;display:flex;align-items:center;justify-content:center;padding:24px;min-height:0}
.preview-box{width:100%;max-width:560px;aspect-ratio:1;background:var(--bg2);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative;overflow:hidden;transition:aspect-ratio 0.3s}
.placeholder-inner{text-align:center;color:var(--text3)}
.placeholder-inner svg{margin-bottom:12px;opacity:0.35}
.placeholder-inner p{font-size:13px}
.placeholder-inner span{font-size:11px;font-family:'JetBrains Mono',monospace;display:block;margin-top:4px}
#outputImg{width:100%;height:100%;object-fit:contain;display:none;border-radius:8px}
.skeleton{width:100%;height:100%;background:linear-gradient(90deg,var(--bg2) 25%,var(--bg3) 50%,var(--bg2) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;display:none}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.progress-bar{position:absolute;bottom:0;left:0;right:0;height:3px;background:var(--border);display:none}
.progress-fill{height:100%;background:var(--accent);border-radius:2px;transition:width 0.5s;width:0}
.seed-badge{position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.75);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text2);display:none;gap:8px;align-items:center;backdrop-filter:blur(4px)}
.seed-badge.show{display:flex}
.copy-seed{cursor:pointer;color:var(--accent);font-size:10px}
.copy-seed:hover{text-decoration:underline}

.history-area{border-top:1px solid var(--border);padding:14px 20px}
.history-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.history-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}
.h-item{aspect-ratio:1;background:var(--bg3);border:1px solid var(--border);border-radius:8px;cursor:pointer;overflow:hidden;transition:all 0.15s;position:relative}
.h-item:hover{border-color:var(--accent);transform:scale(1.04)}
.h-item img{width:100%;height:100%;object-fit:cover}
.h-empty{font-size:11px;color:var(--text3);grid-column:1/-1;text-align:center;padding:6px}

.error-toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(80px);background:#1a0a0a;border:1px solid var(--danger);border-radius:10px;padding:10px 18px;font-size:13px;color:var(--danger);z-index:1000;opacity:0;transition:all 0.3s;pointer-events:none}
.error-toast.show{transform:translateX(-50%) translateY(0);opacity:1}
.notif{position:fixed;bottom:20px;right:20px;background:var(--bg3);border:1px solid var(--border2);border-radius:10px;padding:10px 16px;font-size:13px;color:var(--text);z-index:1000;transform:translateY(80px);opacity:0;transition:all 0.3s;pointer-events:none}
.notif.show{transform:translateY(0);opacity:1}

@media(max-width:768px){
  .app{grid-template-columns:1fr;grid-template-rows:52px auto 1fr}
  .left-panel{border-right:none;border-bottom:1px solid var(--border)}
  .history-grid{grid-template-columns:repeat(4,1fr)}
}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="logo">
      <div class="logo-icon">Z</div>
      Z-Image Turbo
    </div>
    <div class="status-row">
      <div class="dot" id="statusDot"></div>
      <span id="statusText">Ready</span>
    </div>
  </div>

  <div class="left-panel">
    <div class="card">
      <div class="label">Preset Template</div>
      <select id="presetSelect" onchange="applyPreset()">
        <option value="">— Select preset —</option>
        <option value="warrior">Epic Warrior Portrait</option>
        <option value="sniper">Elite Sniper Character</option>
        <option value="cyberpunk">Cyberpunk Mercenary</option>
        <option value="mage">Dark Mage Sorceress</option>
        <option value="assassin">Hooded Assassin</option>
        <option value="dark_forest">Dark Forest Wallpaper</option>
        <option value="space">Deep Space Wallpaper</option>
        <option value="cyberpunk_city">Cyberpunk City Wallpaper</option>
        <option value="mc_horror">Minecraft Horror Thumbnail</option>
        <option value="mc_speedrun">Minecraft Speedrun Thumbnail</option>
      </select>
    </div>

    <div class="card">
      <div class="prompt-header">
        <div class="label" style="margin-bottom:0">Prompt</div>
        <div style="display:flex;gap:6px;align-items:center">
          <span class="char-count" id="charCount">0/500</span>
          <button class="btn btn-sm btn-danger" onclick="clearPrompt()">Clear</button>
        </div>
      </div>
      <textarea id="promptInput" placeholder="Describe your image..." oninput="updateCount()"></textarea>
      <div style="margin-top:10px"><div class="label">Style Tags</div>
        <div class="chips" id="chips"></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-cyan" onclick="enhancePrompt()">✦ Enhance</button>
        <button class="btn btn-sm" onclick="copyPrompt()">Copy</button>
      </div>
    </div>

    <div class="card">
      <div class="label">Negative Prompt</div>
      <textarea id="negPrompt" placeholder="What to avoid: blurry, watermark, ugly...">blurry, low quality, distorted, watermark, text, ugly, deformed, bad anatomy</textarea>
    </div>

    <div class="card">
      <div class="label">Resolution</div>
      <div class="res-grid" id="resGrid"></div>
    </div>

    <div class="card">
      <div class="label">Parameters</div>
      <div class="slider-row">
        <div class="slider-head"><span class="slider-name">Steps</span><span class="slider-val" id="stepsVal">9</span></div>
        <input type="range" min="4" max="15" value="9" step="1" id="stepsSlider" oninput="document.getElementById('stepsVal').textContent=this.value">
      </div>
      <div class="slider-row">
        <div class="slider-head"><span class="slider-name">CFG Scale</span><span class="slider-val" id="cfgVal">1.0</span></div>
        <input type="range" min="1.0" max="7.0" value="1.0" step="0.1" id="cfgSlider" oninput="document.getElementById('cfgVal').textContent=parseFloat(this.value).toFixed(1)">
      </div>
      <div class="seed-row">
        <input type="number" id="seedInput" placeholder="Seed" value="-1" min="-1" max="999999999">
        <button class="btn" onclick="randomSeed()">🎲</button>
      </div>
    </div>

    <button class="btn btn-primary" id="genBtn" onclick="generate()">⚡ Generate</button>
  </div>

  <div class="right-panel">
    <div class="output-top">
      <div class="output-title">Output</div>
      <div class="output-actions">
        <button class="btn btn-sm" id="dlBtn" style="display:none" onclick="downloadImg()">↓ Download</button>
      </div>
    </div>
    <div class="preview-area">
      <div class="preview-box" id="previewBox">
        <div class="placeholder-inner" id="placeholder">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none"><rect x="4" y="8" width="40" height="32" rx="4" stroke="currentColor" stroke-width="1.5"/><circle cx="17" cy="20" r="4" stroke="currentColor" stroke-width="1.5"/><path d="M4 32l10-8 8 6 8-10 14 12" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>
          <p>Image will appear here</p>
          <span id="placeholderSub">Configure and click Generate</span>
        </div>
        <div class="skeleton" id="skeleton"></div>
        <img id="outputImg" alt="Generated">
        <div class="progress-bar" id="progressBar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="seed-badge" id="seedBadge">Seed: <span id="seedUsed">—</span><span class="copy-seed" onclick="copySeed()">copy</span></div>
      </div>
    </div>
    <div class="history-area">
      <div class="history-header">
        <div class="label" style="margin-bottom:0">History</div>
        <button class="btn btn-sm btn-danger" onclick="clearHistory()">Clear</button>
      </div>
      <div class="history-grid" id="historyGrid"><div class="h-empty">No images yet</div></div>
    </div>
  </div>
</div>

<div class="error-toast" id="errorToast"></div>
<div class="notif" id="notif"></div>

<script>
const PRESETS = {
  warrior:"Epic fantasy warrior, close-up portrait, male, scarred face, intense eyes, full plate armor with glowing runes, dramatic rim lighting, dark background, photorealistic, 8k, cinematic, sharp focus, detailed skin texture",
  sniper:"Elite military sniper, mid shot, stoic expression, tactical gear, ghillie suit elements, dense forest background, golden hour lighting, photorealistic, cinematic film grain, depth of field, hyper detailed",
  cyberpunk:"Cyberpunk mercenary, full body, neon-lit alley background, chrome arm implants, glowing eyes, rain-soaked streets, blue and orange neon reflections, photorealistic, cinematic, 4k",
  mage:"Dark mage sorceress, portrait, glowing purple spell in hands, flowing black robes, ancient ruins background, volumetric light, photorealistic, fantasy, dramatic lighting, 8k",
  assassin:"Hooded assassin, mid shot, glowing daggers, moonlit rooftops, fog and shadow, dark fantasy, photorealistic, cinematic, dramatic shadows, 8k",
  dark_forest:"Dense dark eerie forest at night, single lantern on a dirt path, fog rolling through ancient trees, moonlight rays, ultra wide, photorealistic, horror atmosphere, 4k wallpaper",
  space:"Deep space nebula, ringed planet in foreground, asteroid belt, star field, volumetric gas clouds, orange and blue tones, ultra wide 16:9, photorealistic, 8k wallpaper",
  cyberpunk_city:"Futuristic cyberpunk megacity at night, aerial view, neon signs everywhere, rain-soaked streets, flying cars with light trails, massive skyscrapers, ultra wide, photorealistic, blade runner aesthetic",
  mc_horror:"Minecraft horror scene, dark cave, single torch, creeper shadow on wall, scared player character, dramatic lighting, YouTube thumbnail, 16:9, high contrast",
  mc_speedrun:"Minecraft player sprinting toward glowing End portal, ender dragon looming above, dramatic speed blur, player glancing back at camera, YouTube thumbnail 16:9, bold saturated colors"
};

const STYLE_TAGS = ['photorealistic','cinematic','8k','sharp focus','dramatic lighting','film grain','depth of field','volumetric light','fantasy','anime','dark','high contrast'];

const RESOLUTIONS = [
  {label:"Portrait", sub:"768×1024", w:768, h:1024, ar:"768/1024"},
  {label:"Wallpaper", sub:"1280×720", w:1280, h:720, ar:"1280/720"},
  {label:"Square", sub:"1024×1024", w:1024, h:1024, ar:"1/1"},
  {label:"Thumbnail", sub:"1280×720 YT", w:1280, h:720, ar:"1280/720"},
];

let state = {res:0, history:[], generating:false, lastSeed:-1, currentImgData:null};

function init(){
  buildChips(); buildResGrid(); updateCount();
}

function buildChips(){
  const c=document.getElementById('chips');
  STYLE_TAGS.forEach(tag=>{
    const el=document.createElement('span');
    el.className='chip'; el.textContent=tag;
    el.onclick=()=>toggleChip(el,tag);
    c.appendChild(el);
  });
}

function toggleChip(el,tag){
  const ta=document.getElementById('promptInput');
  if(el.classList.contains('active')){
    el.classList.remove('active');
    ta.value=ta.value.replace(', '+tag,'').replace(tag+', ','').replace(tag,'').trim();
  } else {
    el.classList.add('active');
    ta.value=ta.value.trim()+(ta.value.trim()?', ':'')+tag;
  }
  updateCount();
}

function buildResGrid(){
  const g=document.getElementById('resGrid');
  RESOLUTIONS.forEach((r,i)=>{
    const btn=document.createElement('div');
    btn.className='res-btn'+(i===0?' active':'');
    btn.innerHTML=`<span class="res-name">${r.label}</span><span class="res-sub">${r.sub}</span>`;
    btn.onclick=()=>{
      document.querySelectorAll('.res-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active'); state.res=i;
      document.getElementById('previewBox').style.aspectRatio=r.ar;
    };
    g.appendChild(btn);
  });
}

function updateCount(){
  const v=document.getElementById('promptInput').value.length;
  const el=document.getElementById('charCount');
  el.textContent=v+'/500';
  el.style.color=v>450?'var(--warn)':'var(--text3)';
}

function applyPreset(){
  const v=document.getElementById('presetSelect').value;
  if(v&&PRESETS[v]){document.getElementById('promptInput').value=PRESETS[v];updateCount();}
}

function clearPrompt(){
  document.getElementById('promptInput').value='';
  document.getElementById('presetSelect').value='';
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  updateCount();
}

function copyPrompt(){
  const v=document.getElementById('promptInput').value;
  if(!v){notif('Nothing to copy!');return;}
  navigator.clipboard.writeText(v).then(()=>notif('Prompt copied!'));
}

function copySeed(){
  navigator.clipboard.writeText(state.lastSeed).then(()=>notif('Seed copied!'));
}

function randomSeed(){
  document.getElementById('seedInput').value=Math.floor(Math.random()*999999999);
}

function enhancePrompt(){
  const ta=document.getElementById('promptInput');
  if(!ta.value.trim()){notif('Enter a prompt first!');return;}
  ta.value=ta.value.trim()+', photorealistic, 8k resolution, cinematic lighting, sharp focus, hyper detailed, film grain';
  updateCount(); notif('✦ Prompt enhanced!');
}

function downloadImg(){
  if(!state.currentImgData)return;
  const a=document.createElement('a');
  a.href='data:image/png;base64,'+state.currentImgData;
  a.download='zimage_'+state.lastSeed+'.png'; a.click();
  notif('Downloading...');
}

async function generate(){
  if(state.generating)return;
  const prompt=document.getElementById('promptInput').value.trim();
  if(!prompt){notif('Enter a prompt first!');return;}

  state.generating=true;
  setStatus('busy','Generating...');

  const btn=document.getElementById('genBtn');
  btn.disabled=true; btn.textContent='⟳ Generating...';

  document.getElementById('placeholder').style.display='none';
  document.getElementById('outputImg').style.display='none';
  document.getElementById('skeleton').style.display='block';
  document.getElementById('progressBar').style.display='block';
  document.getElementById('seedBadge').classList.remove('show');

  let prog=0;
  const progInt=setInterval(()=>{
    prog=Math.min(prog+2,88);
    document.getElementById('progressFill').style.width=prog+'%';
  },400);

  const res=RESOLUTIONS[state.res];
  const payload={
    prompt,
    negative_prompt: document.getElementById('negPrompt').value,
    width: res.w, height: res.h,
    steps: parseInt(document.getElementById('stepsSlider').value),
    cfg: parseFloat(document.getElementById('cfgSlider').value),
    seed: parseInt(document.getElementById('seedInput').value)||(-1)
  };

  try{
    const resp=await fetch('/generate',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const data=await resp.json();

    clearInterval(progInt);
    document.getElementById('progressFill').style.width='100%';

    if(data.success){
      setTimeout(()=>{
        document.getElementById('skeleton').style.display='none';
        document.getElementById('progressBar').style.display='none';

        const img=document.getElementById('outputImg');
        img.src='data:image/png;base64,'+data.image;
        img.style.display='block';
        state.currentImgData=data.image;
        state.lastSeed=data.seed;

        document.getElementById('seedUsed').textContent=data.seed;
        document.getElementById('seedBadge').classList.add('show');
        document.getElementById('dlBtn').style.display='flex';

        addHistory('data:image/png;base64,'+data.image, data.seed);
        setStatus('ready','Ready');
        notif('Image generated! Seed: '+data.seed);
      },300);
    } else {
      throw new Error(data.error||'Generation failed');
    }
  } catch(err){
    clearInterval(progInt);
    document.getElementById('skeleton').style.display='none';
    document.getElementById('progressBar').style.display='none';
    document.getElementById('placeholder').style.display='block';
    document.getElementById('placeholderSub').textContent='Error: '+err.message;
    setStatus('error','Error');
    showError(err.message);
  }

  state.generating=false;
  btn.disabled=false; btn.textContent='⚡ Generate';
}

function addHistory(src,seed){
  state.history.unshift({src,seed});
  if(state.history.length>8)state.history.pop();
  renderHistory();
}

function renderHistory(){
  const g=document.getElementById('historyGrid');
  if(!state.history.length){g.innerHTML='<div class="h-empty">No images yet</div>';return;}
  g.innerHTML='';
  state.history.forEach((item)=>{
    const el=document.createElement('div');
    el.className='h-item';
    el.innerHTML=`<img src="${item.src}" alt="History">`;
    el.onclick=()=>{
      document.getElementById('outputImg').src=item.src;
      document.getElementById('outputImg').style.display='block';
      document.getElementById('placeholder').style.display='none';
      document.getElementById('seedUsed').textContent=item.seed;
      document.getElementById('seedBadge').classList.add('show');
      state.currentImgData=item.src.split(',')[1];
      state.lastSeed=item.seed;
    };
    g.appendChild(el);
  });
}

function clearHistory(){state.history=[];renderHistory();notif('History cleared');}

function setStatus(type,text){
  const dot=document.getElementById('statusDot');
  dot.className='dot'+(type==='busy'?' busy':type==='error'?' error':'');
  document.getElementById('statusText').textContent=text;
}

function notif(msg){
  const el=document.getElementById('notif');
  el.textContent=msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),2500);
}

function showError(msg){
  const el=document.getElementById('errorToast');
  el.textContent='Error: '+msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),4000);
}

init();
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    try:
        workflow = build_workflow(
            prompt=data['prompt'],
            negative_prompt=data.get('negative_prompt', ''),
            width=data.get('width', 768),
            height=data.get('height', 1024),
            steps=data.get('steps', 9),
            cfg=data.get('cfg', 1.0),
            seed=data.get('seed', -1)
        )
        
        result = queue_prompt(workflow)
        prompt_id = result['prompt_id']
        used_seed = workflow['7']['inputs']['seed']
        
        # Poll for result
        max_wait = 300
        waited = 0
        while waited < max_wait:
            time.sleep(2)
            waited += 2
            history = get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id]['outputs']
                for node_id, output in outputs.items():
                    if 'images' in output:
                        img_info = output['images'][0]
                        img_data = get_image(img_info['filename'], img_info['subfolder'], img_info['type'])
                        img_b64 = base64.b64encode(img_data).decode('utf-8')
                        return jsonify({'success': True, 'image': img_b64, 'seed': used_seed})
        
        return jsonify({'success': False, 'error': 'Timeout — generation took too long'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", headers={"ngrok-skip-browser-warning": "true"}, timeout=5)
        return jsonify({'status': 'ok', 'comfyui': r.status_code == 200})
    except:
        return jsonify({'status': 'ok', 'comfyui': False})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
