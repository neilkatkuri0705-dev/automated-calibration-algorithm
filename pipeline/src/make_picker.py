"""
make_picker.py — generate a click-to-calibrate HTML tool for a camera.

    python make_picker.py                 # single_camera (pools the 10 videos)
    python make_picker.py <camera_name>   # a specific multi-camera video's camera

It builds a median road image for that camera and embeds it in
calibration_picker.html. Open that in a browser, click the 4 road corners
(near-left, near-right, far-right, far-left), enter lanes + dash-stripe count,
click "Build calibration", then "Download calibration.json".

Save the downloaded file as:
    calibrations/<camera>.json
(run_pipeline.py reads calibrations/<camera>.json for each camera).
Camera names:
    single_camera                      -> the pooled 10-video camera
    <video filename without extension> -> each multi-camera video
"""

import os
import sys
import base64
import numpy as np
import cv2
import config


def build_median(video_path, n=120):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1500
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return cv2.convertScaleAbs(bg, alpha=1.3, beta=12)


def videos_in(d):
    import glob
    v = []
    if os.path.isdir(d):
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            v.extend(glob.glob(os.path.join(d, ext)))
    return sorted(v)


def resolve_camera(camera):
    """Return (camera_name, representative_video_path)."""
    single = videos_in(config.VIDEO_SINGLE_DIR)
    if camera in (None, "single_camera"):
        if not single:
            raise SystemExit("no videos in single_camera folder")
        return "single_camera", single[0]
    for v in videos_in(config.VIDEO_MULTI_DIR):
        if os.path.splitext(os.path.basename(v))[0] == camera:
            return camera, v
    raise SystemExit(f"camera '{camera}' not found. Available: single_camera, "
                     + ", ".join(os.path.splitext(os.path.basename(v))[0]
                                 for v in videos_in(config.VIDEO_MULTI_DIR)))


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Calibrate __CAM__</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
#wrap{display:flex;gap:20px;flex-wrap:wrap}
canvas{border:1px solid #555;cursor:crosshair;image-rendering:pixelated}
.panel{max-width:340px}
button{background:#2a6;color:#fff;border:0;padding:8px 14px;border-radius:6px;cursor:pointer;margin:4px 0}
button.sec{background:#555}
input{width:70px;background:#222;color:#eee;border:1px solid #555;border-radius:4px;padding:4px}
.pt{padding:2px 0}
code,textarea{background:#000;color:#6f6;font-family:monospace}
textarea{width:100%;height:90px;border:1px solid #555;border-radius:6px;padding:6px}
.hint{color:#9af;font-size:13px;line-height:1.5}
</style></head><body>
<h2>Calibrate: __CAM__</h2>
<div id="wrap">
<div><canvas id="c" width="1440" height="810"></canvas></div>
<div class="panel">
<p class="hint"><b>Click 4 corners IN ORDER</b>, a rectangle on the road along the lanes:<br>
1. near-left (closest, left) &nbsp; 2. near-right (closest, right)<br>
3. far-right (up road, right) &nbsp; 4. far-left (up road, left)<br>
Near edge at the BOTTOM, far edge at the TOP &mdash; a clean trapezoid, not an hourglass.</p>
<div id="pts"></div><hr>
<p>Lanes spanned: <input id="lanes" type="number" value="1" step="1"> &times;12 = <span id="wft">12</span> ft wide</p>
<p>Dash stripes along one line: <input id="stripes" type="number" value="8" step="1"> &rarr; (n&minus;1)&times;40 = <span id="lft">280</span> ft long</p>
<button onclick="build()">Build calibration</button>
<button class="sec" onclick="reset()">Reset</button>
<button class="sec" onclick="dl()">Download __CAM__.json</button>
<hr><p class="hint">Save it as <code>calibrations/__CAM__.json</code></p>
<textarea id="out" readonly></textarea>
</div></div>
<script>
const img=new Image();img.src="data:image/png;base64,__B64__";
const c=document.getElementById("c"),ctx=c.getContext("2d");
const SCALE=3,W=480,H=270;let pts=[];
const names=["near-left","near-right","far-right","far-left"];
const colors=["#f44","#4f4","#4af","#ff4"];
img.onload=()=>draw();
function draw(){ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,W*SCALE,H*SCALE);
  pts.forEach((p,i)=>{ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(p.x*SCALE,p.y*SCALE,6,0,7);ctx.fill();
    ctx.fillStyle="#fff";ctx.font="14px sans-serif";ctx.fillText((i+1)+" "+names[i],p.x*SCALE+8,p.y*SCALE-6);});
  if(pts.length===4){ctx.strokeStyle="#0ff";ctx.lineWidth=2;ctx.beginPath();
    ctx.moveTo(pts[0].x*SCALE,pts[0].y*SCALE);for(let i=1;i<4;i++)ctx.lineTo(pts[i].x*SCALE,pts[i].y*SCALE);
    ctx.closePath();ctx.stroke();}
  document.getElementById("pts").innerHTML=pts.map((p,i)=>`<div class="pt" style="color:${colors[i]}">${i+1}. ${names[i]}: (${p.x.toFixed(0)}, ${p.y.toFixed(0)})</div>`).join("")
    +(pts.length<4?`<div class="hint">Click ${names[pts.length]}\u2026</div>`:"");}
c.addEventListener("click",e=>{if(pts.length>=4)return;const r=c.getBoundingClientRect();
  const x=(e.clientX-r.left)*(c.width/r.width)/SCALE;const y=(e.clientY-r.top)*(c.height/r.height)/SCALE;
  pts.push({x,y});draw();});
function reset(){pts=[];document.getElementById("out").value="";draw();}
document.getElementById("lanes").oninput=e=>{document.getElementById("wft").textContent=(e.target.value*12).toFixed(0);};
document.getElementById("stripes").oninput=e=>{document.getElementById("lft").textContent=((e.target.value-1)*40).toFixed(0);};
let calib=null;
function build(){if(pts.length!==4){alert("Click all 4 corners first");return;}
  const wft=+document.getElementById("lanes").value*12;const lft=(+document.getElementById("stripes").value-1)*40;
  calib={camera:"__CAM__",source_px:pts.map(p=>[+p.x.toFixed(1),+p.y.toFixed(1)]),
    order:["near_left","near_right","far_right","far_left"],width_ft:wft,length_ft:lft};
  document.getElementById("out").value=JSON.stringify(calib);}
function dl(){if(!calib)build();if(!calib)return;
  const b=new Blob([JSON.stringify(calib,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="__CAM__.json";a.click();}
</script></body></html>"""


def main():
    camera = sys.argv[1] if len(sys.argv) > 1 else "single_camera"
    cam_name, video = resolve_camera(camera)
    bg = build_median(video)
    ok, buf = cv2.imencode(".png", bg)
    b64 = base64.b64encode(buf).decode()
    html = (HTML_TEMPLATE.replace("__B64__", b64).replace("__CAM__", cam_name))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "calibration_picker.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} for camera '{cam_name}'")
    print("Open it, click 4 road corners, enter lanes + dash count, Build, "
          f"Download -> save as  calibrations/{cam_name}.json")


if __name__ == "__main__":
    main()
