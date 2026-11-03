import json
from html import escape


def html_page(figure):
    meta = figure.layout.meta
    # Embed Plotly.js, including its license, instead of depending on a CDN.
    plot = figure.to_html(full_html=False, include_plotlyjs=True, div_id='sfm-viewer',
        default_height='100%', default_width='100%',
        config={'responsive': True, 'scrollZoom': True, 'displaylogo': False,
                'modeBarButtonsToRemove': ['toImage']})
    counts = f'{meta["points_shown"]:,} / {meta["points_total"]:,} points · {meta["cameras"]} cameras'
    settings = json.dumps({'bounds': meta['bounds'], 'camera': meta['initial_camera'],
                           'views': meta['view_cameras'], 'groups': meta['trace_groups']}, ensure_ascii=False)
    up_options = ''.join('<option value="' + escape(name) + '"' +
                         (' selected' if name == meta['view_up'] else '') + '>' +
                         escape({'auto': 'Camera up', 'custom': 'Custom'}.get(name, name.upper())) +
                         '</option>' for name in meta['view_cameras'])
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SfM · Reconstruction viewer</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;background:#0b1220;color:#dde7f2;font-family:system-ui,sans-serif}
header{padding:17px 24px 12px;border-bottom:1px solid #243145;background:#101a2a}
.top{display:flex;align-items:center;gap:18px;flex-wrap:wrap}h1{font-size:19px;font-weight:650;margin:0;letter-spacing:.2px}
.counts{color:#9bb0c8;font-size:13px}.controls{display:flex;align-items:center;gap:17px;flex-wrap:wrap;margin-top:12px;font-size:13px}
label{display:flex;gap:6px;align-items:center;cursor:pointer;white-space:nowrap}input{accent-color:#56d6e6}input[type=range]{width:100px}
button{border:1px solid #344960;border-radius:6px;background:#17263a;color:#dce8f7;padding:5px 11px;cursor:pointer}button:hover{background:#253e57}
select{background:#17263a;color:#dce8f7;border:1px solid #344960;border-radius:5px;padding:4px}
.help{color:#91a7c1;font-size:12px;margin-top:10px}main{height:calc(100% - 134px);min-height:200px}
@media(max-width:720px){header{padding:12px}.controls{gap:9px}main{height:calc(100% - 178px)}}
</style></head><body>
<header id="toolbar"><div class="top"><h1>SfM · Reconstruction viewer</h1><span class="counts">''' + escape(counts) + '''</span></div>
<div class="controls">
<label><input type="checkbox" id="points" checked>Points</label>
<label><input type="checkbox" id="cameras" checked>Cameras</label>
<label><input type="checkbox" id="trajectory">Image-order links</label>
<label><input type="checkbox" id="axes">Camera XYZ axes</label>
<label>Point size <input type="range" id="point-size" min="1" max="7" step="0.5" value="2"></label>
<label title="Camera up follows image orientations; it is not measured gravity.">Up <select id="view-up">''' + up_options + '''</select></label>
<label><input type="checkbox" id="orthographic">Orthographic</label>
<button data-fit="subject">Subject</button><button data-fit="full">All points</button><button data-fit="cameras">Cameras</button>
<button id="reset-view">Reset view</button>
</div><div class="help">Drag to orbit · Scroll to zoom · Right-drag to pan · Equal scale on all axes. Up changes the view only; coordinates stay unchanged.</div></header>
<main id="viewport">''' + plot + '''</main>
<script>
const settings = ''' + settings + ''';
const viewer = document.getElementById('sfm-viewer');
function updateVisibility(){
  for(const [name,indices] of Object.entries(settings.groups)){
    const visible=document.getElementById(name).checked && (name!=='axes'||document.getElementById('cameras').checked);
    Plotly.restyle(viewer, {visible}, indices);
  }
}
for(const name of Object.keys(settings.groups))document.getElementById(name).addEventListener('change',updateVisibility);
document.getElementById('point-size').addEventListener('input',event=>Plotly.restyle(viewer,{'marker.size':Number(event.target.value)},[0]));
function resetView(){
  const camera=JSON.parse(JSON.stringify(settings.views[document.getElementById('view-up').value]));
  camera.projection={type:document.getElementById('orthographic').checked?'orthographic':'perspective'};
  return Plotly.relayout(viewer,{'scene.camera':camera});
}
document.getElementById('view-up').addEventListener('change',resetView);
document.getElementById('reset-view').addEventListener('click',resetView);
document.getElementById('orthographic').addEventListener('change',event=>
  Plotly.relayout(viewer,{'scene.camera.projection.type':event.target.checked?'orthographic':'perspective'}));
for(const button of document.querySelectorAll('[data-fit]'))button.addEventListener('click',()=>{
  const box=settings.bounds[button.dataset.fit];
  const spans=box.map(range=>range[1]-range[0]);
  const scale=Math.cbrt(spans[0]*spans[1]*spans[2]);
  Plotly.relayout(viewer,{'scene.xaxis.range':box[0],'scene.yaxis.range':box[1],
    'scene.zaxis.range':box[2],'scene.aspectmode':'manual',
    'scene.aspectratio':{x:spans[0]/scale,y:spans[1]/scale,z:spans[2]/scale}});
});
function resize(){
  document.getElementById('viewport').style.height=Math.max(200,window.innerHeight-document.getElementById('toolbar').offsetHeight)+'px';
  Plotly.Plots.resize(viewer);
}
window.addEventListener('resize',resize);
window.addEventListener('load',resize);
</script></body></html>'''
