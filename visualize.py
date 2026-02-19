from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from graph import extract_subgraph, extract_comparison_subgraph
from ingest import graph_html_path

GROUP_STYLES = {
    "brand":      {"color": "#FF4500", "glow": "#FF4500", "size": 50},
    "post":       {"color": "#1E90FF", "glow": "#1E90FF", "size": None},
    "hashtag":    {"color": "#00E676", "glow": "#00E676", "size": 14},
    "mention":    {"color": "#FF9800", "glow": "#FF9800", "size": 18},
    "theme":      {"color": "#E040FB", "glow": "#E040FB", "size": 26},
    "month":      {"color": "#FFD600", "glow": "#FFD600", "size": 24},
    "media_type": {"color": "#26C6DA", "glow": "#26C6DA", "size": 20},
}

COMPARISON_BRAND_STYLES = {
    "brand_a":    {"color": "#FF4500", "glow": "#FF4500", "size": 50},
    "brand_b":    {"color": "#00B4FF", "glow": "#00B4FF", "size": 50},
    "brand_c":    {"color": "#00E676", "glow": "#00E676", "size": 50},
    "brand_d":    {"color": "#FFD600", "glow": "#FFD600", "size": 50},
    "post_a":     {"color": "#FF7043", "glow": "#FF7043", "size": None},
    "post_b":     {"color": "#29B6F6", "glow": "#29B6F6", "size": None},
    "post_c":     {"color": "#66BB6A", "glow": "#66BB6A", "size": None},
    "post_d":     {"color": "#FFF176", "glow": "#FFF176", "size": None},
    "hashtag":    {"color": "#00E676", "glow": "#00E676", "size": 14},
    "mention":    {"color": "#FF9800", "glow": "#FF9800", "size": 18},
    "theme":      {"color": "#E040FB", "glow": "#E040FB", "size": 26},
    "month":      {"color": "#FFD600", "glow": "#FFD600", "size": 24},
    "media_type": {"color": "#26C6DA", "glow": "#26C6DA", "size": 20},
}

EDGE_COLORS = {
    "POSTED":      "#FF450060",
    "HAS_HASHTAG": "#00E67650",
    "MENTIONS":    "#FF980060",
    "THEME":       "#E040FB60",
    "BELONGS_TO":  "#FFD60050",
    "IS_TYPE":     "#26C6DA50",
}

EDGE_WIDTHS = {
    "POSTED":      2.5,
    "THEME":       2.0,
    "MENTIONS":    1.8,
    "HAS_HASHTAG": 1.2,
    "BELONGS_TO":  1.0,
    "IS_TYPE":     1.0,
}


def generate_visualization(username: str, filters: Optional[dict] = None, output_path: Optional[Path] = None) -> Path:
    suffix   = _filter_suffix(filters)
    out      = output_path or graph_html_path(username, suffix)
    subgraph = extract_subgraph(username, filters)
    _render(subgraph, out, f"@{username} — Instagram Knowledge Graph", GROUP_STYLES)
    return out


def generate_comparison_visualization(usernames: list[str], filters: Optional[dict] = None, output_path: Optional[Path] = None) -> Path:
    label    = "_vs_".join(u.lower() for u in usernames)
    comp_dir = Path("data") / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    out      = output_path or (comp_dir / f"graph_comparison_{label}{_filter_suffix(filters)}.html")
    subgraph = extract_comparison_subgraph(usernames, filters)
    title    = " vs ".join(f"@{u}" for u in usernames) + " — Comparison Graph"
    _render(subgraph, out, title, COMPARISON_BRAND_STYLES, is_comparison=True, usernames=usernames)
    return out


def _filter_suffix(filters: Optional[dict]) -> str:
    if not filters:
        return ""
    parts = []
    if "month"      in filters: parts.append(filters["month"])
    if "media_type" in filters: parts.append(filters["media_type"])
    if "min_likes"  in filters: parts.append(f"min{filters['min_likes']}")
    return ("_" + "_".join(parts)) if parts else ""


def _render(
    subgraph: dict,
    output_path: Path,
    title: str,
    styles: dict,
    is_comparison: bool = False,
    usernames: Optional[list[str]] = None,
) -> Path:
    nodes_json        = json.dumps(subgraph["nodes"])
    edges_json        = json.dumps(subgraph["edges"])
    meta              = subgraph.get("meta", {})
    group_styles_json = json.dumps(styles)
    edge_colors_json  = json.dumps(EDGE_COLORS)
    edge_widths_json  = json.dumps(EDGE_WIDTHS)

    brand_colors = ["#FF4500", "#00B4FF", "#00E676", "#FFD600"]

    if is_comparison and usernames:
        comparison_legend = "<hr>" + "".join(
            f'<div class="li"><div class="dot" style="background:{brand_colors[i % len(brand_colors)]}"></div><span>@{u}</span></div>'
            for i, u in enumerate(usernames)
        )
    else:
        comparison_legend = ""

    stats_html = (
        f"Nodes <span>{len(subgraph['nodes'])}</span> &nbsp;&nbsp; Edges <span>{len(subgraph['edges'])}</span>"
        if is_comparison else
        f"Posts <span>{meta.get('total_posts','?')}</span> &nbsp;&nbsp; "
        f"Hashtags <span>{meta.get('total_hashtags','?')}</span> &nbsp;&nbsp; "
        f"Mentions <span>{meta.get('total_mentions','?')}</span> &nbsp;&nbsp; "
        f"Themes <span>{meta.get('total_themes','?')}</span> &nbsp;&nbsp; "
        f"Nodes <span>{len(subgraph['nodes'])}</span> &nbsp;&nbsp; "
        f"Edges <span>{len(subgraph['edges'])}</span>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#060a0f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;}}
  body::before{{content:'';position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse at 20% 20%,rgba(255,69,0,.06) 0%,transparent 60%),radial-gradient(ellipse at 80% 80%,rgba(30,144,255,.06) 0%,transparent 60%),radial-gradient(ellipse at 50% 50%,rgba(224,64,251,.04) 0%,transparent 70%);pointer-events:none;}}
  #header{{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(6,10,15,.92);border-bottom:1px solid rgba(255,255,255,.08);padding:10px 20px;display:flex;align-items:center;gap:16px;backdrop-filter:blur(12px);flex-wrap:wrap;}}
  .logo{{color:#FF4500;font-size:16px;font-weight:900;letter-spacing:1px;text-shadow:0 0 20px rgba(255,69,0,.5);white-space:nowrap;}}
  .title-text{{color:#c9d1d9;font-size:11px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:400px;}}
  .live-badge{{display:flex;align-items:center;gap:6px;background:rgba(0,230,118,.1);border:1px solid rgba(0,230,118,.3);border-radius:20px;padding:3px 10px;color:#00E676;font-size:10px;font-weight:600;margin-left:auto;white-space:nowrap;}}
  .live-dot{{width:6px;height:6px;border-radius:50%;background:#00E676;animation:pulse 1.5s ease-in-out infinite;}}
  @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.8)}}}}
  #network{{position:fixed;top:48px;left:0;right:0;bottom:0;z-index:1}}
  #legend{{position:fixed;top:62px;right:14px;z-index:200;background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px 18px;color:#c9d1d9;font-size:11px;min-width:160px;backdrop-filter:blur(12px);box-shadow:0 4px 40px rgba(0,0,0,.6);max-height:calc(100vh - 120px);overflow-y:auto;}}
  #legend h3{{margin:0 0 10px;font-size:10px;color:#8b949e;font-weight:600;text-transform:uppercase;letter-spacing:.8px}}
  #legend hr{{border:none;border-top:1px solid rgba(255,255,255,.08);margin:10px 0;}}
  .li{{display:flex;align-items:center;gap:10px;margin:6px 0;cursor:pointer;transition:opacity .15s}}
  .li:hover{{opacity:.7}}
  .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
  #stats{{position:fixed;bottom:56px;left:0;right:0;z-index:200;display:flex;justify-content:center;pointer-events:none;}}
  .stats-inner{{background:rgba(6,10,15,.88);border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:6px 18px;color:#8b949e;font-size:11px;backdrop-filter:blur(12px);}}
  .stats-inner span{{color:#58a6ff;font-weight:600}}
  #controls{{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);z-index:200;display:flex;gap:8px;}}
  .btn{{background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.1);color:#c9d1d9;padding:8px 16px;border-radius:20px;font-size:11px;cursor:pointer;backdrop-filter:blur(12px);transition:all .2s;user-select:none;font-weight:500;}}
  .btn:hover{{background:rgba(88,166,255,.15);border-color:#58a6ff;color:#58a6ff}}
  .btn.active{{background:rgba(88,166,255,.2);border-color:#58a6ff;color:#58a6ff}}
  #search-wrap{{position:fixed;top:62px;left:14px;z-index:200;}}
  #search{{background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;color:#c9d1d9;font-size:11px;width:200px;outline:none;backdrop-filter:blur(12px);}}
  #search::placeholder{{color:#484f58}}
  #search:focus{{border-color:#58a6ff}}
  #tip{{position:fixed;background:rgba(6,10,15,.97);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:12px 15px;color:#c9d1d9;font-size:11px;pointer-events:auto;display:none;z-index:300;max-width:310px;backdrop-filter:blur(16px);line-height:1.7;box-shadow:0 8px 40px rgba(0,0,0,.7);transition:opacity 0.15s ease;}}
  #tip .tip-title{{font-weight:700;font-size:12px;color:#e6edf3;margin-bottom:6px}}
  #tip .tip-row{{display:flex;justify-content:space-between;gap:12px}}
  #tip .tip-label{{color:#8b949e}}
  #tip .tip-val{{color:#58a6ff;font-weight:600}}
  #tip .tip-themes{{margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}}
  #tip .tip-themes-title{{color:#8b949e;font-size:10px;margin-bottom:4px;}}
  #tip .tip-theme-bar{{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:10px;}}
  #tip .tip-theme-name{{color:#c9d1d9;min-width:80px;}}
  #tip .tip-theme-track{{flex:1;background:rgba(255,255,255,.08);border-radius:4px;height:5px;overflow:hidden;}}
  #tip .tip-theme-fill{{height:100%;border-radius:4px;background:#E040FB;}}
  #tip .tip-theme-score{{color:#E040FB;font-weight:600;min-width:34px;text-align:right;}}
  #tip .tip-caption{{margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);color:#8b949e;font-size:10px;line-height:1.5;font-style:italic;}}
  #tip a{{display:inline-block;margin-top:8px;padding:5px 12px;background:rgba(255,69,0,.15);border:1px solid rgba(255,69,0,.4);border-radius:6px;color:#FF4500;text-decoration:none;font-weight:600;font-size:11px;transition:background .15s;}}
  #tip a:hover{{background:rgba(255,69,0,.3)}}
  #selected-info{{position:fixed;top:62px;left:50%;transform:translateX(-50%);z-index:200;display:none;background:rgba(6,10,15,.92);border:1px solid rgba(88,166,255,.3);border-radius:20px;padding:5px 16px;color:#58a6ff;font-size:11px;backdrop-filter:blur(12px);white-space:nowrap;}}
</style>
</head>
<body>
<div id="header">
  <div class="logo">⬡ GRAPH</div>
  <div class="title-text">{title}</div>
  <div class="live-badge"><div class="live-dot"></div>LIVE DATA</div>
</div>
<div id="network"></div>
<div id="search-wrap"><input id="search" type="text" placeholder="Search nodes…" oninput="searchNodes(this.value)"/></div>
<div id="legend">
  <h3>Node Types</h3>
  <div class="li" onclick="filterGroup('brand')"><div class="dot" style="background:#FF4500"></div><span>Brand</span></div>
  <div class="li" onclick="filterGroup('post')"><div class="dot" style="background:#1E90FF"></div><span>Post</span></div>
  <div class="li" onclick="filterGroup('hashtag')"><div class="dot" style="background:#00E676"></div><span>Hashtag</span></div>
  <div class="li" onclick="filterGroup('mention')"><div class="dot" style="background:#FF9800"></div><span>Mention</span></div>
  <div class="li" onclick="filterGroup('theme')"><div class="dot" style="background:#E040FB"></div><span>Theme</span></div>
  <div class="li" onclick="filterGroup('month')"><div class="dot" style="background:#FFD600"></div><span>Month</span></div>
  <div class="li" onclick="filterGroup('media_type')"><div class="dot" style="background:#26C6DA"></div><span>Media Type</span></div>
  {comparison_legend}
</div>
<div id="stats"><div class="stats-inner">{stats_html}</div></div>
<div id="controls">
  <button class="btn active" id="physicsBtn" onclick="togglePhysics()">⚙ Physics ON</button>
  <button class="btn" onclick="resetView()">⊙ Fit</button>
  <button class="btn" onclick="clearFilter()">✕ Clear Filter</button>
</div>
<div id="selected-info"></div>
<div id="tip"></div>
<script>
const rawNodes={nodes_json};
const rawEdges={edges_json};
const GROUP_STYLES={group_styles_json};
const EDGE_COLORS={edge_colors_json};
const EDGE_WIDTHS={edge_widths_json};
const SHAPES={{brand:'star',brand_a:'star',brand_b:'star',brand_c:'star',brand_d:'star',post:'dot',post_a:'dot',post_b:'dot',post_c:'dot',post_d:'dot',hashtag:'diamond',mention:'triangle',theme:'hexagon',month:'square',media_type:'ellipse'}};
const visNodes=rawNodes.map(n=>{{
  const s=GROUP_STYLES[n.group]||{{color:'#ffffff',glow:'#ffffff',size:12}};
  const sz=n.size||s.size||12;
  return{{id:n.id,label:n.label,group:n.group,size:sz,
    color:{{background:s.color,border:s.color,highlight:{{background:s.color,border:'#ffffff'}},hover:{{background:s.color,border:'#ffffff'}}}},
    shape:SHAPES[n.group]||'dot',
    font:{{color:'#e6edf3',size:n.group==='brand'||n.group.startsWith('brand_')?18:n.group==='theme'?13:10,bold:n.group==='brand'||n.group.startsWith('brand_')||n.group==='theme',strokeWidth:3,strokeColor:'rgba(6,10,15,.8)'}},
    borderWidth:n.group==='brand'||n.group.startsWith('brand_')?4:2,borderWidthSelected:4,
    shadow:{{enabled:true,color:s.glow,size:n.group==='brand'||n.group.startsWith('brand_')?30:n.group==='theme'?20:10,x:0,y:0}},
    _like_count:n.like_count,_comment_count:n.comment_count,_media_type:n.media_type,
    _url:n.url,_caption:n.caption,_theme_scores:n.theme_scores||{{}}}};
}});
const visEdges=rawEdges.map(e=>({{from:e.from,to:e.to,label:'',
  color:{{color:EDGE_COLORS[e.label]||'#ffffff15',highlight:'#ffffff80',hover:'#ffffff60',inherit:false}},
  width:EDGE_WIDTHS[e.label]||1,arrows:{{to:{{enabled:true,scaleFactor:.5,type:'arrow'}}}},
  smooth:{{enabled:true,type:'continuous',roundness:.4}},shadow:{{enabled:true,color:'rgba(0,0,0,.4)',size:5,x:0,y:0}},title:e.label,_label:e.label}}));
const container=document.getElementById('network');
const dataset={{nodes:new vis.DataSet(visNodes),edges:new vis.DataSet(visEdges)}};
const options={{
  physics:{{enabled:true,forceAtlas2Based:{{gravitationalConstant:-80,centralGravity:.008,springLength:160,springConstant:.06,damping:.45,avoidOverlap:1.0}},maxVelocity:60,minVelocity:.08,solver:'forceAtlas2Based',stabilization:{{enabled:true,iterations:1500,updateInterval:25,fit:true}}}},
  interaction:{{hover:true,tooltipDelay:80,zoomView:true,dragView:true,hideEdgesOnDrag:true,multiselect:false,navigationButtons:false}},
  nodes:{{scaling:{{min:8,max:50}}}},edges:{{scaling:{{min:1,max:4}}}},
}};
const net=new vis.Network(container,dataset,options);
let physicsOn=true;
net.on('stabilizationIterationsDone',()=>{{net.setOptions({{physics:{{enabled:false}}}});physicsOn=false;document.getElementById('physicsBtn').textContent='⚙ Physics OFF';}});
function togglePhysics(){{physicsOn=!physicsOn;net.setOptions({{physics:{{enabled:physicsOn}}}});document.getElementById('physicsBtn').textContent=physicsOn?'⚙ Physics ON':'⚙ Physics OFF';document.getElementById('physicsBtn').classList.toggle('active',physicsOn);}}
function resetView(){{net.fit({{animation:{{duration:600,easingFunction:'easeInOutQuad'}}}});}}
const tip=document.getElementById('tip');
let tipPinned=false,hideTimer=null;
tip.addEventListener('mouseenter',()=>{{tipPinned=true;if(hideTimer){{clearTimeout(hideTimer);hideTimer=null;}}}});
tip.addEventListener('mouseleave',()=>{{tipPinned=false;tip.style.display='none';}});
function showTip(html,x,y){{tip.innerHTML=html;tip.style.display='block';tipPinned=false;const tw=310,th=tip.offsetHeight||200,margin=16;let left=x+margin,top=y-10;if(left+tw>window.innerWidth-10)left=x-tw-margin;if(top+th>window.innerHeight-10)top=window.innerHeight-th-10;top=Math.max(52,top);tip.style.left=left+'px';tip.style.top=top+'px';}}
function buildThemeBars(scores){{
  if(!scores||!Object.keys(scores).length)return'';
  const entries=Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const bars=entries.map(([name,score])=>{{
    const pct=Math.round(score*100);
    return`<div class="tip-theme-bar"><span class="tip-theme-name">${{name}}</span><div class="tip-theme-track"><div class="tip-theme-fill" style="width:${{pct}}%"></div></div><span class="tip-theme-score">${{pct}}%</span></div>`;
  }}).join('');
  return`<div class="tip-themes"><div class="tip-themes-title">Theme confidence (DeBERTa)</div>${{bars}}</div>`;
}}
net.on('hoverNode',params=>{{
  const n=dataset.nodes.get(params.node);
  if(!n)return;
  let html=`<div class="tip-title">${{n.label.replace(/\\n/g,' ')}}</div>`;
  html+=`<div class="tip-row"><span class="tip-label">Type</span><span class="tip-val">${{n.group}}</span></div>`;
  if(n._like_count!=null)html+=`<div class="tip-row"><span class="tip-label">Likes</span><span class="tip-val">${{n._like_count.toLocaleString()}}</span></div>`;
  if(n._comment_count!=null)html+=`<div class="tip-row"><span class="tip-label">Comments</span><span class="tip-val">${{n._comment_count.toLocaleString()}}</span></div>`;
  if(n._media_type)html+=`<div class="tip-row"><span class="tip-label">Media</span><span class="tip-val">${{n._media_type}}</span></div>`;
  html+=buildThemeBars(n._theme_scores);
  if(n._caption)html+=`<div class="tip-caption">"${{n._caption.substring(0,100)}}"</div>`;
  if(n._url)html+=`<a href="${{n._url}}" target="_blank" rel="noopener">↗ Open on Instagram</a>`;
  const domPos=net.canvasToDOM(net.getPosition(params.node));
  showTip(html,domPos.x,domPos.y);
}});
net.on('blurNode',()=>{{if(tipPinned)return;hideTimer=setTimeout(()=>{{if(!tipPinned)tip.style.display='none';}},300);}});
const selInfo=document.getElementById('selected-info');
net.on('click',params=>{{if(!params.nodes.length){{selInfo.style.display='none';return;}}const n=dataset.nodes.get(params.nodes[0]);net.focus(params.nodes[0],{{scale:1.4,animation:{{duration:500,easingFunction:'easeInOutQuad'}}}});selInfo.textContent=n.label.replace(/\\n/g,' ')+(n._like_count?` — ${{n._like_count.toLocaleString()}} likes`:'');selInfo.style.display='block';}});
let activeGroup=null;
function filterGroup(group){{if(activeGroup===group){{clearFilter();return;}}activeGroup=group;const groupIds=new Set(visNodes.filter(n=>n.group===group||n.group.startsWith(group.replace('brand','brand_'))).map(n=>n.id));const neighbours=new Set();visEdges.forEach(e=>{{if(groupIds.has(e.from))neighbours.add(e.to);if(groupIds.has(e.to))neighbours.add(e.from);}});dataset.nodes.update(visNodes.map(n=>({{id:n.id,opacity:(groupIds.has(n.id)||neighbours.has(n.id))?1:0.06}})));}}
function clearFilter(){{activeGroup=null;dataset.nodes.update(visNodes.map(n=>({{id:n.id,opacity:1}})));selInfo.style.display='none';}}
function searchNodes(query){{if(!query){{dataset.nodes.update(visNodes.map(n=>({{id:n.id,opacity:1}})));return;}}const q=query.toLowerCase();dataset.nodes.update(visNodes.map(n=>{{const match=n.label.toLowerCase().includes(q);return{{id:n.id,opacity:match?1:0.06}}}}));const match=visNodes.find(n=>n.label.toLowerCase().includes(q));if(match)net.focus(match.id,{{scale:1.5,animation:{{duration:400,easingFunction:'easeInOutQuad'}}}});}}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path