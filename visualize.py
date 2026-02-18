"""
visualize.py — Nike Instagram Knowledge Graph Visualisation
Standalone HTML using vis-network. Designed to match fast-graphrag demo quality:
- Particle-like node glow
- Cluster-style layout with distinct node type sizing
- Animated edges with varied weights
- Rich tooltip with caption preview
- Mini-map + search
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

OUTPUT_HTML = Path("graph_view.html")

# Colour + glow per group
GROUP_STYLES = {
    "brand":      {"color": "#FF4500", "glow": "#FF4500", "size": 50},
    "post":       {"color": "#1E90FF", "glow": "#1E90FF", "size": None},   # dynamic
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
    "POSTED": 2.5,
    "THEME":  2.0,
    "MENTIONS": 1.8,
    "HAS_HASHTAG": 1.2,
    "BELONGS_TO": 1.0,
    "IS_TYPE": 1.0,
}


def generate_visualization(
    filters: Optional[dict] = None,
    output_path: Path = OUTPUT_HTML,
    title: str = "Nike Instagram Knowledge Graph",
) -> Path:
    from graph import extract_subgraph
    print("[visualize] Extracting subgraph …")
    subgraph = extract_subgraph(filters)
    print(f"[visualize] {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    path = _render(subgraph, output_path, title)
    print(f"[visualize]   Saved -> {path.resolve()}")
    return path


def _render(subgraph: dict, output_path: Path, title: str) -> Path:
    nodes_json = json.dumps(subgraph["nodes"])
    edges_json = json.dumps(subgraph["edges"])
    meta       = subgraph.get("meta", {})

    stats_html = (
        f"Posts <span>{meta.get('total_posts','?')}</span> &nbsp;&nbsp; "
        f"Hashtags <span>{meta.get('total_hashtags','?')}</span> &nbsp;&nbsp; "
        f"Mentions <span>{meta.get('total_mentions','?')}</span> &nbsp;&nbsp; "
        f"Themes <span>{meta.get('total_themes','?')}</span> &nbsp;&nbsp; "
        f"Nodes <span>{len(subgraph['nodes'])}</span> &nbsp;&nbsp; "
        f"Edges <span>{len(subgraph['edges'])}</span>"
    )

    group_styles_json = json.dumps(GROUP_STYLES)
    edge_colors_json  = json.dumps(EDGE_COLORS)
    edge_widths_json  = json.dumps(EDGE_WIDTHS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    background:#060a0f;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    overflow:hidden;
  }}

  /* Animated gradient background */
  body::before{{
    content:'';position:fixed;inset:0;z-index:0;
    background:
      radial-gradient(ellipse at 20% 20%, rgba(255,69,0,.06) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 80%, rgba(30,144,255,.06) 0%, transparent 60%),
      radial-gradient(ellipse at 50% 50%, rgba(224,64,251,.04) 0%, transparent 70%);
    pointer-events:none;
  }}

  #header{{
    position:fixed;top:0;left:0;right:0;z-index:100;
    background:rgba(6,10,15,.92);
    border-bottom:1px solid rgba(255,255,255,.08);
    padding:12px 20px;display:flex;align-items:center;gap:16px;
    backdrop-filter:blur(12px);
  }}
  .logo{{
    color:#FF4500;font-size:18px;font-weight:900;letter-spacing:1px;
    text-shadow:0 0 20px rgba(255,69,0,.5);
  }}
  .title-text{{color:#c9d1d9;font-size:12px;font-weight:500}}
  .live-badge{{
    display:flex;align-items:center;gap:6px;
    background:rgba(0,230,118,.1);border:1px solid rgba(0,230,118,.3);
    border-radius:20px;padding:3px 10px;color:#00E676;font-size:10px;font-weight:600;
    margin-left:auto;
  }}
  .live-dot{{
    width:6px;height:6px;border-radius:50%;background:#00E676;
    animation:pulse 1.5s ease-in-out infinite;
  }}
  @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.8)}}}}

  #network{{position:fixed;top:48px;left:0;right:0;bottom:0;z-index:1}}

  /* Legend */
  #legend{{
    position:fixed;top:62px;right:14px;z-index:200;
    background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.08);
    border-radius:12px;padding:16px 18px;color:#c9d1d9;
    font-size:11px;min-width:160px;backdrop-filter:blur(12px);
    box-shadow:0 4px 40px rgba(0,0,0,.6);
  }}
  #legend h3{{margin:0 0 12px;font-size:11px;color:#8b949e;font-weight:600;text-transform:uppercase;letter-spacing:.8px}}
  .li{{display:flex;align-items:center;gap:10px;margin:7px 0;cursor:pointer;transition:opacity .15s}}
  .li:hover{{opacity:.7}}
  .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}

  /* Stats bar */
  #stats{{
    position:fixed;bottom:56px;left:0;right:0;z-index:200;
    display:flex;justify-content:center;
    pointer-events:none;
  }}
  .stats-inner{{
    background:rgba(6,10,15,.88);border:1px solid rgba(255,255,255,.08);
    border-radius:20px;padding:6px 18px;color:#8b949e;
    font-size:11px;backdrop-filter:blur(12px);
  }}
  .stats-inner span{{color:#58a6ff;font-weight:600}}

  /* Controls */
  #controls{{
    position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
    z-index:200;display:flex;gap:8px;
  }}
  .btn{{
    background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.1);
    color:#c9d1d9;padding:8px 16px;border-radius:20px;
    font-size:11px;cursor:pointer;backdrop-filter:blur(12px);
    transition:all .2s;user-select:none;font-weight:500;
  }}
  .btn:hover{{background:rgba(88,166,255,.15);border-color:#58a6ff;color:#58a6ff}}
  .btn.active{{background:rgba(88,166,255,.2);border-color:#58a6ff;color:#58a6ff}}

  /* Search */
  #search-wrap{{
    position:fixed;top:62px;left:14px;z-index:200;
  }}
  #search{{
    background:rgba(6,10,15,.92);border:1px solid rgba(255,255,255,.08);
    border-radius:8px;padding:8px 12px;color:#c9d1d9;font-size:11px;
    width:180px;outline:none;backdrop-filter:blur(12px);
  }}
  #search::placeholder{{color:#484f58}}
  #search:focus{{border-color:#58a6ff}}

  /* Tooltip */
  #tip{{
    position:fixed;
    background:rgba(6,10,15,.97);
    border:1px solid rgba(255,255,255,.1);
    border-radius:10px;padding:12px 15px;color:#c9d1d9;font-size:11px;
    pointer-events:auto;display:none;z-index:300;max-width:260px;
    backdrop-filter:blur(16px);line-height:1.7;
    box-shadow:0 8px 40px rgba(0,0,0,.7);
  }}
  #tip .tip-title{{font-weight:700;font-size:12px;color:#e6edf3;margin-bottom:6px}}
  #tip .tip-row{{display:flex;justify-content:space-between;gap:12px}}
  #tip .tip-label{{color:#8b949e}}
  #tip .tip-val{{color:#58a6ff;font-weight:600}}
  #tip .tip-caption{{
    margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);
    color:#8b949e;font-size:10px;line-height:1.5;font-style:italic;
  }}
  #tip a{{color:#FF4500;text-decoration:none;font-weight:600}}
  #tip a:hover{{text-decoration:underline}}

  /* Node click highlight ring */
  #selected-info{{
    position:fixed;top:62px;left:50%;transform:translateX(-50%);
    z-index:200;display:none;
    background:rgba(6,10,15,.92);border:1px solid rgba(88,166,255,.3);
    border-radius:20px;padding:5px 16px;color:#58a6ff;font-size:11px;
    backdrop-filter:blur(12px);
  }}
</style>
</head>
<body>

<div id="header">
  <div class="logo"> NIKE</div>
  <div class="title-text">Instagram Knowledge Graph &nbsp;&nbsp; Last 60 Days</div>
  <div class="live-badge"><div class="live-dot"></div>LIVE DATA</div>
</div>

<div id="network"></div>

<div id="search-wrap">
  <input id="search" type="text" placeholder="Search nodes…" oninput="searchNodes(this.value)"/>
</div>

<div id="legend">
  <h3>Node Types</h3>
  <div class="li" onclick="filterGroup('brand')">
    <div class="dot" style="background:#FF4500;box-shadow:0 0 8px #FF4500"></div>Brand
  </div>
  <div class="li" onclick="filterGroup('post')">
    <div class="dot" style="background:#1E90FF;box-shadow:0 0 8px #1E90FF"></div>Post
  </div>
  <div class="li" onclick="filterGroup('theme')">
    <div class="dot" style="background:#E040FB;box-shadow:0 0 8px #E040FB"></div>Campaign Theme
  </div>
  <div class="li" onclick="filterGroup('mention')">
    <div class="dot" style="background:#FF9800;box-shadow:0 0 8px #FF9800"></div>Mention
  </div>
  <div class="li" onclick="filterGroup('hashtag')">
    <div class="dot" style="background:#00E676;box-shadow:0 0 8px #00E676"></div>Hashtag
  </div>
  <div class="li" onclick="filterGroup('month')">
    <div class="dot" style="background:#FFD600;box-shadow:0 0 8px #FFD600"></div>Month
  </div>
  <div class="li" onclick="filterGroup('media_type')">
    <div class="dot" style="background:#26C6DA;box-shadow:0 0 8px #26C6DA"></div>Media Type
  </div>
  <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.06);color:#484f58;font-size:10px">
    Click legend to focus  Drag to explore
  </div>
</div>

<div id="stats"><div class="stats-inner">{stats_html}</div></div>

<div id="selected-info"></div>

<div id="controls">
  <div class="btn" onclick="net.fit();this.classList.add('active');setTimeout(()=>this.classList.remove('active'),300)"> Fit All</div>
  <div class="btn" id="physicsBtn" onclick="togglePhysics()"> Physics ON</div>
  <div class="btn" onclick="resetView()"> Reset</div>
</div>

<div id="tip"></div>

<script>
const GROUP_STYLES  = {group_styles_json};
const EDGE_COLORS   = {edge_colors_json};
const EDGE_WIDTHS   = {edge_widths_json};

const SHAPES = {{
  brand:'star', post:'dot', hashtag:'dot',
  mention:'dot', theme:'hexagon', month:'box', media_type:'ellipse'
}};

const rawNodes = {nodes_json};
const rawEdges = {edges_json};

const visNodes = rawNodes.map(n => {{
  const s   = GROUP_STYLES[n.group] || {{color:'#888', glow:'#888', size:12}};
  const sz  = n.size || s.size || 12;
  return {{
    id:    n.id,
    label: n.label,
    group: n.group,
    size:  sz,
    color: {{
      background: s.color,
      border:     s.color,
      highlight:  {{background: s.color, border:'#ffffff'}},
      hover:      {{background: s.color, border:'#ffffff'}},
    }},
    shape:        SHAPES[n.group] || 'dot',
    font: {{
      color: '#e6edf3',
      size:  n.group==='brand'?18 : n.group==='theme'?13 : n.group==='mention'?12 : 10,
      bold:  n.group==='brand' || n.group==='theme',
      strokeWidth: 3,
      strokeColor: 'rgba(6,10,15,.8)',
    }},
    borderWidth:       n.group==='brand'?4:2,
    borderWidthSelected: 4,
    shadow: {{
      enabled: true,
      color:   s.glow,
      size:    n.group==='brand'?30 : n.group==='theme'?20 : n.group==='mention'?16 : 10,
      x: 0, y: 0,
    }},
    // extra data for tooltip
    _like_count:   n.like_count,
    _comment_count:n.comment_count,
    _media_type:   n.media_type,
    _url:          n.url,
    _caption:      n.caption,
  }};
}});

const visEdges = rawEdges.map(e => ({{
  from:   e.from,
  to:     e.to,
  label:  '',   // labels clutter; use tooltip instead
  color:  {{
    color:     EDGE_COLORS[e.label] || '#ffffff15',
    highlight: '#ffffff80',
    hover:     '#ffffff60',
    inherit:   false,
  }},
  width:      EDGE_WIDTHS[e.label] || 1,
  arrows:     {{to:{{enabled:true, scaleFactor:.5, type:'arrow'}}}},
  smooth:     {{enabled:true, type:'continuous', roundness:.4}},
  shadow:     {{enabled:true, color:'rgba(0,0,0,.4)', size:5, x:0, y:0}},
  title:      e.label,   // shown on edge hover
  _label:     e.label,
}}));

const container = document.getElementById('network');
const dataset = {{
  nodes: new vis.DataSet(visNodes),
  edges: new vis.DataSet(visEdges),
}};

const options = {{
  physics:{{
    enabled: true,
    forceAtlas2Based:{{
      gravitationalConstant: -80,
      centralGravity:        .008,
      springLength:          160,
      springConstant:        .06,
      damping:               .45,
      avoidOverlap:          1.0,
    }},
    maxVelocity:    60,
    minVelocity:    .08,
    solver:         'forceAtlas2Based',
    stabilization:  {{enabled:true, iterations:1500, updateInterval:25, fit:true}},
  }},
  interaction:{{
    hover:          true,
    tooltipDelay:   80,
    zoomView:       true,
    dragView:       true,
    hideEdgesOnDrag:true,
    multiselect:    false,
    navigationButtons: false,
  }},
  nodes:{{ scaling:{{min:8, max:50}} }},
  edges:{{ scaling:{{min:1, max:4}} }},
}};

const net = new vis.Network(container, dataset, options);
let physicsOn = true;

net.on('stabilizationIterationsDone', () => {{
  net.setOptions({{physics:{{enabled:false}}}});
  physicsOn = false;
  document.getElementById('physicsBtn').textContent = ' Physics OFF';
}});

function togglePhysics() {{
  physicsOn = !physicsOn;
  net.setOptions({{physics:{{enabled:physicsOn}}}});
  document.getElementById('physicsBtn').textContent = physicsOn ? ' Physics ON' : ' Physics OFF';
  document.getElementById('physicsBtn').classList.toggle('active', physicsOn);
}}

function resetView() {{
  net.fit({{animation:{{duration:600, easingFunction:'easeInOutQuad'}}}});
}}

// ----- Tooltip -----
const tip = document.getElementById('tip');
let hoveringTip = false;

tip.addEventListener('mouseenter', () => {{
  hoveringTip = true;
}});

tip.addEventListener('mouseleave', () => {{
  hoveringTip = false;
  tip.style.display = 'none';
}});

net.on('hoverNode', params => {{
  const n = dataset.nodes.get(params.node);
  if (!n) return;
  let html = `<div class="tip-title">${{n.label}}</div>`;
  html += `<div class="tip-row"><span class="tip-label">Type</span><span class="tip-val">${{n.group}}</span></div>`;
  if (n._like_count    != null) html += `<div class="tip-row"><span class="tip-label">Likes</span><span class="tip-val">${{n._like_count.toLocaleString()}}</span></div>`;
  if (n._comment_count != null) html += `<div class="tip-row"><span class="tip-label">Comments</span><span class="tip-val">${{n._comment_count.toLocaleString()}}</span></div>`;
  if (n._media_type)             html += `<div class="tip-row"><span class="tip-label">Media</span><span class="tip-val">${{n._media_type}}</span></div>`;
  if (n._caption)                html += `<div class="tip-caption">"${{n._caption}}"</div>`;
  if (n._url)                    html += `<div style="margin-top:8px"><a href="${{n._url}}" target="_blank"> Open on Instagram</a></div>`;
  tip.innerHTML = html;
  tip.style.display = 'block';
}});
net.on('blurNode', () => {{
  if (!hoveringTip) tip.style.display='none';
}});
document.addEventListener('mousemove', e => {{
  tip.style.left = (e.clientX+18)+'px';
  tip.style.top  = Math.max(52,(e.clientY-10))+'px';
}});

// ----- Click: zoom to node -----
const selInfo = document.getElementById('selected-info');
net.on('click', params => {{
  if (!params.nodes.length) {{ selInfo.style.display='none'; return; }}
  const n = dataset.nodes.get(params.nodes[0]);
  net.focus(params.nodes[0], {{scale:1.4, animation:{{duration:500, easingFunction:'easeInOutQuad'}}}});
  selInfo.textContent = n.label + (n._like_count ? ` — ${{n._like_count.toLocaleString()}} likes` : '');
  selInfo.style.display = 'block';
}});

// ----- Legend group filter -----
let activeGroup = null;
function filterGroup(group) {{
  if (activeGroup === group) {{
    // reset
    dataset.nodes.update(visNodes.map(n => ({{id:n.id, opacity:1, hidden:false}})));
    activeGroup = null;
    return;
  }}
  activeGroup = group;
  // dim everything except selected group + their direct neighbours
  const groupIds = new Set(visNodes.filter(n => n.group===group).map(n => n.id));
  const neighbours = new Set();
  visEdges.forEach(e => {{
    if (groupIds.has(e.from)) neighbours.add(e.to);
    if (groupIds.has(e.to))   neighbours.add(e.from);
  }});
  dataset.nodes.update(visNodes.map(n => ({{
    id:      n.id,
    opacity: (groupIds.has(n.id) || neighbours.has(n.id)) ? 1 : 0.08,
  }})));
}}

// ----- Search -----
function searchNodes(query) {{
  if (!query) {{
    dataset.nodes.update(visNodes.map(n=>({{id:n.id,opacity:1}})));
    return;
  }}
  const q = query.toLowerCase();
  dataset.nodes.update(visNodes.map(n=>{{
    const match = n.label.toLowerCase().includes(q);
    return {{id:n.id, opacity: match?1:0.08}};
  }}));
  // focus first match
  const match = visNodes.find(n=>n.label.toLowerCase().includes(q));
  if (match) net.focus(match.id, {{scale:1.5, animation:{{duration:400, easingFunction:'easeInOutQuad'}}}});
}}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generate_visualization()