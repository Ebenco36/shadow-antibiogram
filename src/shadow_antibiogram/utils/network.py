# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse

try:
    import community as community_louvain  # python-louvain
except Exception as _e:
    community_louvain = None

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False


# ------------------------------- small helpers -------------------------------
import re
from textwrap import dedent

def _inject_recorder_controls(
    html_path,
    mode='webm',             # 'webm' | 'png_seq' | 'apng'
    duration_sec=8,
    fps=30,
    scale=2,                 # upscale factor for recorder frames (png_seq/apng) or hi-res stills
    upscale_webm=False,      # if True, record WebM from a high-res offscreen canvas
    include_hires_button=True,
    hires_scale=None,        # None => fallback to `scale`; else use this for the still PNG
    hires_filename="network_highres.png"
):
    """
    Injects a control panel into the saved PyVis HTML with:
      - Start/Stop recording (webm/png_seq/apng)
      - 'Save High-Res PNG' button for a one-shot HD still (no animation)
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    hires_scale = int(hires_scale or scale)

    # Extra libs per mode
    extra_scripts = []
    if mode == "png_seq":
        extra_scripts.append('<script src="https://cdn.jsdelivr.net/npm/jszip"></script>')
    if mode == "apng":
        extra_scripts.append('<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>')
        extra_scripts.append('<script src="https://cdnjs.cloudflare.com/ajax/libs/upng-js/2.1.0/UPNG.js"></script>')

    panel = dedent(f"""
    <div style="position:fixed;right:16px;bottom:16px;z-index:9999;background:rgba(0,0,0,0.65);padding:10px;border-radius:8px;color:#fff;font-family:system-ui,Arial,sans-serif;">
      <div style="margin-bottom:6px;font-weight:600;">Recorder</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button id="startRecBtn">Start</button>
        <button id="stopRecBtn" disabled>Stop & Download</button>
        {"<button id='saveHiResBtn'>Save High-Res PNG</button>" if include_hires_button else ""}
      </div>
      <div id="recStatus" style="margin-top:6px;font-size:12px;opacity:0.9;"></div>
      <div style="margin-top:6px;font-size:11px;opacity:0.85;">mode: {mode}, fps: {int(fps)}, scale: {int(scale)}</div>
    </div>
    {''.join(extra_scripts)}
    <script>
    (function() {{
      const MODE = "{mode}";
      const DURATION_MS = {int(duration_sec*1000)};
      const FPS = {int(fps)};
      const SCALE = {int(scale)};
      const UPSCALE_WEBM = {str(bool(upscale_webm)).lower()};
      const HIRESPNG_SCALE = {hires_scale};
      const HIRESPNG_NAME = "{hires_filename}";

      const statusEl = document.getElementById('recStatus');
      const startBtn = document.getElementById('startRecBtn');
      const stopBtn = document.getElementById('stopRecBtn');
      const saveBtn = document.getElementById('saveHiResBtn');

      function findCanvas() {{
        const cvs = Array.from(document.querySelectorAll('canvas')).filter(c => c.offsetWidth && c.offsetHeight);
        if (!cvs.length) return null;
        cvs.sort((a,b) => (b.width*b.height) - (a.width*a.height));
        return cvs[0];
      }}
      function downloadBlob(blob, filename) {{
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
        URL.revokeObjectURL(url);
      }}
      function dataURLtoBlob(dataURL) {{
        const arr = dataURL.split(','), mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]); let n = bstr.length; const u8 = new Uint8Array(n);
        while(n--) u8[n] = bstr.charCodeAt(n);
        return new Blob([u8], {{type:mime}});
      }}
      function makeHiResCanvas(canvas, scale) {{
        const c = document.createElement('canvas');
        const w = canvas.width * scale;
        const h = canvas.height * scale;
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        return [c, ctx, w, h];
      }}

      let rec=null, chunks=[], frames=[], frameTimer=null, copyTimer=null, autoStopTimer=null;
      let srcCanvas=null, hiC=null, hiCtx=null, hiW=0, hiH=0;

      function copyToHiRes() {{
        try {{
          hiCtx.clearRect(0,0,hiW,hiH);
          hiCtx.drawImage(srcCanvas, 0, 0, hiW, hiH);
        }} catch(e) {{ console.error(e); }}
      }}

      if (saveBtn) {{
        saveBtn.onclick = () => {{
          const canvas = findCanvas();
          if (!canvas) {{ alert('No canvas found.'); return; }}
          const [c, ctx, w, h] = makeHiResCanvas(canvas, HIRESPNG_SCALE);
          try {{
            ctx.drawImage(canvas, 0, 0, w, h);
            const dataURL = c.toDataURL('image/png');
            downloadBlob(dataURLtoBlob(dataURL), HIRESPNG_NAME);
            statusEl.textContent = `Saved high-res PNG: ${{w}}x${{h}}`;
          }} catch (e) {{
            console.error(e);
            statusEl.textContent = 'High-res PNG save failed.';
          }}
        }};
      }}

      startBtn.onclick = () => {{
        srcCanvas = findCanvas();
        if (!srcCanvas) {{ alert('No canvas found to record.'); return; }}
        [hiC, hiCtx, hiW, hiH] = makeHiResCanvas(srcCanvas, SCALE);

        startBtn.disabled = true; stopBtn.disabled = false;

        if (MODE === 'webm') {{
          chunks = [];
          if (UPSCALE_WEBM) {{
            const stream = hiC.captureStream(FPS);
            const mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(m => MediaRecorder.isTypeSupported(m));
            rec = new MediaRecorder(stream, mime ? {{ mimeType: mime }} : undefined);
            rec.ondataavailable = e => e.data && chunks.push(e.data);
            rec.onstop = () => downloadBlob(new Blob(chunks, {{ type:'video/webm' }}), 'network_hi.webm');
            rec.start(Math.max(1, Math.floor(1000/FPS)));
            const interval = Math.max(1, Math.floor(1000 / FPS));
            copyTimer = setInterval(copyToHiRes, interval);
          }} else {{
            const stream = srcCanvas.captureStream(FPS);
            const mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(m => MediaRecorder.isTypeSupported(m));
            rec = new MediaRecorder(stream, mime ? {{ mimeType: mime }} : undefined);
            rec.ondataavailable = e => e.data && chunks.push(e.data);
            rec.onstop = () => downloadBlob(new Blob(chunks, {{ type:'video/webm' }}), 'network.webm');
            rec.start(Math.max(1, Math.floor(1000/FPS)));
          }}
          statusEl.textContent = 'Recording video...';
          autoStopTimer = setTimeout(() => {{
            if (rec && rec.state==='recording') rec.stop();
            if (copyTimer) clearInterval(copyTimer);
            stopBtn.disabled = true; startBtn.disabled = false;
            statusEl.textContent = 'Saved video.';
          }}, {int(duration_sec*1000)});

        }} else if (MODE === 'png_seq' || MODE === 'apng') {{
          frames = [];
          const interval = Math.max(1, Math.floor(1000 / FPS));
          const capture = () => {{
            try {{
              copyToHiRes();
              if (MODE === 'png_seq') {{
                frames.push(hiC.toDataURL('image/png'));
              }} else {{
                const img = hiCtx.getImageData(0,0,hiW,hiH);
                frames.push(img.data.buffer);
              }}
            }} catch(e) {{ console.error(e); }}
          }};
          frameTimer = setInterval(capture, interval);
          statusEl.textContent = MODE==='apng' ? 'Capturing frames for APNG (hi-res)...' : 'Capturing PNG frames (hi-res)...';
          autoStopTimer = setTimeout(() => stopBtn.click(), {int(duration_sec*1000)});
        }}
      }};

      stopBtn.onclick = async () => {{
        stopBtn.disabled = true; startBtn.disabled = false;

        if (MODE === 'webm') {{
          if (rec && rec.state === 'recording') {{ rec.stop(); statusEl.textContent='Stopping...'; }}
          if (autoStopTimer) clearTimeout(autoStopTimer);
          if (copyTimer) clearInterval(copyTimer);

        }} else if (MODE === 'png_seq') {{
          if (frameTimer) clearInterval(frameTimer);
          if (autoStopTimer) clearTimeout(autoStopTimer);
          statusEl.textContent = 'Packaging frames (ZIP)...';
          if (!window.JSZip) {{ alert('JSZip not available.'); statusEl.textContent='JSZip missing.'; return; }}
          const zip = new JSZip();
          for (let i=0;i<frames.length;i++) {{
            const base64 = frames[i].split(',')[1];
            zip.file(`frame_${{String(i).padStart(4,'0')}}.png`, base64, {{base64:true}});
          }}
          const blob = await zip.generateAsync({{ type:'blob' }});
          downloadBlob(blob, 'frames_hi.zip');
          statusEl.textContent = `Saved ${{frames.length}} frames.`;
          frames = [];

        }} else if (MODE === 'apng') {{
          if (frameTimer) clearInterval(frameTimer);
          if (autoStopTimer) clearTimeout(autoStopTimer);
          if (!window.UPNG) {{ alert('UPNG.js not available.'); statusEl.textContent='UPNG missing.'; return; }}
          statusEl.textContent = 'Encoding APNG (hi-res)...';
          const delay = Math.max(1, Math.floor(1000 / FPS));
          const delays = new Array(frames.length).fill(delay);
          try {{
            const apng = UPNG.encode(frames, hiW, hiH, 0, delays); // 0 -> lossless RGBA
            const blob = new Blob([apng], {{ type:'image/png' }});
            downloadBlob(blob, 'network_animated_hi.png');
            statusEl.textContent = `Saved APNG with ${{frames.length}} frames.`;
          }} catch (e) {{
            console.error(e);
            statusEl.textContent = 'APNG encode failed.';
          }}
          frames = [];
        }}
      }};
    }})();
    </script>
    """)

    # Insert panel before </body></html>
    if re.search(r"</body>\s*</html>\s*$", html, flags=re.IGNORECASE):
        final_html = re.sub(r"</body>\s*</html>\s*$", panel + "\n</body></html>", html, flags=re.IGNORECASE)
    else:
        final_html = html + panel

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
def _safe_to_float_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure float dtype, symmetrize, clamp to [0,1], and fill diag=1."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.astype(float)
    # symmetrize if small numeric drift exists
    out = (out + out.T) / 2.0
    # clamp to [0,1] (typical for similarity)
    out = out.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(out.values, 1.0)
    return out


def _resolve_color_for_group(
    group_id: int,
    custom_palette: Optional[Dict[int, str]],
    default_palette: List[str]
) -> str:
    if isinstance(custom_palette, dict) and group_id in custom_palette:
        return custom_palette[group_id]
    # cycle default palette
    return default_palette[group_id % len(default_palette)]


def _map_edge_widths(
    values: np.ndarray,
    *,
    mode: str = "percentile",                   # "raw" | "normalized" | "percentile" | "log"
    out_range: Tuple[float, float] = (1.0, 8.0),
    gamma: float = 1.2,
    clip_pct: Tuple[float, float] = (10.0, 90.0)
) -> np.ndarray:
    """Map similarity weights to visible stroke width."""
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = out_range
    if not len(v):
        return np.array([], dtype=float)

    if mode == "raw":
        x = v
        vmin, vmax = float(np.min(x)), float(np.max(x))
        if np.isclose(vmax, vmin):
            return np.full_like(x, (lo + hi) / 2.0)
        y = (x - vmin) / (vmax - vmin)

    elif mode == "normalized":
        # robust z-score to [0,1]
        mu = float(np.mean(v))
        sd = float(np.std(v) + 1e-12)
        z = (v - mu) / sd
        z = np.clip((z - z.min()) / (z.max() - z.min() + 1e-12), 0, 1)
        y = z

    elif mode == "percentile":
        p_lo, p_hi = np.percentile(v, clip_pct)
        if np.isclose(p_hi, p_lo):
            y = np.ones_like(v) * 0.5
        else:
            y = (np.clip(v, p_lo, p_hi) - p_lo) / (p_hi - p_lo)

    elif mode == "log":
        x = np.log1p(np.maximum(v, 0.0))
        vmin, vmax = float(np.min(x)), float(np.max(x))
        if np.isclose(vmax, vmin):
            return np.full_like(x, (lo + hi) / 2.0)
        y = (x - vmin) / (vmax - vmin)

    else:
        raise ValueError("edge_width_mode must be one of {'raw','normalized','percentile','log'}")

    # gamma emphasizes strong ties for gamma>1
    y = np.power(y, float(gamma))
    return lo + (hi - lo) * y


def _explode_positions(pos: Dict[Any, Tuple[float, float]], factor: float) -> Dict[Any, Tuple[float, float]]:
    """Scale positions radially from centroid by 'factor' (>1 spreads, <1 contracts)."""
    if factor == 1.0 or not pos:
        return pos
    pts = np.array(list(pos.values()), dtype=float)
    ctr = pts.mean(axis=0)
    out = {}
    for n, (x, y) in pos.items():
        out[n] = (ctr[0] + (x - ctr[0]) * factor, ctr[1] + (y - ctr[1]) * factor)
    return out


def _relax_min_distance(
    pos: Dict[Any, Tuple[float, float]],
    *,
    min_d: float = 1.0,
    iters: int = 30,
    step: float = 0.6,
    radii: Optional[Dict[Any, float]] = None,
    recenter: bool = True
) -> Dict[Any, Tuple[float, float]]:
    """Iteratively push points apart until every pair ≥ min_d (+ optional per-node radius)."""
    keys = list(pos.keys())
    P = np.array([pos[k] for k in keys], dtype=float)

    r = np.zeros(len(keys), dtype=float)
    if radii:
        for i, k in enumerate(keys):
            r[i] = float(radii.get(k, 0.0))

    for _ in range(max(1, int(iters))):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dx = P[j, 0] - P[i, 0]
                dy = P[j, 1] - P[i, 1]
                dist = float(np.hypot(dx, dy)) + 1e-12
                need = min_d + r[i] + r[j]
                if dist < need:
                    overlap = need - dist
                    ux, uy = dx / dist, dy / dist
                    shift = step * overlap * 0.5
                    P[i, 0] -= ux * shift
                    P[i, 1] -= uy * shift
                    P[j, 0] += ux * shift
                    P[j, 1] += uy * shift
                    moved = True
        if not moved:
            break

    if recenter:
        ctr = P.mean(axis=0)
        P -= ctr

    return {k: (float(P[i, 0]), float(P[i, 1])) for i, k in enumerate(keys)}


def _community_circle_bounds(pos: Dict[Any, Tuple[float, float]], nodes: List[Any]) -> Tuple[np.ndarray, float]:
    """Return (center, radius) for a set of nodes based on max distance."""
    pts = np.array([pos[n] for n in nodes if n in pos], dtype=float)
    if len(pts) == 0:
        return np.array([0.0, 0.0]), 0.0
    ctr = pts.mean(axis=0)
    rad = float(np.max(np.linalg.norm(pts - ctr, axis=1))) if len(pts) > 1 else 0.5
    return ctr, rad


def _ellipse_for_points(points: List[Tuple[float, float]], pad: float = 0.6) -> Optional[Tuple[float, float, float, float, float]]:
    """Robust oriented ellipse covering the points; returns (cx, cy, width, height, angle_deg)."""
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        (cx, cy) = pts[0]
        return float(cx), float(cy), 2 * pad, 2 * pad, 0.0
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = (x2 - x1), (y2 - y1)
        dist = float(np.hypot(dx, dy))
        ang = float(np.degrees(np.arctan2(dy, dx)))
        major = dist + 2 * pad
        minor = max(2 * pad, 0.6 * major)
        return float(cx), float(cy), float(major), float(minor), float(ang)
    cx, cy = pts.mean(axis=0)
    X = pts - [cx, cy]
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    rx, ry = (2.0 * s / np.sqrt(len(pts)))  # robust spread
    rx = float(rx + pad)
    ry = float(ry + pad)
    ang = float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0])))
    return float(cx), float(cy), 2 * rx, 2 * ry, ang


# ----------------------------- main visualization ----------------------------

def visualize_antibiotic_network(
    data_input,
    # EDGE FILTERING
    threshold: float = 0.30,
    top_m_per_node: Optional[int] = None,
    max_edges_overall: Optional[int] = None,
    remove_isolated: bool = True,
    community_gamma=1.0,

    # OUTPUTS
    output_dir: str = ".",
    output_html: str = "antibiotic_network.html",
    output_image: str = "antibiotic_network.png",
    gexf_path: Optional[str] = "antibiotic_network.gexf",
    output_pdf: str = "antibiotic_network.pdf",


    # HTML (PyVis)
    show_buttons: bool = True,
    physics_layout: str = "forceAtlas2Based",   # 'forceAtlas2Based' | 'barnesHut' | 'repulsion'
    show_edge_values: bool = False,
    edge_label_decimals: int = 2,
    edge_tooltip_decimals: int = 3,

    # EDGE WIDTH MAPPING (HTML + PNG)
    edge_width_mode: str = "percentile",        # "raw" | "normalized" | "percentile" | "log"
    edge_width_range_html: Tuple[float, float] = (1.0, 10.0),
    edge_width_range_png:  Tuple[float, float] = (0.8, 7.0),
    edge_width_gamma: float = 1.2,
    edge_width_clip: Tuple[float, float] = (10.0, 90.0),

    # PNG (Matplotlib) layout & spacing
    separate_clusters: bool = True,
    outer_layout: str = "spring",               # "spring" | "kamada"
    inner_layout: str = "kamada",               # "spring" | "kamada"
    # spring-only knobs
    k_outer: float = 3.0,
    k_inner: float = 2.0,
    # universal spacing
    scale_outer: float = 7.0,
    scale_inner: float = 2.8,
    figsize: Tuple[int, int] = (16, 16),

    # INTRA-cluster spacing
    intra_min_dist: float = 1.6,
    intra_iters: int = 25,
    intra_step: float = 0.8,
    intra_respect_node_size: bool = True,

    # INTER-cluster spacing
    inter_min_dist: float = 1.4,
    inter_iters: int = 20,
    inter_step: float = 0.55,
    community_padding: float = 0.6,

    # EXPLODE knobs
    outer_explode: float = 1.0,
    inner_explode: float = 1.0,

    # Layout weighting flags
    use_layout_weights_outer: bool = True,
    use_layout_weights_inner: bool = True,

    # NODES
    use_weighted_degree: bool = False,          # size by sum of weights or edge count
    base_node_size: float = 10.0,               # HTML base
    node_size_scale: float = 50.0,              # HTML multiplier
    png_base_node_size: float = 220.0,          # PNG base
    png_size_per_edge: float = 160.0,           # PNG increment per degree unit

    # LABEL styling
    label_font_size_html: int = 22,
    label_font_face_html: str = "Arial",
    label_stroke_width_html: int = 5,
    label_stroke_color_html: str = "#FFFFFF",

    label_font_size_png: int = 16,
    label_font_weight_png: str = "heavy",
    label_outline_width_png: float = 3.5,
    label_outline_color_png: str = "white",

    # LOOK & FEEL
    title: str = "Antibiotic Co-testing Network",
    cluster_colors: Optional[Dict[int, str]] = None,

    # Community bounds (PNG)
    shade_communities_png: bool = True,
    bubble_pad: float = 0.7,
    bubble_alpha: float = 0.12,
    bubble_face: str = "#000000",
    bubble_edge: str = "#00000022",

    node_color_mode: str = "cluster",  # one of: "cluster" | "neutral" | "semantic"
    # - cluster  : color by Louvain community (current behavior)
    # - neutral  : single color for all nodes (temporal-safe)
    # - semantic : color by external stable category (e.g. AWaRe)
    semantic_color_map: Optional[Dict[str, str]] = None,
    neutral_node_color: str = "#5A5A5A",
) -> Dict[str, Any]:
    """
    Visualize a similarity matrix (index==columns) as a network with:
    - thresholding + (optional) top-m per node + (optional) global cap
    - Louvain communities, node sizing, color overrides per community
    - guaranteed intra/inter-cluster spacing (+ explode factors)
    - consistent edge-width mapping for HTML & PNG
    - outlined labels, readable across backgrounds
    - exports HTML (if pyvis present), PNG, GEXF
    Returns a small report dict with output paths and basic stats.
    """

    if community_louvain is None:
        raise RuntimeError("python-louvain is required. Install with: pip install python-louvain")

    # --- Data prep / input handling ---
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(data_input, str):
        df = pd.read_csv(io.StringIO(data_input), index_col=0)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError("data_input must be a CSV string or a pandas DataFrame.")

    if not df.index.equals(df.columns):
        raise ValueError("Similarity matrix must have identical index and columns.")

    df = _safe_to_float_df(df)

    # --- Build candidate edges (upper triangle) & apply threshold ---
    nodes = list(df.index)
    edge_rows: List[Tuple[str, str, float]] = []
    for i, u in enumerate(nodes):
        row = df.loc[u].iloc[i + 1 :]
        strong = row[row >= threshold]
        for v, w in strong.items():
            edge_rows.append((u, v, float(w)))

    # Optional top-m per node (symmetric greedy)
    if top_m_per_node is not None and top_m_per_node > 0:
        edge_rows.sort(key=lambda t: t[2], reverse=True)
        kept, degc = [], {}
        for u, v, w in edge_rows:
            cu, cv = degc.get(u, 0), degc.get(v, 0)
            if cu < top_m_per_node or cv < top_m_per_node:
                kept.append((u, v, w))
                degc[u] = cu + 1
                degc[v] = cv + 1
        edge_rows = kept

    # Optional global cap
    if max_edges_overall is not None:
        edge_rows = sorted(edge_rows, key=lambda t: t[2], reverse=True)[: int(max_edges_overall)]

    # --- Graph ---
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edge_rows:
        G.add_edge(u, v, value=w)

    if remove_isolated:
        iso = [n for n in G if G.degree(n) == 0]
        if iso:
            G.remove_nodes_from(iso)

    if G.number_of_nodes() == 0:
        print("No nodes after filtering; relax threshold / top_m_per_node.")
        return {
            "paths": {"html": None, "png": None, "gexf": None},
            "graph": {"nodes": 0, "edges": 0},
            "communities": {},
        }

    # --- Communities ---
    partition = community_louvain.best_partition(G, weight="value", random_state=42, resolution=community_gamma)

    # --- Node sizes (HTML + PNG) ---
    if use_weighted_degree:
        dv = pd.Series(dict(G.degree(weight="value")), dtype=float)
        dnorm = (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
        html_sizes = {n: float(base_node_size + node_size_scale * dnorm[n]) for n in G}
        png_sizes = {
            n: float(png_base_node_size + png_size_per_edge * (dnorm[n] * (len(G) / 8)))
            for n in G
        }
    else:
        # HTML: normalized degree centrality; PNG: raw degree
        dc = nx.degree_centrality(G)
        html_sizes = {n: float(base_node_size + node_size_scale * dc[n]) for n in G}
        deg_counts = pd.Series(dict(G.degree()))
        png_sizes = {
            n: float(png_base_node_size + png_size_per_edge * float(deg_counts[n])) for n in G
        }

    # --- Node attributes ---
    for n in G:
        G.nodes[n]["group"] = int(partition[n])
        G.nodes[n]["size"] = html_sizes[n]
        G.nodes[n]["title"] = f"Antibiotic: {n}\nCommunity: {partition[n]}\nDegree: {G.degree[n]}"

    # --- Colors (by community, optionally override) ---
    palette_names = [
        "deepskyblue", "orange", "limegreen", "gold", "violet", "tomato",
        "cyan", "magenta", "dodgerblue", "darkorange", "mediumseagreen", "goldenrod",
        "orchid", "slateblue", "lightseagreen", "firebrick"
    ]
    default_palette = [mcolors.CSS4_COLORS[name] for name in palette_names]
    # node_color_map = {
    #     n: _resolve_color_for_group(partition[n], cluster_colors, default_palette) for n in G
    # }

    # --- Node coloring (dynamic switch) ---
    if node_color_mode == "cluster":
        node_color_map = {
            n: _resolve_color_for_group(partition[n], cluster_colors, default_palette)
            for n in G
        }

    elif node_color_mode == "neutral":
        node_color_map = {n: neutral_node_color for n in G}

    elif node_color_mode == "semantic":
        if semantic_color_map is None:
            raise ValueError(
                "semantic_color_map must be provided when node_color_mode='semantic'"
            )
        node_color_map = {
            n: semantic_color_map.get(n, neutral_node_color)
            for n in G
        }

    else:
        raise ValueError(
            "node_color_mode must be one of {'cluster', 'neutral', 'semantic'}"
        )


    # ----------------------------- HTML (PyVis) -----------------------------
    html_path = os.path.join(output_dir, output_html)
    if _HAS_PYVIS:
        net = Network(
            notebook=False,
            height="1200px",
            width="100%",
            bgcolor="white",
            font_color="black",
            directed=False,
        )
        net.from_nx(G)

        # style nodes
        for n in net.nodes:
            c = node_color_map[n["id"]]
            n["color"] = {"background": c, "border": "black", "highlight": {"background": c, "border": "red"}}
            n["borderWidth"] = 2
            n["font"] = {
                "size": int(label_font_size_html),
                "bold": True,
                "color": "black",
                "face": label_font_face_html,
                "strokeWidth": int(label_stroke_width_html),
                "strokeColor": label_stroke_color_html,
            }

        # consistent edge width mapping (HTML)
        Ehtml = list(G.edges(data=True))
        vals_html = np.array([d.get("value", 0.0) for (_, _, d) in Ehtml], dtype=float)
        widths_html = _map_edge_widths(
            vals_html,
            mode=edge_width_mode,
            out_range=edge_width_range_html,
            gamma=edge_width_gamma,
            clip_pct=edge_width_clip,
        )
        for e, w in zip(net.edges, widths_html):
            src, tgt = e["from"], e["to"]
            e["width"] = float(w)
            e["color"] = {"color": node_color_map.get(src, "gray"), "highlight": node_color_map.get(src, "gray")}
            e["smooth"] = {"type": "continuous"}
            if show_edge_values:
                val = float(G.get_edge_data(src, tgt).get("value", 0.0))
                e["label"] = f"{val:.{edge_label_decimals}f}"
                e["title"] = f"value: {val:.{edge_tooltip_decimals}f}"

        # physics choice
        if physics_layout == "forceAtlas2Based":
            net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                                   spring_length=120, spring_strength=0.08,
                                   damping=0.40, overlap=0)
        elif physics_layout == "barnesHut":
            net.barnes_hut(gravity=-80000, central_gravity=0.30,
                           spring_length=250, spring_strength=0.001,
                           damping=0.09, overlap=0)
        elif physics_layout == "repulsion":
            net.repulsion(node_distance=240, central_gravity=0.20,
                          spring_length=180, spring_strength=0.05,
                          damping=0.09)

        if show_buttons:
            net.show_buttons(filter_=["physics"])
        try:
            net.save_graph(html_path)
            
            _inject_recorder_controls(
                html_path, mode='webm', 
                duration_sec=8, fps=20, 
                scale=3, upscale_webm=True
            )

            print(f"[HTML] Saved to '{html_path}'")
        except Exception as e:
            print(f"[HTML] Save error: {e}")
            html_path = None
    else:
        print("[HTML] Skipped (pyvis not installed).")
        html_path = None

    # --------------------------- PNG (Matplotlib) --------------------------
    image_path = os.path.join(output_dir, output_image)
    plt.figure(figsize=figsize)
    plt.rcParams["figure.facecolor"] = "white"

    # group nodes by community
    comm_nodes: Dict[int, List[str]] = {}
    for n, g in partition.items():
        if n in G:
            comm_nodes.setdefault(g, []).append(n)
    comm_ids = list(comm_nodes.keys())

    # ---- Layout (two-level optional) ----
    if separate_clusters:
        # meta graph between communities
        from collections import Counter
        Gc = nx.Graph()
        Gc.add_nodes_from(comm_ids)
        inter = Counter()
        for u, v, d in G.edges(data=True):
            cu, cv = partition[u], partition[v]
            if cu != cv:
                inter[tuple(sorted((cu, cv)))] += 1
        for (cu, cv), w in inter.items():
            Gc.add_edge(cu, cv, weight=w)

        outer_weight = "weight" if use_layout_weights_outer else None
        if outer_layout == "spring":
            pos_outer = nx.spring_layout(Gc, k=k_outer, seed=42, scale=scale_outer, weight=outer_weight)
        elif outer_layout == "kamada":
            pos_outer = nx.kamada_kawai_layout(Gc, weight=outer_weight)
            # uniform scaling
            for c in pos_outer:
                pos_outer[c] = (pos_outer[c][0] * scale_outer, pos_outer[c][1] * scale_outer)
        else:
            raise ValueError("outer_layout must be 'spring' or 'kamada'")

        pos_outer = _explode_positions(pos_outer, float(outer_explode))

        # inner per community
        pos: Dict[Any, Tuple[float, float]] = {}
        for cid, nlist in comm_nodes.items():
            sub = G.subgraph(nlist)
            if len(sub) == 1:
                pos[nlist[0]] = pos_outer[cid]
                continue

            inner_weight = "value" if use_layout_weights_inner else None
            if inner_layout == "spring":
                sub_pos = nx.spring_layout(sub, k=k_inner, seed=42, scale=scale_inner, weight=inner_weight)
            elif inner_layout == "kamada":
                sub_pos = nx.kamada_kawai_layout(sub, weight=inner_weight)
                # normalize + scale
                xs = np.array([xy[0] for xy in sub_pos.values()])
                ys = np.array([xy[1] for xy in sub_pos.values()])
                xs = (xs - xs.mean()) / (xs.std() + 1e-9) * (scale_inner / 2.0)
                ys = (ys - ys.mean()) / (ys.std() + 1e-9) * (scale_inner / 2.0)
                sub_pos = {n: (float(x), float(y)) for n, x, y in zip(sub_pos.keys(), xs, ys)}
            else:
                raise ValueError("inner_layout must be 'spring' or 'kamada'")

            sub_pos = _explode_positions(sub_pos, float(inner_explode))

            # guaranteed intra spacing
            radii = None
            if intra_respect_node_size:
                vals = np.array([png_sizes[n] for n in sub.nodes()], dtype=float)
                vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                radii = {n: float(0.35 * intra_min_dist * vals[k]) for k, n in enumerate(sub.nodes())}

            sub_pos = _relax_min_distance(
                sub_pos, min_d=float(intra_min_dist), iters=int(intra_iters), step=float(intra_step),
                radii=radii, recenter=True
            )

            # anchor to outer centroid
            ox, oy = pos_outer[cid]
            for n, (x, y) in sub_pos.items():
                pos[n] = (ox + x, oy + y)

        # Inter-cluster spacing (push circles apart)
        for _ in range(max(1, int(inter_iters))):
            moved = False
            bounds = {cid: _community_circle_bounds(pos, ns) for cid, ns in comm_nodes.items()}
            # pad radius
            bounds = {cid: (ctr, r + float(community_padding)) for cid, (ctr, r) in bounds.items()}
            for i in range(len(comm_ids)):
                for j in range(i + 1, len(comm_ids)):
                    c1, c2 = comm_ids[i], comm_ids[j]
                    (p1, r1), (p2, r2) = bounds[c1], bounds[c2]
                    dx, dy = p2 - p1
                    dist = float(np.hypot(dx, dy)) or 1e-9
                    need = r1 + r2 + float(inter_min_dist)
                    if dist < need:
                        overlap = need - dist
                        ux, uy = dx / dist, dy / dist
                        shift = inter_step * overlap
                        for n in comm_nodes[c1]:
                            x, y = pos[n]
                            pos[n] = (x - ux * shift * 0.5, y - uy * shift * 0.5)
                        for n in comm_nodes[c2]:
                            x, y = pos[n]
                            pos[n] = (x + ux * shift * 0.5, y + uy * shift * 0.5)
                        moved = True
            if not moved:
                break
    else:
        # single-level layout
        weight_outer = "value" if use_layout_weights_outer else None
        if outer_layout == "spring":
            pos = nx.spring_layout(G, k=k_outer, seed=42, scale=scale_outer, weight=weight_outer)
        elif outer_layout == "kamada":
            pos = nx.kamada_kawai_layout(G, weight=weight_outer)
            xs = np.array([xy[0] for xy in pos.values()])
            ys = np.array([xy[1] for xy in pos.values()])
            xs = (xs - xs.mean()) / (xs.std() + 1e-9) * (scale_outer / 2.0)
            ys = (ys - ys.mean()) / (ys.std() + 1e-9) * (scale_outer / 2.0)
            pos = {n: (float(x), float(y)) for n, x, y in zip(pos.keys(), xs, ys)}
        else:
            raise ValueError("outer_layout must be 'spring' or 'kamada'")

        # still do intra spacing per community
        for cid, nlist in comm_nodes.items():
            sub_pos = {n: pos[n] for n in nlist if n in pos}
            if len(sub_pos) < 2:
                continue
            arr = np.array(list(sub_pos.values()))
            center = arr.mean(axis=0)
            local = {n: (xy[0] - center[0], xy[1] - center[1]) for n, xy in sub_pos.items()}
            radii = None
            if intra_respect_node_size:
                vals = np.array([png_sizes[n] for n in sub_pos.keys()], dtype=float)
                vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                radii = {n: float(0.35 * intra_min_dist * vals[k]) for k, n in enumerate(sub_pos.keys())}
            local = _relax_min_distance(
                local, min_d=float(intra_min_dist), iters=int(intra_iters), step=float(intra_step),
                radii=radii, recenter=True
            )
            for n, (x, y) in local.items():
                pos[n] = (center[0] + x, center[1] + y)

    # ---- Draw PNG ----
    ax = plt.gca()
    if separate_clusters and shade_communities_png:
        for cid, nlist in comm_nodes.items():
            pts = [pos[n] for n in nlist if n in pos]
            out = _ellipse_for_points(pts, pad=float(bubble_pad))
            if out:
                cx, cy, w, h, ang = out
                e = Ellipse(
                    (cx, cy),
                    width=w,
                    height=h,
                    angle=ang,
                    facecolor=mcolors.to_rgba(bubble_face, alpha=float(bubble_alpha)),
                    edgecolor=bubble_edge,
                    lw=1.0,
                    zorder=0,
                )
                ax.add_patch(e)

    node_colors = [node_color_map[n] for n in G]
    node_sizes_png = [png_sizes[n] for n in G]

    # --- Edge widths (PNG) from weights via shared mapper ---
    Epng = list(G.edges(data=True))  # fixed order
    vals_png = np.array([d.get("value", 0.0) for (_, _, d) in Epng], dtype=float)
    edge_widths = _map_edge_widths(
        vals_png,
        mode=edge_width_mode,
        out_range=edge_width_range_png,
        gamma=edge_width_gamma,
        clip_pct=edge_width_clip,
    )
    # color edges by source node color (stable & readable)
    edge_colors = [node_color_map[u] for (u, v, _) in Epng]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for (u, v, _) in Epng],
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.60
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes_png,
        edgecolors="black",
        linewidths=0.7
    )

    # crisp, outlined labels
    texts = nx.draw_networkx_labels(
        G, pos,
        font_size=int(label_font_size_png),
        font_color="black",
        font_weight=label_font_weight_png
    )
    for t in texts.values():
        t.set_path_effects([
            pe.withStroke(linewidth=float(label_outline_width_png), foreground=label_outline_color_png),
            pe.Normal()
        ])

    if show_edge_values and len(Epng) <= 1000:  # guard to avoid overcrowding/slow plots
        edge_labels = {(u, v): f"{d.get('value', 0):.{edge_label_decimals}f}" for u, v, d in Epng}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8, font_color="black",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
        )

    plt.title(title, fontsize=20, color="black")
    plt.axis("off")
    try:
        plt.savefig(image_path, format=output_image.split(".")[-1], bbox_inches="tight", dpi=300)
        print(f"[PNG]  Saved to '{image_path}'")
    except Exception as e:
        print(f"[PNG]  Save error: {e}")
        image_path = None
    
    # Add PDF save
    pdf_path = os.path.join(output_dir, output_pdf)
    try:
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[PDF]  Saved to '{pdf_path}'")
    except Exception as e:
        print(f"[PDF]  Save error: {e}")
        pdf_path = None
    plt.close()

    # ---- GEXF ----
    gexf_out = None
    if gexf_path:
        try:
            gexf_out = os.path.join(output_dir, gexf_path) if os.path.dirname(gexf_path) == "" else gexf_path
            nx.write_gexf(G, gexf_out)
            print(f"[GEXF] Saved to '{gexf_out}'")
        except Exception as e:
            print(f"[GEXF] Save error: {e}")
            gexf_out = None
    # print(len(set(partition.values())), dict(pd.Series(list(partition.values())).value_counts().sort_index()))
    return {
        "paths": {"html": html_path, "png": image_path, "gexf": gexf_out},
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "communities": {
            "n_communities": len(set(partition.values())),
            "sizes": dict(pd.Series(list(partition.values())).value_counts().sort_index())
        }
    }


def visualize_antibiotic_graph_from_partition(
    G: nx.Graph,
    partition: Dict[Any, int],
    *,
    label_map: Optional[Dict[Any, str]] = None,

    # OUTPUTS
    output_dir: str = ".",
    output_html: str = "antibiotic_network_partition.html",
    output_image: str = "antibiotic_network_partition.png",
    gexf_path: Optional[str] = "antibiotic_network_partition.gexf",
    output_pdf: str = "antibiotic_network.pdf",


    # HTML (PyVis)
    show_buttons: bool = True,
    physics_layout: str = "forceAtlas2Based",
    show_edge_values: bool = False,

    # LOOK & FEEL
    title: str = "Antibiotic Network (given partition)",
) -> Dict[str, Any]:
    """
    Visualize an already-clustered graph:

      - G: networkx.Graph with 'value' edge weights (same graph used in evaluation)
      - partition: dict {node -> community_id}, e.g. LouvainClusterer output
      - label_map: optional dict {node -> short label} (e.g. CIP -> CIP)

    No additional Louvain is run here, so clusters match your CSV / ARI / NMI.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----------------- sanity checks -----------------
    if G.number_of_nodes() == 0:
        # print("[viz_from_partition] WARNING: graph has 0 nodes.")
        return {
            "paths": {"html": None, "png": None, "gexf": None},
            "graph": {"nodes": 0, "edges": 0},
            "communities": {"n_communities": 0, "sizes": {}},
        }

    # Only keep partition entries that correspond to nodes in G
    partition = {n: int(g) for n, g in partition.items() if n in G}

    if not partition:
        # print("[viz_from_partition] WARNING: partition is empty or mismatched with graph.")
        return {
            "paths": {"html": None, "png": None, "gexf": None},
            "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
            "communities": {"n_communities": 0, "sizes": {}},
        }

    # --------------- basic color palette ---------------
    palette_names = [
        "deepskyblue", "orange", "limegreen", "gold", "violet", "tomato",
        "cyan", "magenta", "dodgerblue", "darkorange", "mediumseagreen", "goldenrod",
        "orchid", "slateblue", "lightseagreen", "firebrick"
    ]
    palette = [mcolors.CSS4_COLORS[name] for name in palette_names]

    def color_for_group(g: int) -> str:
        return palette[g % len(palette)]

    # --------------- HTML with PyVis ----------------
    html_path = os.path.join(output_dir, output_html)
    if _HAS_PYVIS:
        net = Network(
            notebook=False,
            height="900px",
            width="100%",
            bgcolor="white",
            font_color="black",
            directed=False,
        )

        # add nodes
        for n in G.nodes():
            g = int(partition.get(n, 0))
            label = label_map.get(n, n) if label_map is not None else str(n)
            net.add_node(
                n,
                label=str(label),
                group=g,
                color=color_for_group(g),
                title=f"Node: {label}<br>Community: {g}<br>Degree: {G.degree[n]}",
            )

        # add edges
        for u, v, data in G.edges(data=True):
            val = float(data.get("value", 0.0))
            if show_edge_values:
                title = f"value: {val:.3f}"
            else:
                title = ""
            net.add_edge(u, v, value=val, title=title)

        # physics
        if physics_layout == "forceAtlas2Based":
            net.force_atlas_2based()
        elif physics_layout == "barnesHut":
            net.barnes_hut()
        elif physics_layout == "repulsion":
            net.repulsion()

        if show_buttons:
            net.show_buttons(filter_=["physics"])

        try:
            net.save_graph(html_path)
            print(f"[HTML] Saved to '{html_path}'")
        except Exception as e:
            print(f"[HTML] Save error: {e}")
            html_path = None
    else:
        print("[HTML] Skipped (pyvis not installed).")
        html_path = None

    # --------------- PNG with matplotlib ----------------
    image_path = os.path.join(output_dir, output_image)

    plt.figure(figsize=(12, 12))
    plt.rcParams["figure.facecolor"] = "white"

    # simple spring layout
    pos = nx.spring_layout(G, seed=42, weight="value")

    # node colors and labels
    node_colors = []
    node_labels = {}
    for n in G.nodes():
        g = int(partition.get(n, 0))
        node_colors.append(color_for_group(g))
        node_labels[n] = label_map.get(n, n) if label_map is not None else str(n)

    # edge widths scaled by weight (but very simple)
    weights = np.array(
        [float(data.get("value", 0.0)) for _, _, data in G.edges(data=True)],
        dtype=float,
    )
    if len(weights) > 0:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            widths = 0.5 + 4.5 * (weights - w_min) / (w_max - w_min)
        else:
            widths = np.full_like(weights, 2.0)
    else:
        widths = []

    nx.draw_networkx_edges(
        G, pos,
        width=widths if len(widths) > 0 else 1.0,
        alpha=0.6,
        edge_color="grey",
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=600,
        edgecolors="black",
        linewidths=0.8,
    )

    texts = nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=12,
        font_color="black",
        font_weight="bold",
    )
    # add white outline so labels are readable
    for t in texts.values():
        t.set_path_effects([
            pe.withStroke(linewidth=3.0, foreground="white"),
            pe.Normal(),
        ])

    plt.title(title, fontsize=18)
    plt.axis("off")
    try:
        plt.savefig(image_path, bbox_inches="tight", dpi=300)
        print(f"[PNG] Saved to '{image_path}'")
    except Exception as e:
        print(f"[PNG] Save error: {e}")
        image_path = None

    # Add PDF save
    pdf_path = os.path.join(output_dir, output_pdf)
    try:
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[PDF]  Saved to '{pdf_path}'")
    except Exception as e:
        print(f"[PDF]  Save error: {e}")
        pdf_path = None
    plt.close()


    # --------------- GEXF export ----------------
    gexf_out = None
    if gexf_path:
        try:
            gexf_out = (
                os.path.join(output_dir, gexf_path)
                if os.path.dirname(gexf_path) == ""
                else gexf_path
            )
            nx.write_gexf(G, gexf_out)
            print(f"[GEXF] Saved to '{gexf_out}'")
        except Exception as e:
            print(f"[GEXF] Save error: {e}")
            gexf_out = None

    # --------------- summary ----------------
    community_sizes = (
        pd.Series(list(partition.values()))
        .value_counts()
        .sort_index()
        .to_dict()
    )
    # print(
    #     f"[viz_from_partition] communities: {len(community_sizes)} {community_sizes}, "
    #     f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
    # )

    return {
        "paths": {"html": html_path, "png": image_path, "gexf": gexf_out},
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "communities": {
            "n_communities": len(community_sizes),
            "sizes": community_sizes,
        },
    }