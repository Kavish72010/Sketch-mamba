"""SketchMamba — Gradio demo.

Loads the inference bundle produced by `src.export` and serves a polished
sketchpad UI: live progressive classification (Top-5 bar chart) and
multi-sample stroke auto-completion (drawn as dashed coloured overlays on
the user's canvas).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model import build_model, parse_gmm

CANVAS_W = 480
CANVAS_H = 480
# QuickDraw / Sketch-RNN convention: longest bbox side is normalised to 255 units.
TARGET_BBOX = 255.0
# Mean magnitude of consecutive-point deltas in QuickDraw after RDP simplification.
# Training distribution has dx_std ~= 57, dy_std ~= 43, so a pitch of ~45 model units
# reproduces the spacing the model trained on.  Earlier value of 6 *canvas pixels*
# pre-normalisation yielded ~3-unit model deltas (15x too small) and broke inference.
MODEL_SPACE_PITCH = 45.0


# --------------------------------------------------------------------------- #
def load_bundle(bundle_dir: Path):
    with open(bundle_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    classes = cfg["classes"]
    model = build_model(cfg["model"])
    sd = torch.load(bundle_dir / "weights.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    norm = np.load(bundle_dir / "norm_stats.npz")
    return {
        "cfg": cfg,
        "classes": classes,
        "model": model,
        "dx_std": float(norm["dx_std"]),
        "dy_std": float(norm["dy_std"]),
        "delta_clip": int(norm["delta_clip"]),
    }


# --------------------------------------------------------------------------- #
def strokes_to_stroke5(
    strokes: list[list[dict]],
    dx_std: float,
    dy_std: float,
    delta_clip: int,
    for_autocomplete: bool = False,
) -> tuple[np.ndarray, dict]:
    """Convert canvas-space strokes into the model's stroke-5 representation.

    Pipeline (mirrors the Sketch-RNN / QuickDraw preprocessing the model
    trained on):
      1. Drop degenerate strokes (<2 points).
      2. Bbox-normalise the union of all points so the longest side of the
         drawing's bounding box equals `TARGET_BBOX` (= 255, QuickDraw scale).
      3. Resample each stroke at `MODEL_SPACE_PITCH` *in model space* so
         per-step deltas match the training distribution's magnitude.
      4. Emit Δx, Δy, pen-state, normalise Δ's by the training stds.

    If `for_autocomplete=True`, the last delta of the *final* stroke is
    emitted with pen_down instead of pen_up.  This makes the sequence look
    like the user is still mid-stroke, avoiding the "next token is end-of-
    sketch" distribution the model otherwise predicts after a pen-up delta
    (which causes tiny, zero-mean GMM samples and immediate termination).
    Classification is unaffected and should keep the pen-up marker.
    """
    empty = (
        np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        {"scale": 1.0, "ox": 0.0, "oy": 0.0},
    )
    valid = [
        np.asarray([[p["x"], p["y"]] for p in s], dtype=np.float64)
        for s in strokes if len(s) >= 2
    ]
    if not valid:
        return empty

    # 1) Bbox-normalise to QuickDraw scale (canvas pixels -> model units).
    all_pts = np.concatenate(valid, axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    scale = TARGET_BBOX / span
    ox, oy = float(lo[0]), float(lo[1])

    # 2) Transform to model space, then resample each stroke at a fixed pitch
    #    matched to the training-data delta distribution.
    strokes_m = [(s - np.array([ox, oy])) * scale for s in valid]
    strokes_m = [_resample_polyline(s, MODEL_SPACE_PITCH) for s in strokes_m]
    strokes_m = [s for s in strokes_m if s.shape[0] >= 2]
    if not strokes_m:
        return empty

    # 3) Build (Δx, Δy, pen_up) sequence; mark last delta of every stroke
    #    as pen-up (QuickDraw convention — pen lifts after the final point).
    #    In autocomplete mode, skip this marker on the *last* stroke so the
    #    prefix looks mid-stroke to the model.
    rows: list[tuple[float, float, int]] = []
    last = None
    last_stroke_idx = len(strokes_m) - 1
    for si, s in enumerate(strokes_m):
        stroke_start = len(rows)
        for pt in s:
            if last is None:
                last = pt
                continue
            dx = float(np.clip(pt[0] - last[0], -delta_clip, delta_clip))
            dy = float(np.clip(pt[1] - last[1], -delta_clip, delta_clip))
            rows.append((dx, dy, 0))
            last = pt
        mark_pen_up = len(rows) > stroke_start and not (
            for_autocomplete and si == last_stroke_idx
        )
        if mark_pen_up:
            r = list(rows[-1])
            r[2] = 1
            rows[-1] = tuple(r)

    if not rows:
        return empty

    # 4) Assemble stroke-5 with a start token.  No explicit end token — we want
    #    the model to think the drawing is still in progress (important both
    #    for classification *and* for autocomplete sampling).
    L = len(rows)
    out = np.zeros((L + 1, 5), dtype=np.float32)
    out[0, 2] = 1.0
    for i, (dx, dy, pu) in enumerate(rows, start=1):
        out[i, 0] = dx / dx_std
        out[i, 1] = dy / dy_std
        out[i, 2] = 1.0 if pu == 0 else 0.0
        out[i, 3] = 1.0 if pu == 1 else 0.0

    return out, {"scale": float(scale), "ox": float(ox), "oy": float(oy)}


def _resample_polyline(pts: np.ndarray, pitch: float) -> np.ndarray:
    if pts.shape[0] < 2:
        return pts
    seg = np.diff(pts, axis=0)
    dist = np.sqrt((seg ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(dist)])
    total = float(cum[-1])
    if total < pitch:
        return pts[[0, -1]]
    n = max(2, int(math.ceil(total / pitch)) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for k, t in enumerate(targets):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        seg_len = max(1e-9, cum[j + 1] - cum[j])
        a = (t - cum[j]) / seg_len
        out[k] = (1 - a) * pts[j] + a * pts[j + 1]
    return out


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_topk(model, stroke5: np.ndarray, num_classes: int, k: int = 5):
    x = torch.from_numpy(stroke5).unsqueeze(0)
    out = model(x)
    last = out["cls_logits"][0, -1]                # (C,)
    probs = F.softmax(last.float(), dim=-1).numpy()
    idx = np.argsort(-probs)[:k]
    return idx, probs[idx], probs


@torch.no_grad()
def sample_continuations(
    model,
    stroke5: np.ndarray,
    gmm_components: int,
    n_samples: int,
    max_new: int,
    temperature: float,
    seed: int = 0,
):
    """Returns a list of length n_samples; each element is a list of strokes,
    each stroke a list of (x_model, y_model) absolute coordinates beginning
    at the user's last point."""
    rng = torch.Generator().manual_seed(seed)
    base_seq = torch.from_numpy(stroke5).unsqueeze(0)
    completions = []
    for s in range(n_samples):
        seq = base_seq.clone()
        cur_x, cur_y = 0.0, 0.0
        # accumulate to find current absolute position
        for row in stroke5[1:]:
            cur_x += float(row[0])
            cur_y += float(row[1])
        strokes = [[(cur_x, cur_y)]]                    # start a new stroke from current pos
        for _ in range(max_new):
            out = model(seq)
            gmm = parse_gmm(out["gmm"].float()[:, -1:], gmm_components)
            pen_logits = out["pen_logits"].float()[:, -1]
            # Sample mixture component.
            logits = gmm.pi[:, 0] / max(temperature, 1e-3)
            comp = torch.distributions.Categorical(logits=logits).sample()
            mu_x = gmm.mu_x[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            mu_y = gmm.mu_y[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            sx = gmm.sigma_x[:, 0].gather(-1, comp[:, None]).squeeze(-1) * math.sqrt(temperature)
            sy = gmm.sigma_y[:, 0].gather(-1, comp[:, None]).squeeze(-1) * math.sqrt(temperature)
            rho = gmm.rho[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            eps1 = torch.randn(1, generator=rng)
            eps2 = torch.randn(1, generator=rng)
            dx = (mu_x + sx * eps1).item()
            dy = (mu_y + sy * (rho * eps1 + torch.sqrt(1 - rho * rho) * eps2)).item()
            pen_probs = F.softmax(pen_logits / max(temperature, 1e-3), dim=-1)
            pen = int(torch.distributions.Categorical(probs=pen_probs).sample().item())
            cur_x += dx
            cur_y += dy
            token = torch.zeros(1, 1, 5)
            token[0, 0, 0] = dx
            token[0, 0, 1] = dy
            token[0, 0, 2 + pen] = 1.0
            seq = torch.cat([seq, token], dim=1)
            if pen == 0:
                strokes[-1].append((cur_x, cur_y))
            elif pen == 1:
                strokes[-1].append((cur_x, cur_y))
                strokes.append([(cur_x, cur_y)])
            else:
                break
        completions.append([s for s in strokes if len(s) >= 2])
    return completions


# --------------------------------------------------------------------------- #
def _norm_to_canvas(strokes_norm: list[list[tuple[float, float]]],
                    transform: dict, dx_std: float, dy_std: float) -> list[list[dict]]:
    """Convert normalized-space (Δ/std cumulative) absolute points into canvas
    pixels by inverting the transform applied in `strokes_to_stroke5`:
        canvas_x = ox + (norm_x * dx_std) / scale
        canvas_y = oy + (norm_y * dy_std) / scale
    """
    scale = max(transform["scale"], 1e-9)
    ox, oy = transform["ox"], transform["oy"]
    out = []
    for stroke in strokes_norm:
        if not stroke:
            continue
        seg = [
            {"x": ox + (x_n * dx_std) / scale, "y": oy + (y_n * dy_std) / scale}
            for (x_n, y_n) in stroke
        ]
        out.append(seg)
    return out


# --------------------------------------------------------------------------- #
def build_app(bundle_dir: Path):
    bundle = load_bundle(bundle_dir)
    model = bundle["model"]
    classes = bundle["classes"]
    cfg = bundle["cfg"]
    gmm_M = cfg["model"]["gmm_components"]

    js_path = Path(__file__).with_name("sketch_canvas.js")
    canvas_js = js_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------- #
    def predict_fn(strokes_json: str):
        strokes = json.loads(strokes_json) if strokes_json else []
        if not strokes:
            empty = pd.DataFrame({"class": classes[:5], "prob": [0.0] * 5})
            return empty, "Draw something!"
        x5, _ = strokes_to_stroke5(strokes, bundle["dx_std"], bundle["dy_std"], bundle["delta_clip"])
        idx, probs, _ = predict_topk(model, x5, len(classes), k=5)
        df = pd.DataFrame(
            {"class": [classes[i] for i in idx], "prob": probs.astype(float)}
        )
        top = f"**Top-1**: {classes[idx[0]]}  ({probs[0]*100:.1f}%)"
        return df, top

    def autocomplete_fn(strokes_json: str, n_samples: int, temperature: float, max_new: int):
        strokes = json.loads(strokes_json) if strokes_json else []
        if not strokes:
            return [], "Draw something first."
        x5, transform = strokes_to_stroke5(
            strokes, bundle["dx_std"], bundle["dy_std"], bundle["delta_clip"],
            for_autocomplete=True,
        )
        completions = sample_continuations(
            model, x5, gmm_components=gmm_M, n_samples=int(n_samples),
            max_new=int(max_new), temperature=float(temperature),
        )
        # convert each completion back to canvas-space strokes
        canvas_completions = []
        for comp in completions:
            canvas_completions.append(
                _norm_to_canvas(comp, transform, bundle["dx_std"], bundle["dy_std"])
            )
        return canvas_completions, f"Sampled {len(completions)} continuations."

    # ------------------------------------------------------------------- #
    custom_css = """
    .gradio-container { max-width: 1100px !important; }
    #sketchmamba-canvas-root { display:flex; justify-content:center; }
    """

    head = f"<script>{canvas_js}</script>"
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, head=head, title="SketchMamba") as demo:
        gr.Markdown(
            "# SketchMamba\n"
            "**Joint progressive classification + stroke auto-completion** with a lightweight Mamba state-space backbone."
        )
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("<div id='sketchmamba-canvas-root'></div>")
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    complete_btn = gr.Button("Auto-complete")
                    clear_btn = gr.Button("Clear")
                with gr.Accordion("Auto-completion settings", open=False):
                    n_samples = gr.Slider(1, 6, value=3, step=1, label="Samples")
                    temperature = gr.Slider(0.05, 1.5, value=0.4, step=0.05, label="Temperature τ")
                    max_new = gr.Slider(20, 200, value=80, step=10, label="Max new tokens")
            with gr.Column(scale=2):
                gr.Markdown("### Live Top-5 prediction")
                top_md = gr.Markdown("Draw something!")
                bar = gr.BarPlot(
                    value=pd.DataFrame({"class": classes[:5], "prob": [0.0] * 5}),
                    x="class", y="prob", y_lim=[0.0, 1.0],
                    title="Class probabilities", height=320,
                )
                status = gr.Markdown("")

        # Hidden plumbing: a Textbox that JS fills with the strokes JSON.
        strokes_box = gr.Textbox(visible=False)

        predict_btn.click(
            fn=predict_fn,
            inputs=[strokes_box],
            outputs=[bar, top_md],
            js="() => [JSON.stringify(window.sketchmamba_strokes || [])]",
        )

        # Hidden text payload carrying the completions JSON; its `.change`
        # handler invokes a JS-only function that draws overlays on the canvas.
        overlay_payload = gr.Textbox(visible=False)

        def _on_complete(strokes_json, n, t, m):
            comps, msg = autocomplete_fn(strokes_json, n, t, m)
            return json.dumps(comps), msg

        complete_btn.click(
            fn=_on_complete,
            inputs=[strokes_box, n_samples, temperature, max_new],
            outputs=[overlay_payload, status],
            js="(s, n, t, m) => [JSON.stringify(window.sketchmamba_strokes || []), n, t, m]",
        )

        overlay_payload.change(
            fn=None,
            inputs=[overlay_payload],
            outputs=None,
            js=(
                "(payload) => { try { const data = JSON.parse(payload); "
                "if (window.sketchmamba_overlay) window.sketchmamba_overlay(data); } "
                "catch(e) { console.error('overlay parse', e); } }"
            ),
        )

        clear_btn.click(
            fn=lambda: (pd.DataFrame({"class": classes[:5], "prob": [0.0] * 5}), "Cleared.", ""),
            inputs=None,
            outputs=[bar, top_md, status],
            js="() => { if (window.sketchmamba_clear) window.sketchmamba_clear(); }",
        )

        gr.Markdown(
            f"**Model**: `{cfg['model']['type']}`  |  "
            f"**classes**: {len(classes)}  |  "
            f"**checkpoint epoch**: {cfg.get('epoch', '?')}  |  "
            f"**git**: `{cfg.get('git_sha', '?')}`"
        )
    return demo


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="exported/sketchmamba_v1", type=Path)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo = build_app(args.bundle)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
