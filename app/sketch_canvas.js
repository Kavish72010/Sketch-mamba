// SketchMamba — vector stroke canvas.
// Captures pointer events into window.sketchmamba_strokes as an array of strokes,
// each stroke being an array of {x, y} absolute canvas-coordinate points.
// Also exposes window.sketchmamba_overlay(points[, color]) to draw a coloured
// completion polyline over the user's sketch (called from the Python side).

(function () {
  if (window.__sketchmamba_init) return;
  window.__sketchmamba_init = true;

  function setup() {
    const root = document.getElementById("sketchmamba-canvas-root");
    if (!root) { setTimeout(setup, 100); return; }
    if (root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    const W = 480, H = 480;
    root.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;">
        <canvas id="sketchmamba-canvas" width="${W}" height="${H}"
          style="background:#fff;border:1px solid #cbd5e1;border-radius:12px;
                 box-shadow:0 1px 4px rgba(0,0,0,0.08);touch-action:none;cursor:crosshair;">
        </canvas>
        <div style="font-size:12px;color:#64748b;margin-top:6px;">
          Draw with mouse / pen / touch. Strokes are captured as vectors.
        </div>
      </div>`;
    const canvas = document.getElementById("sketchmamba-canvas");
    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#0f172a";

    let strokes = [];
    let overlays = [];                // [{segments:[[{x,y},...], ...], color:"#..."}]
    window.sketchmamba_strokes = strokes;

    let drawing = false;
    let cur = null;

    function redraw() {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 2.5;
      for (const s of strokes) {
        if (s.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(s[0].x, s[0].y);
        for (let i = 1; i < s.length; i++) ctx.lineTo(s[i].x, s[i].y);
        ctx.stroke();
      }
      for (const ov of overlays) {
        ctx.strokeStyle = ov.color || "#ef4444";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 4]);
        for (const seg of ov.segments) {
          if (seg.length < 2) continue;
          ctx.beginPath();
          ctx.moveTo(seg[0].x, seg[0].y);
          for (let i = 1; i < seg.length; i++) ctx.lineTo(seg[i].x, seg[i].y);
          ctx.stroke();
        }
        ctx.setLineDash([]);
      }
    }

    function pos(ev) {
      const r = canvas.getBoundingClientRect();
      const t = ev.touches ? ev.touches[0] : ev;
      return { x: (t.clientX - r.left) * (canvas.width / r.width),
               y: (t.clientY - r.top)  * (canvas.height / r.height) };
    }

    function down(ev) {
      ev.preventDefault();
      drawing = true;
      cur = [pos(ev)];
      strokes.push(cur);
      window.sketchmamba_strokes = strokes;
      overlays = [];
      redraw();
    }
    function move(ev) {
      if (!drawing) return;
      ev.preventDefault();
      cur.push(pos(ev));
      redraw();
    }
    function up(ev) {
      if (!drawing) return;
      ev.preventDefault();
      drawing = false;
      window.sketchmamba_strokes = strokes;
      // Notify any listeners (used to trigger live predictions).
      window.dispatchEvent(new CustomEvent("sketchmamba:stroke"));
    }

    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("pointerleave", up);

    window.sketchmamba_clear = function () {
      strokes.length = 0;
      overlays = [];
      window.sketchmamba_strokes = strokes;
      redraw();
    };

    // completions: array of completions; each completion is an array of strokes;
    // each stroke an array of {x, y}.  colors: optional array (one per completion).
    window.sketchmamba_overlay = function (completions, colors) {
      const palette = colors && colors.length
        ? colors
        : ["#ef4444", "#10b981", "#3b82f6", "#a855f7", "#f59e0b", "#06b6d4"];
      overlays = (completions || []).map((segs, i) => ({
        segments: segs || [],
        color: palette[i % palette.length],
      }));
      redraw();
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
  // Re-attempt periodically (Gradio may insert the root after our script runs).
  let tries = 0;
  const iv = setInterval(() => {
    setup();
    if (++tries > 50) clearInterval(iv);
  }, 200);
})();
