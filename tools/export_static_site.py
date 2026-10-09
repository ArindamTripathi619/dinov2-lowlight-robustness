#!/usr/bin/env python3
"""Export apps/presentation.py to a static, server-free single-page website.

The Streamlit app remains the single source of truth. This script installs a small
`streamlit` shim, imports the *real* presentation module, renders every section
offline — including every widget state (severity sliders, estimator toggle,
corruption radio) — and writes:

    index.html            one-page site: sidebar nav + all 14 sections
    static/gen/*.png      matplotlib figures produced by the app's own helpers

Repository assets (output/**, colab_results/**) are referenced in place at their
committed paths — nothing is copied or duplicated. The result opens straight from
file:// (double-click index.html) and deploys to Vercel with zero configuration
(framework preset "Other", root directory served).

Usage:
    python3 tools/export_static_site.py            # build + self-check
    python3 tools/export_static_site.py --quiet    # build, errors only

After changing apps/presentation.py, re-run this script and commit the refreshed
index.html + static/gen so the deployed site stays in sync.
"""
from __future__ import annotations

import html
import importlib.util
import itertools
import json
import re
import sys
import textwrap
import types
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT / "apps" / "presentation.py"
OUT_HTML = PROJECT / "index.html"
GEN_DIR = PROJECT / "static" / "gen"

MAX_SECTION_RUNS = 30  # safety cap on widget-combination renders per section


class ExportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_lib():
    import markdown as _markdown
    return _markdown


def render_markdown(text: str) -> str:
    """Streamlit-style markdown: dedent triple-quoted bodies, then convert."""
    if not text:
        return ""
    dedented = textwrap.dedent(str(text)).strip()
    if not dedented:
        return ""
    out = _md_lib().markdown(dedented, extensions=["extra", "sane_lists"])
    return out


def render_inline(text: str) -> str:
    """Markdown for a single paragraph/phrase: unwrap the outer <p> if present."""
    out = render_markdown(text)
    m = re.fullmatch(r"<p>(.*)</p>\s*", out, flags=re.S)
    if m:
        return m.group(1)
    return out


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "w"


# ---------------------------------------------------------------------------
# Shim document model
# ---------------------------------------------------------------------------

LEAF_TYPES = {"h", "md", "cap", "table", "fig", "img", "metric", "warn",
              "json", "code", "hr", "widget"}


class Doc:
    """Collects element nodes for the current render run."""

    def __init__(self) -> None:
        self.section: str = ""
        self.overrides: dict = {}
        self.widgets: list[dict] = []
        # root/stack are created by reset(); Scope is defined below.
        self.root = None
        self.stack: list = []

    def reset(self, section: str, overrides: dict | None = None) -> None:
        self.section = section
        self.overrides = overrides or {}
        self.widgets = []
        self.root = Scope(self, "root")
        self.stack = [self.root]

    @property
    def scope(self) -> Scope:
        return self.stack[-1]

    def register_widget(self, w: dict) -> str:
        w["name"] = f"wg_{self.section}_{w['id']}"
        self.widgets.append(w)
        return self.overrides.get((self.section, w["id"]), w["default"])


DOC = Doc()


class Scope:
    """A render scope: the root, an expander, a column, or a tab panel."""

    def __init__(self, doc: Doc, kind: str, payload=None) -> None:
        self._doc = doc
        self.kind = kind
        self.payload = payload
        self.children: list[dict] = []
        self._node: dict | None = None  # set for columns/tabs host nodes

    # -- context manager (expander / column / tab) --------------------------
    def __enter__(self) -> "Scope":
        self._doc.stack.append(self)
        return self

    def __exit__(self, *exc) -> bool:
        self._doc.stack.pop()
        if self.kind == "expander":
            self._doc.scope._add({
                "t": "expander",
                "label": self.payload,
                "children": self.children,
            })
        return False

    # -- internals ----------------------------------------------------------
    def _add(self, node: dict) -> None:
        self.children.append(node)

    # -- elements -----------------------------------------------------------
    def subheader(self, text, **kw):
        self._add({"t": "h", "html": render_inline(text)})

    def title(self, text, **kw):
        self._add({"t": "h", "html": render_inline(text)})

    def markdown(self, text, **kw):
        self._add({"t": "md", "html": render_markdown(text)})

    def caption(self, text, **kw):
        self._add({"t": "cap", "html": render_inline(text)})

    def table(self, data, **kw):
        self._add({"t": "table", "html": _df_html(data, hide_index=False)})

    def dataframe(self, data, hide_index=False, **kw):
        self._add({"t": "table", "html": _df_html(data, hide_index=hide_index)})

    def metric(self, label, value, delta=None, **kw):
        self._add({"t": "metric", "label": str(label),
                   "value": str(value), "delta": None if delta is None else str(delta)})

    def image(self, path, caption=None, **kw):
        src = _norm_src(path)
        self._add({"t": "img", "src": src,
                   "cap": None if caption is None else render_inline(str(caption))})

    def warning(self, text, **kw):
        self._add({"t": "warn", "html": render_markdown(text)})

    def json(self, obj, **kw):
        self._add({"t": "json", "text": json.dumps(obj, indent=2, default=str)})

    def code(self, text, language=None, **kw):
        self._add({"t": "code", "text": str(text), "lang": language or "text"})

    def divider(self, **kw):
        self._add({"t": "hr"})

    def pyplot(self, fig=None, **kw):
        import matplotlib.pyplot as plt
        f = fig if fig is not None else plt.gcf()
        from io import BytesIO
        buf = BytesIO()
        f.savefig(buf, format="png", dpi=110)
        data = buf.getvalue()
        try:
            plt.close(f)
        except Exception:
            pass
        self._add({"t": "fig", "bytes": data,
                   "sha": __import__("hashlib").sha1(data).hexdigest()})

    # -- containers ---------------------------------------------------------
    def expander(self, label, **kw):
        return Scope(self._doc, "expander", str(label))

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        scopes = [Scope(self._doc, "column") for _ in range(n)]
        node = {"t": "columns", "n": n, "scopes": scopes}
        self._add(node)
        return scopes

    def tabs(self, labels, **kw):
        labels = [str(x) for x in labels]
        scopes = [Scope(self._doc, "tab", lab) for lab in labels]
        node = {"t": "tabs", "labels": labels, "scopes": scopes}
        self._add(node)
        return scopes

    # -- widgets ------------------------------------------------------------
    def slider(self, label, min_value=None, max_value=None, value=None,
               key=None, step=None, **kw):
        wid = key or slug(label)
        lo, hi = int(min_value), int(max_value)
        default = int(value)
        w = {"t": "widget", "kind": "slider", "id": wid, "label": str(label),
             "min": lo, "max": hi, "step": int(step or 1), "default": default}
        self._add(w)
        return int(DOC.register_widget(w))

    def checkbox(self, label, value=False, key=None, **kw):
        wid = key or slug(label)
        w = {"t": "widget", "kind": "checkbox", "id": wid,
             "label": str(label), "default": bool(value)}
        self._add(w)
        return bool(DOC.register_widget(w))

    def radio(self, label, options, index=0, key=None, horizontal=False, **kw):
        wid = key or slug(label)
        options = [str(o) for o in options]
        w = {"t": "widget", "kind": "radio", "id": wid, "label": str(label),
             "options": options, "default": options[index],
             "horizontal": bool(horizontal)}
        self._add(w)
        return str(DOC.register_widget(w))

    # -- helpers ------------------------------------------------------------
    def widget_values(self, w: dict) -> list:
        if w["kind"] == "slider":
            return list(range(w["min"], w["max"] + 1, w["step"]))
        if w["kind"] == "checkbox":
            return [False, True]
        if w["kind"] == "radio":
            return list(w["options"])
        raise ExportError(f"unknown widget kind {w['kind']}")


def _df_html(data, hide_index: bool) -> str:
    import pandas as pd
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    return df.to_html(classes="df", index=not hide_index, border=0)


def _norm_src(path) -> str:
    p = Path(str(path))
    if p.is_absolute():
        try:
            return p.relative_to(PROJECT).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def build_streamlit_shim() -> types.ModuleType:
    m = types.ModuleType("streamlit")

    def cache_data(func=None, **kw):
        if func is None:
            return lambda f: f
        return func

    def dispatch(name):
        def fn(*a, **k):
            return getattr(DOC.scope, name)(*a, **k)
        return fn

    m.set_page_config = lambda **kw: None
    m.cache_data = cache_data
    m.cache_resource = cache_data
    m.session_state = {}
    m.sidebar = types.SimpleNamespace(
        title=lambda *a, **k: None,
        caption=lambda *a, **k: None,
        radio=lambda *a, **k: None,
    )
    for name in ("subheader", "title", "markdown", "caption", "table",
                 "dataframe", "metric", "image", "warning", "json", "code",
                 "pyplot", "divider", "expander", "tabs", "columns",
                 "slider", "checkbox", "radio"):
        setattr(m, name, dispatch(name))
    return m


def load_presentation():
    sys.modules.setdefault("streamlit", build_streamlit_shim())
    spec = importlib.util.spec_from_file_location("presentation_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise ExportError(f"cannot load {APP_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["presentation_app"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Render runs + tree merge
# ---------------------------------------------------------------------------

def run_section(app, key: str, overrides: dict | None = None) -> tuple[list, list]:
    DOC.reset(key, overrides)
    app.RENDERERS[key]()
    import matplotlib.pyplot as plt
    plt.close("all")
    return list(DOC.root.children), list(DOC.widgets)


def canon(node: dict) -> str:
    t = node["t"]
    if t == "fig":
        return f"fig:{node['sha']}"
    if t == "img":
        return f"img:{node['src']}|{node.get('cap')}"
    if t == "widget":
        # widgets are serialized with their defaults — state-independent
        return "widget:" + json.dumps({k: v for k, v in node.items() if k != "t"},
                                      sort_keys=True, default=str)
    if t in ("table",):
        return f"table:{node['html']}"
    if t == "metric":
        return "metric:" + json.dumps([node["label"], node["value"], node["delta"]])
    if t == "code":
        return "code:" + node["text"]
    if t == "json":
        return "json:" + node["text"]
    if t in ("h", "md", "cap", "warn"):
        return f"{t}:{node['html']}"
    if t == "hr":
        return "hr"
    raise ExportError(f"canon: unknown node type {t}")


def merge_children(pairs: list[tuple[dict, list]]) -> list:
    """pairs: [(state, children_list)] — same lengths, aligned elements."""
    lengths = {len(c) for _, c in pairs}
    if len(lengths) != 1:
        detail = [len(c) for _, c in pairs]
        raise ExportError(f"section structure differs across widget runs: {detail}")
    if not pairs[0][1]:
        return []
    out = []
    for i in range(len(pairs[0][1])):
        out.append(merge([(st, children[i]) for st, children in pairs]))
    return out


def merge(pairs) -> dict:
    kinds = {n["t"] for _, n in pairs}
    if len(kinds) != 1:
        raise ExportError(f"node type mismatch across runs: {kinds}")
    node = pairs[0][1]
    t = node["t"]

    if t == "widget":
        return node

    if t in ("expander", "columns", "tabs"):
        if t == "expander":
            labels = {n["label"] for _, n in pairs}
            if len(labels) != 1:
                raise ExportError(f"expander label varies across runs: {labels}")
            return {"t": "expander", "label": node["label"],
                    "children": merge_children([(st, n["children"]) for st, n in pairs])}
        if t == "columns":
            ns = {n["n"] for _, n in pairs}
            if len(ns) != 1:
                raise ExportError("column count varies across runs")
            merged = []
            for col in range(node["n"]):
                merged.append(merge_children(
                    [(st, n["scopes"][col].children) for st, n in pairs]))
            return {"t": "columns", "n": node["n"], "panels": merged}
        labels_l = [tuple(n["labels"]) for _, n in pairs]
        if len(set(labels_l)) != 1:
            raise ExportError("tab labels vary across runs")
        merged = []
        for ti in range(len(node["labels"])):
            merged.append(merge_children(
                [(st, n["scopes"][ti].children) for st, n in pairs]))
        return {"t": "tabs", "labels": node["labels"], "panels": merged}

    # leaf: group runs by canonical serialization, preserving first-seen order
    groups: dict[str, dict] = {}
    order: list[str] = []
    for st, n in pairs:
        c = canon(n)
        if c not in groups:
            groups[c] = {"states": [], "node": n}
            order.append(c)
        groups[c]["states"].append(st)
    if len(order) == 1:
        return groups[order[0]]["node"]
    return {"t": "variant",
            "groups": [groups[c] for c in order]}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class Serializer:
    def __init__(self) -> None:
        self.tabs_seq = 0
        self.fig_count = 0

    def figure(self, node: dict) -> str:
        sha = node["sha"]
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        path = GEN_DIR / f"{sha[:16]}.png"
        if not path.exists():
            path.write_bytes(node["bytes"])
        self.fig_count += 1
        return (f'<figure class="fig"><img loading="lazy" src="{path.relative_to(PROJECT).as_posix()}" '
                f'alt="generated figure"></figure>')

    def ser(self, node: dict) -> str:
        t = node["t"]
        if t == "h":
            return f'<h2 class="sh">{node["html"]}</h2>'
        if t == "md":
            return f'<div class="md">{node["html"]}</div>'
        if t == "cap":
            return f'<p class="cap">{node["html"]}</p>'
        if t == "table":
            return f'<div class="tw">{node["html"]}</div>'
        if t == "metric":
            delta = ""
            if node["delta"] is not None:
                d = node["delta"]
                cls = "up" if d.startswith("+") else ("down" if d.startswith("-") else "flat")
                delta = f'<span class="delta {cls}">{html.escape(d)}</span>'
            return (f'<div class="metric"><span class="m-label">{html.escape(node["label"])}</span>'
                    f'<span class="m-value">{html.escape(node["value"])}</span>{delta}</div>')
        if t == "img":
            cap = f'<figcaption>{node["cap"]}</figcaption>' if node.get("cap") else ""
            return (f'<figure class="fig"><img loading="lazy" src="{html.escape(node["src"], quote=True)}" '
                    f'alt="{html.escape(re.sub("<[^>]+>", "", node.get("cap") or "figure"), quote=True)}">{cap}</figure>')
        if t == "fig":
            return self.figure(node)
        if t == "warn":
            return f'<div class="warn">{node["html"]}</div>'
        if t == "json":
            return f'<pre class="json">{html.escape(node["text"])}</pre>'
        if t == "code":
            return f'<pre class="code" data-lang="{html.escape(node["lang"], quote=True)}"><code>{html.escape(node["text"])}</code></pre>'
        if t == "hr":
            return "<hr>"
        if t == "widget":
            return self.widget(node)
        if t == "expander":
            body = "".join(self.ser(c) for c in node["children"])
            return (f'<details class="exp"><summary>{node["label"]}</summary>'
                    f'<div class="exp-body">{body}</div></details>')
        if t == "columns":
            cols = "".join(f'<div class="col">{"".join(self.ser(c) for c in panel)}</div>'
                           for panel in node["panels"])
            n = min(node["n"], 4)
            return f'<div class="cols cols-{n}">{cols}</div>'
        if t == "tabs":
            i = self.tabs_seq
            self.tabs_seq += 1
            heads, panels = [], []
            for j, label in enumerate(node["labels"]):
                tid = f"tb{i}_{j}"
                checked = " checked" if j == 0 else ""
                heads.append(f'<input type="radio" name="tb{i}" id="{tid}"{checked}>'
                             f'<label for="{tid}">{label}</label>')
                body = "".join(self.ser(c) for c in node["panels"][j])
                panels.append(f'<div class="panel" id="p_{tid}">{body}</div>')
            css = "".join(
                f'#{f"tb{i}_{j}"}:checked ~ .panels #p_tb{i}_{j}{{display:block}}'
                for j in range(len(node["labels"])))
            SERIALIZER_CSS.append(css)
            return (f'<div class="tabs">{"".join(heads)}'
                    f'<div class="panels">{"".join(panels)}</div></div>')
        if t == "variant":
            groups = node["groups"]
            data = json.dumps([g["states"] for g in groups],
                              separators=(",", ":"), sort_keys=True)
            inner = []
            for gi, g in enumerate(groups):
                style = "" if gi == 0 else ' style="display:none"'
                inner.append(f'<div class="vg"{style}>{self.ser(g["node"])}</div>')
            return (f'<div class="vv" data-variants="{html.escape(data, quote=True)}">'
                    f'{"".join(inner)}</div>')
        raise ExportError(f"serialize: unknown node type {t}")

    def widget(self, node: dict) -> str:
        kind = node["kind"]
        wid = node["id"]
        label = html.escape(node["label"])
        if kind == "slider":
            w_id = f"w_{DOC.section}_{wid}"
            return (f'<div class="w w-slider"><label for="{w_id}">{label}</label>'
                    f'<div class="w-row"><input type="range" id="{w_id}" data-w="{html.escape(wid, quote=True)}"'
                    f' min="{node["min"]}" max="{node["max"]}" step="{node["step"]}"'
                    f' value="{node["default"]}"><span class="w-val" id="{w_id}_v">{node["default"]}</span></div></div>')
        if kind == "checkbox":
            w_id = f"w_{DOC.section}_{wid}"
            checked = " checked" if node["default"] else ""
            return (f'<label class="w w-check"><input type="checkbox" id="{w_id}"'
                    f' data-w="{html.escape(wid, quote=True)}"{checked}><span>{label}</span></label>')
        if kind == "radio":
            opts = []
            for oi, opt in enumerate(node["options"]):
                w_id = f"w_{DOC.section}_{wid}_{oi}"
                checked = " checked" if opt == node["default"] else ""
                opts.append(f'<label class="w-opt"><input type="radio" name="w_{DOC.section}_{wid}"'
                            f' id="{w_id}" data-w="{html.escape(wid, quote=True)}"'
                            f' value="{html.escape(opt, quote=True)}"{checked}>'
                            f'<span>{html.escape(opt)}</span></label>')
            cls = "w w-radio horizontal" if node.get("horizontal") else "w w-radio"
            return f'<div class="{cls}"><span class="w-title">{label}</span><div class="w-opts">{"".join(opts)}</div></div>'
        raise ExportError(f"widget: unknown kind {kind}")


SERIALIZER_CSS: list[str] = []


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#f4f6f9; --panel:#ffffff; --ink:#1a2230; --muted:#5d6a7d; --line:#e3e8ef;
  --accent:#2563eb; --accent-2:#0ea5e9; --dark:#0e1726; --dark-2:#16223a;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
img{max-width:100%;height:auto}
/* ---------- sidebar ---------- */
.sidebar{position:fixed;top:0;left:0;bottom:0;width:264px;background:var(--dark);
  color:#cfd8e6;overflow-y:auto;padding:22px 14px 30px;z-index:20}
.sidebar .brand{font-weight:700;font-size:17px;color:#fff;letter-spacing:.2px;
  display:flex;align-items:center;gap:8px}
.sidebar .brand-sub{font-size:12px;color:#8b98ad;margin:6px 2px 16px;line-height:1.5}
.sidebar nav a{display:block;padding:7px 10px;margin:1px 0;border-radius:8px;
  color:#b9c4d6;font-size:13.5px;line-height:1.35}
.sidebar nav a:hover{background:var(--dark-2);color:#fff;text-decoration:none}
.sidebar nav a.active{background:var(--accent);color:#fff}
.sidebar .side-note{font-size:11.5px;color:#69768c;margin-top:18px;padding:0 10px;line-height:1.5}
/* ---------- main ---------- */
main{margin-left:264px;padding:34px clamp(16px,4vw,54px) 90px;max-width:1240px}
.hero{margin-bottom:8px}
.hero h1{font-size:30px;line-height:1.2;margin:0 0 8px}
.hero p{color:var(--muted);margin:0 0 6px;max-width:70ch}
.hero .badges{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
.badge{background:#e8eefc;color:#1d4ed8;font-size:12.5px;padding:3px 10px;
  border-radius:999px;border:1px solid #d4e0fb}
section.sec{background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:30px clamp(18px,3vw,40px) 34px;margin:26px 0;scroll-margin-top:18px;
  box-shadow:0 1px 2px rgba(16,24,40,.04)}
h2.sh{font-size:22px;line-height:1.3;margin:6px 0 14px;letter-spacing:-.2px}
section.sec > h2.sh:first-child{font-size:26px;margin-top:0}
.md{margin:12px 0}
.md p{margin:10px 0}
.md ul,.md ol{margin:10px 0;padding-left:24px}
.md li{margin:5px 0}
.md strong{color:#0f172a}
.md h1,.md h2,.md h3,.md h4{margin:18px 0 8px;line-height:1.3}
code{font-family:var(--mono);font-size:.88em;background:#eef1f6;padding:1.5px 5px;
  border-radius:5px;color:#be123c}
pre.code,pre.json{background:#0e1726;color:#dbe4f0;border-radius:12px;padding:16px 18px;
  overflow-x:auto;font-family:var(--mono);font-size:13px;line-height:1.55}
pre.code code{background:none;color:inherit;padding:0}
.cap{color:var(--muted);font-size:13.5px;margin:8px 0 14px}
hr{border:0;border-top:1px solid var(--line);margin:26px 0}
/* ---------- figures & tables ---------- */
figure.fig{margin:14px 0;background:#fbfcfe;border:1px solid var(--line);
  border-radius:12px;padding:10px;text-align:center}
figure.fig img{border-radius:6px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:8px}
.tw{overflow-x:auto;margin:14px 0}
table.df{border-collapse:collapse;font-size:13.5px;width:100%;background:#fff}
table.df th{background:#eef2f8;text-align:left;font-weight:600;color:#33415c}
table.df th,table.df td{border:1px solid var(--line);padding:7px 11px;white-space:nowrap}
table.df tr:nth-child(even) td{background:#f8fafc}
/* ---------- metrics ---------- */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0}
.cols-4 .col, .cols-3 .col, .cols-2 .col{min-width:0}
.metric{background:linear-gradient(180deg,#f8fafd,#fff);border:1px solid var(--line);
  border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.m-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.m-value{font-size:22px;font-weight:700;color:#0f172a}
.delta{font-size:12.5px}
.delta.up{color:#15803d}.delta.down{color:#b91c1c}.delta.flat{color:var(--muted)}
/* ---------- columns ---------- */
.cols{display:grid;gap:16px;margin:14px 0}
.cols-2{grid-template-columns:repeat(2,1fr)}
.cols-3{grid-template-columns:repeat(3,1fr)}
.cols-4{grid-template-columns:repeat(4,1fr)}
.cols-1{grid-template-columns:1fr}
/* ---------- expander ---------- */
details.exp{border:1px solid var(--line);border-radius:12px;margin:12px 0;background:#fcfdff}
details.exp summary{cursor:pointer;padding:13px 18px;font-weight:600;font-size:15px;
  color:#1e293b;list-style:none;display:flex;justify-content:space-between;align-items:center}
details.exp summary::after{content:"+";font-size:18px;color:var(--accent);font-weight:400}
details.exp[open] summary::after{content:"\\2212"}
details.exp summary::-webkit-details-marker{display:none}
details.exp[open] summary{border-bottom:1px solid var(--line)}
.exp-body{padding:6px 18px 16px}
/* ---------- tabs ---------- */
.tabs{margin:16px 0;position:relative}
.tabs > input[type=radio]{position:absolute;opacity:0;pointer-events:none}
.tabs > label{display:inline-block;padding:7px 15px;margin:0 6px 0 0;border-radius:999px;
  border:1px solid var(--line);background:#f4f6fa;font-size:13.5px;cursor:pointer;
  color:#475569;user-select:none}
.tabs > label:hover{border-color:#c8d4e8}
.tabs > input:checked + label{background:var(--accent);border-color:var(--accent);color:#fff}
.tabs .panels{margin-top:14px}
.tabs .panel{display:none}
/* ---------- widgets ---------- */
.w{background:#f8fafd;border:1px solid var(--line);border-radius:12px;
  padding:12px 16px;margin:12px 0}
.w-slider label{font-size:13.5px;color:#33415c;font-weight:600;display:block;margin-bottom:6px}
.w-row{display:flex;align-items:center;gap:14px}
input[type=range]{flex:1;accent-color:var(--accent);height:4px}
.w-val{font-family:var(--mono);font-size:14px;background:#fff;border:1px solid var(--line);
  border-radius:7px;padding:2px 10px;min-width:34px;text-align:center}
.w-check{display:flex;gap:10px;align-items:center;cursor:pointer;font-size:14px}
.w-check input{accent-color:var(--accent);width:16px;height:16px}
.w-title{font-size:13.5px;color:#33415c;font-weight:600;display:block;margin-bottom:6px}
.w-opts{display:flex;gap:8px;flex-wrap:wrap}
.w-opt{display:inline-flex;align-items:center;gap:7px;background:#fff;
  border:1px solid var(--line);border-radius:999px;padding:5px 14px;cursor:pointer;font-size:13.5px}
.w-opt input{accent-color:var(--accent)}
.w-opt:has(input:checked){border-color:var(--accent);background:#e8f0fe;color:#1d4ed8}
/* ---------- misc ---------- */
.warn{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:12px;
  padding:12px 16px;margin:12px 0;font-size:14.5px}
.badge-note{font-size:13px;color:var(--muted)}
footer{margin-top:40px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--line);
  padding-top:18px}
footer code{font-size:12px}
.vv > .vg{display:contents}
/* ---------- responsive ---------- */
@media (max-width:1000px){
  .sidebar{position:static;width:auto;bottom:auto;padding:16px}
  .sidebar nav{display:flex;flex-wrap:wrap;gap:4px}
  .sidebar nav a{background:var(--dark-2)}
  .sidebar .side-note{display:none}
  main{margin-left:0;padding:20px 14px 70px}
  .cols-2,.cols-3,.cols-4{grid-template-columns:1fr}
}
"""

JS = """
(function(){
  document.documentElement.setAttribute('data-js','on');
  function sectionState(sec){
    var st={};
    sec.querySelectorAll('input[data-w]').forEach(function(i){
      if(i.type==='radio'){ if(i.checked) st[i.dataset.w]=i.value; }
      else if(i.type==='checkbox'){ st[i.dataset.w]=i.checked; }
      else { st[i.dataset.w]=i.value; }
    });
    return st;
  }
  function applySection(sec){
    var st=sectionState(sec);
    sec.querySelectorAll('[data-variants]').forEach(function(el){
      var groups;
      try{ groups=JSON.parse(el.dataset.variants); }catch(e){ return; }
      var idx=-1;
      groups.forEach(function(states,i){
        for(var j=0;j<states.length;j++){
          var s=states[j], ok=true;
          for(var k in s){ if(String(s[k])!==String(st[k])){ ok=false; break; } }
          if(ok){ idx=i; return; }
        }
      });
      var kids=el.children;
      for(var i=0;i<kids.length;i++){ kids[i].style.display = (i===idx)?'':'none'; }
    });
  }
  document.addEventListener('input', function(e){
    var t=e.target;
    if(t.matches && t.matches('input[type=range][data-w]')){
      var v=document.getElementById(t.id+'_v');
      if(v) v.textContent=t.value;
    }
    var sec=t.closest && t.closest('section.sec');
    if(sec) applySection(sec);
  });
  document.addEventListener('change', function(e){
    var sec=e.target.closest && e.target.closest('section.sec');
    if(sec) applySection(sec);
  });
  window.addEventListener('DOMContentLoaded', function(){
    document.querySelectorAll('section.sec').forEach(applySection);
    var links=[].slice.call(document.querySelectorAll('.sidebar nav a'));
    var io=new IntersectionObserver(function(entries){
      entries.forEach(function(en){
        if(en.isIntersecting){
          links.forEach(function(a){
            a.classList.toggle('active', a.getAttribute('href')==='#'+en.target.id);
          });
        }
      });
    },{rootMargin:'-15% 0px -70% 0px'});
    document.querySelectorAll('section.sec').forEach(function(s){ io.observe(s); });
  });
})();
"""


def build_page(app, sections_html: dict) -> str:
    labels = app.SECTION_LABELS
    nav = "".join(
        f'<a href="#{key}"{(" class=\"active\"" if i == 0 else "")}>{html.escape(labels[key])}</a>'
        for i, key in enumerate(app.SECTION_ORDER))
    body = "".join(sections_html[key] for key in app.SECTION_ORDER)
    extra_css = "".join(SERIALIZER_CSS)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DINOv2 Low-Light Study — Static Walkthrough</title>
<meta name="description" content="Static export of the DINOv2 low-light robustness
 presentation: how badly the model breaks in the dark, where, why, and the 0.99%-parameter fix.">
<style>{CSS}{extra_css}</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">\U0001F52C DINOv2 low-light study</div>
  <div class="brand-sub">11 phases + methods, bugs, gallery — reads committed
   artifacts only. Static build of <code style="background:#1d2a44;color:#93c5fd;
   padding:1px 4px;border-radius:4px">apps/presentation.py</code>.</div>
  <nav>{nav}</nav>
  <div class="side-note">Generated by <code>tools/export_static_site.py</code>.
   Regenerate after changing the presentation, then commit the refreshed
   <code>index.html</code>.</div>
</aside>
<main>
 <div class="hero">
  <h1>How badly does DINOv2 break in the dark — and can 1% of the parameters fix it?</h1>
  <p>A frozen self-supervised vision model scores 91% on clean photos and collapses to
   ~9% under extreme darkness. This is the full study: where it breaks, why it breaks,
   the cheap-profiling dead end, and the tiny LoRA patch that recovers most of it —
   every number read from committed artifacts, no experiments re-run.</p>
  <div class="badges"><span class="badge">14 sections</span>
   <span class="badge">84 committed figures</span>
   <span class="badge">0.99% parameters patched</span>
   <span class="badge">3 seeds × 9 arms</span>
   <span class="badge">ExDark + ViT-B/14 replications</span></div>
 </div>
 {body}
 <footer>
  Static export of the Streamlit presentation — one source of truth, two renderers.
  Regenerate with <code>python3 tools/export_static_site.py</code>; run
  <code>PRESENTATION_SMOKE=1 python3 apps/presentation.py</code> for the interactive
  app's own check. Values match the CSV/PNG artifacts of record in
  <code>output/</code> and <code>colab_results/</code>.
 </footer>
</main>
<script>{JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    app = load_presentation()
    order = list(app.SECTION_ORDER)
    serializer = Serializer()

    sections_html: dict[str, str] = {}
    stats = {"runs": 0, "figures": 0, "variant_nodes": 0}

    def count_variants(nodes):
        n = 0
        for node in nodes:
            t = node.get("t")
            if t == "variant":
                n += 1
            for key in ("children",):
                if key in node:
                    n += count_variants(node[key])
            if t == "columns":
                for panel in node["panels"]:
                    n += count_variants(panel)
            if t == "tabs":
                for panel in node["panels"]:
                    n += count_variants(panel)
        return n

    for key in order:
        children0, widgets = run_section(app, key)
        default_state = {w["id"]: w["default"] for w in widgets}
        runs = [(default_state, children0)]
        if widgets:
            ranges = [serializer_values(w) for w in widgets]
            combos = list(itertools.product(*ranges))
            if len(combos) + 1 > MAX_SECTION_RUNS:
                raise ExportError(f"{key}: {len(combos)} widget combos exceeds cap")
            for combo in combos:
                state = {w["id"]: v for w, v in zip(widgets, combo)}
                if state == default_state:
                    continue
                ov = {(key, w["id"]): v for w, v in zip(widgets, combo)}
                children_i, widgets_i = run_section(app, key, ov)
                ids_i = [w["id"] for w in widgets_i]
                if ids_i != [w["id"] for w in widgets]:
                    raise ExportError(f"{key}: widget set differs across runs")
                runs.append((state, children_i))
        merged = merge_children([(st, ch) for st, ch in runs])
        stats["runs"] += len(runs)
        stats["variant_nodes"] += count_variants(merged)
        sections_html[key] = (
            f'<section class="sec" id="{key}">'
            + "".join(serializer.ser(node) for node in merged)
            + "</section>")

    page = build_page(app, sections_html)
    OUT_HTML.write_text(page, encoding="utf-8")

    # ---------------- self-checks ----------------
    problems: list[str] = []
    if len(sections_html) != len(order):
        problems.append(f"expected {len(order)} sections, built {len(sections_html)}")
    for key in order:
        if f'id="{key}"' not in page:
            problems.append(f"section {key} missing from HTML")
    srcs = re.findall(r'<img[^>]+src="([^"]+)"', page)
    for src in sorted(set(srcs)):
        if not (PROJECT / src).exists():
            problems.append(f"broken image src: {src}")
    n_tabs = page.count('class="tabs"')
    n_details = page.count("<details")
    n_tables = page.count('class="tw"')
    n_metrics = page.count('class="metric"')
    n_widgets = page.count('data-w=')
    if n_tabs == 0 or n_details == 0 or n_tables == 0 or n_metrics == 0:
        problems.append(f"suspicious element counts: tabs={n_tabs} details={n_details} "
                        f"tables={n_tables} metrics={n_metrics}")
    if n_widgets == 0:
        problems.append("no interactive widgets rendered")

    if problems:
        print("EXPORT FAIL:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 1

    size_kb = OUT_HTML.stat().st_size // 1024
    if not quiet:
        print(f"EXPORT OK: {OUT_HTML.relative_to(PROJECT)} ({size_kb} KB)")
        print(f"  sections: {len(order)}  render runs: {stats['runs']}  "
              f"variant nodes: {stats['variant_nodes']}")
        print(f"  elements: tabs={n_tabs} expanders={n_details} tables={n_tables} "
              f"metrics={n_metrics} widgets={n_widgets}")
        print(f"  images referenced: {len(set(srcs))} (all exist)  "
              f"generated figures: {serializer.fig_count} in {GEN_DIR.relative_to(PROJECT)}/")
    return 0


def serializer_values(w: dict) -> list:
    if w["kind"] == "slider":
        return list(range(w["min"], w["max"] + 1, w["step"]))
    if w["kind"] == "checkbox":
        return [False, True]
    if w["kind"] == "radio":
        return list(w["options"])
    raise ExportError(f"unknown widget kind {w['kind']}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
