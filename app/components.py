"""HTML/CSS building blocks for the Streamlit app (pure strings: no Streamlit import).

Every function returns an HTML fragment rendered with ``st.html``. Styling lives in
the single :data:`CSS` block, using ``dd-`` prefixed class names only; the few
rules that reach into Streamlit widgets go through the stable ``st-key-<key>``
container classes. Hero images are hotlinked from Valve's CDN (URLs from
``data/artifacts/hero_assets.json``); a missing or broken image falls back to the
hero's initials, and the name is always printed next to the portrait.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

REPO_URL = "https://github.com/JPD989/Dota2-draft-win-predictor"
ATTR_LABEL = {"str": "Strength", "agi": "Agility", "int": "Intelligence", "all": "Universal"}

CSS = """
<style>
:root {
  --dd-surface: #262624; --dd-surface-2: #30302e; --dd-border: #3a3936;
  --dd-text: #f0eee6; --dd-muted: #a39f94; --dd-gold: #d97757;
  --dd-radiant: #6fae6a; --dd-radiant-soft: rgba(111, 174, 106, .16);
  --dd-dire: #d0625a; --dd-dire-soft: rgba(208, 98, 90, .16);
  --dd-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
}
.dd-title, .dd-section h3, .dd-team-name { font-family: var(--dd-serif); }
.dd-win-pct, .dd-rec-p, .dd-delta, .dd-num, .dd-count, .dd-win-foot, .dd-chip b {
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }
/* shape system: containers 12px, media 6px (small icons 4px), controls 8px (theme), chips pill */
.dd-header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
  gap: .75rem 2rem; padding-bottom: .9rem; border-bottom: 1px solid var(--dd-border); }
.dd-eyebrow { color: var(--dd-gold); font-size: .72rem; font-weight: 500; letter-spacing: .14em;
  text-transform: uppercase; margin-bottom: .25rem; }
.dd-title { font-size: 2.2rem; font-weight: 500; line-height: 1.15; letter-spacing: -.01em; margin: 0; color: var(--dd-text); }
.dd-sub { color: var(--dd-muted); font-size: .92rem; margin-top: .35rem; max-width: 46rem; }
.dd-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.dd-chip { border: 1px solid var(--dd-border); background: var(--dd-surface); color: var(--dd-muted);
  border-radius: 999px; padding: .2rem .65rem; font-size: .76rem; white-space: nowrap; }
.dd-chip b { color: var(--dd-text); font-weight: 500; }
a.dd-chip { text-decoration: none; } a.dd-chip:hover { border-color: var(--dd-gold); color: var(--dd-text); }

.dd-note { border: 1px solid var(--dd-border); border-left: 3px solid var(--dd-muted);
  background: var(--dd-surface); border-radius: 12px; padding: .55rem .8rem; font-size: .84rem;
  color: var(--dd-muted); }
.dd-note b { color: var(--dd-text); }
.dd-note.dd-warn { border-left-color: var(--dd-gold); }

.dd-win { background: var(--dd-surface); border: 1px solid var(--dd-border); border-radius: 12px;
  padding: 1rem 1.2rem .9rem; }
.dd-win-row { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.dd-win-side { display: flex; flex-direction: column; }
.dd-win-side.dd-right { align-items: flex-end; }
.dd-win-label { font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 500; }
.dd-win-pct { font-size: 2.3rem; font-weight: 500; letter-spacing: -.01em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.dd-win-mid { color: var(--dd-muted); font-size: .85rem; text-align: center; align-self: center; }
.dd-bar { position: relative; display: flex; height: 12px; border-radius: 999px; overflow: hidden;
  margin: .7rem 0 .45rem; background: var(--dd-surface-2); }
.dd-bar > span { display: block; height: 100%; }
.dd-bar-mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px;
  background: var(--dd-text); opacity: .55; }
.dd-win-foot { color: var(--dd-muted); font-size: .76rem; text-align: center; }
.dd-radiant-c { color: var(--dd-radiant); } .dd-dire-c { color: var(--dd-dire); }
.dd-radiant-bg { background: var(--dd-radiant); } .dd-dire-bg { background: var(--dd-dire); }

.dd-team-head { display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: .55rem; }
.dd-team-name { font-weight: 500; font-size: 1.3rem; }
.dd-count { color: var(--dd-muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.dd-slots { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .5rem; }
.dd-slot { min-width: 0; }
.dd-portrait { position: relative; aspect-ratio: 16 / 9; border-radius: 6px; overflow: hidden;
  background: var(--dd-surface-2); border: 1px solid var(--dd-border); }
.dd-img { position: absolute; inset: 0; background: center / cover no-repeat; }
.dd-initials { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--dd-muted); font-weight: 500; font-size: .95rem; letter-spacing: .05em; }
.dd-slot.dd-empty .dd-portrait { border-style: dashed; background: transparent; }
.dd-slot-name { font-size: .76rem; margin-top: .3rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; color: var(--dd-text); }
.dd-empty .dd-slot-name { color: var(--dd-muted); }
.dd-radiant .dd-slot:not(.dd-empty) .dd-portrait { border-bottom: 2px solid var(--dd-radiant); }
.dd-dire .dd-slot:not(.dd-empty) .dd-portrait { border-bottom: 2px solid var(--dd-dire); }

.dd-bans { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; min-height: 34px; }
.dd-ban { position: relative; width: 72px; aspect-ratio: 16 / 9; border-radius: 6px; overflow: hidden;
  background: var(--dd-surface-2); border: 1px solid var(--dd-border); }
.dd-ban .dd-img { filter: grayscale(1) brightness(.6); }
.dd-ban::after { content: ""; position: absolute; left: -10%; right: -10%; top: 50%; height: 2px;
  background: var(--dd-dire); transform: rotate(-20deg); opacity: .85; }
.dd-ban .dd-initials { font-size: .62rem; }
.dd-muted { color: var(--dd-muted); font-size: .84rem; }

.dd-section { margin: .2rem 0 .8rem; }
.dd-section h3 { font-size: 1.4rem; font-weight: 500; margin: 0; padding: 0; line-height: 1.25; }
.dd-section p { margin: .2rem 0 0; max-width: 65ch; }

.dd-rec { background: var(--dd-surface); border: 1px solid var(--dd-border); border-radius: 12px;
  overflow: hidden; }
.dd-rec .dd-portrait { border: 0; border-radius: 0; }
.dd-rec-rank { position: absolute; top: 6px; left: 6px; background: rgba(31, 30, 29, .85);
  color: var(--dd-text); font-size: .72rem; font-weight: 500; font-variant-numeric: tabular-nums; border-radius: 4px; padding: 1px 6px; }
.dd-rec-body { padding: .5rem .65rem .6rem; }
.dd-rec-name { font-weight: 500; font-size: .9rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.dd-rec-meta { color: var(--dd-muted); font-size: .72rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.dd-rec-nums { display: flex; justify-content: space-between; align-items: baseline; margin-top: .3rem; }
.dd-rec-p { font-size: 1.15rem; font-weight: 500; font-variant-numeric: tabular-nums; }
.dd-delta { font-size: .78rem; font-weight: 500; font-variant-numeric: tabular-nums; }
.dd-up { color: var(--dd-text); } .dd-down { color: var(--dd-muted); } .dd-flat { color: var(--dd-muted); }

.dd-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
.dd-table th { color: var(--dd-muted); font-weight: 500; text-align: left; font-size: .72rem;
  text-transform: uppercase; letter-spacing: .08em; padding: 0 .4rem .4rem; white-space: nowrap; }
.dd-scroll { overflow-x: auto; }
.dd-table td { padding: .32rem .4rem; border-top: 1px solid var(--dd-border); vertical-align: middle; }
.dd-table td.dd-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.dd-hero { display: flex; align-items: center; gap: .5rem; min-width: 0; }
.dd-icon { position: relative; flex: none; width: 44px; aspect-ratio: 16 / 9; border-radius: 4px;
  overflow: hidden; background: var(--dd-surface-2); }
.dd-icon .dd-initials { font-size: .58rem; }
.dd-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dd-ci { position: relative; width: 100%; min-width: 90px; height: 14px; }
.dd-ci-axis { position: absolute; left: 0; right: 0; top: 6px; height: 2px; background: var(--dd-surface-2); }
.dd-ci-mid { position: absolute; left: 50%; top: 1px; height: 12px; width: 1px; background: var(--dd-muted); }
.dd-ci-band { position: absolute; top: 4px; height: 6px; border-radius: 3px; background: var(--dd-gold);
  opacity: .55; }
.dd-ci-pt { position: absolute; top: 2px; width: 3px; height: 10px; margin-left: -1px; border-radius: 1px;
  background: var(--dd-text); }

.dd-footer { color: var(--dd-muted); font-size: .76rem; border-top: 1px solid var(--dd-border);
  padding-top: .8rem; margin-top: 1.5rem; display: flex; flex-wrap: wrap; justify-content: space-between;
  gap: .5rem; }
.dd-footer a { color: var(--dd-muted); }

@media (max-width: 640px) {
  .dd-rec { display: flex; } .dd-rec .dd-portrait { width: 132px; flex: none; }
  .dd-rec-body { flex: 1; min-width: 0; } .dd-win-pct { font-size: 1.8rem; }
}
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }
/* tactile feedback on the one-click pick buttons; static under reduced motion */
@media (prefers-reduced-motion: no-preference) {
  [class*="st-key-pick_"] button { transition: transform .15s cubic-bezier(.16, 1, .3, 1); }
  [class*="st-key-pick_"] button:active { transform: translateY(1px) scale(.98); }
}
.st-key-radiant_panel, .st-key-dire_panel { background: var(--dd-surface); border-radius: 12px; }
.st-key-radiant_panel { border-top: 3px solid var(--dd-radiant) !important; }
.st-key-dire_panel { border-top: 3px solid var(--dd-dire) !important; }
.st-key-radiant_panel [data-baseweb="tag"] { background: var(--dd-radiant-soft); color: var(--dd-text); }
.st-key-dire_panel [data-baseweb="tag"] { background: var(--dd-dire-soft); color: var(--dd-text); }
.st-key-bans_panel [data-baseweb="tag"] { background: var(--dd-surface-2); color: var(--dd-muted); }
</style>
"""


def _e(x: Any) -> str:
    return escape(str(x), quote=True)


def initials(name: str) -> str:
    """Up to two initials for the image fallback (``"Nature's Prophet"`` -> ``"NP"``)."""
    parts = [p for p in name.replace("-", " ").split() if p[:1].isalnum()]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def portrait(name: str, url: str | None, cls: str = "dd-portrait", extra: str = "") -> str:
    """Image box: initials underneath, the CDN image layered on top as a CSS background.

    A background image that fails to load paints nothing (no broken-image icon),
    so the initials show through.
    """
    img = f'<i class="dd-img" style="background-image:url({_e(repr(url))})"></i>' if url else ""
    return f'<div class="{cls}"><div class="dd-initials">{_e(initials(name))}</div>{img}{extra}</div>'


def header(chips: Sequence[tuple[str, str]]) -> str:
    """Page header with title, one-line pitch and ``(label, value)`` info chips."""
    chip_html = "".join(f'<span class="dd-chip">{_e(k)} <b>{_e(v)}</b></span>' for k, v in chips)
    chip_html += f'<a class="dd-chip" href="{REPO_URL}" target="_blank" rel="noopener">GitHub &#8599;</a>'
    return (
        '<div class="dd-header"><div>'
        '<div class="dd-eyebrow">Draft-only benchmark &middot; Dota 2</div>'
        '<h1 class="dd-title">Draft Win Predictor</h1>'
        '<div class="dd-sub">Build both line-ups to see who the draft favours and the strongest '
        'next pick. Expect numbers near 50%: the draft alone says little.</div></div>'
        f'<div class="dd-chips">{chip_html}</div></div>'
    )


def note(html_text: str, warn: bool = False) -> str:
    """Slim callout; ``html_text`` is trusted (built by the app, not user input)."""
    return f'<div class="dd-note{" dd-warn" if warn else ""}">{html_text}</div>'


def win_bar(p_radiant: float, n_radiant: int, n_dire: int) -> str:
    """Focal split bar: Radiant share on the left, Dire on the right, tick at 50%."""
    p = min(max(float(p_radiant), 0.0), 1.0)
    edge = abs(p - 0.5) * 100
    if n_radiant + n_dire == 0:
        mid = "Empty draft &middot; train-split side bias only"
    elif edge < 0.5:
        mid = "Even draft"
    else:
        fav = "Radiant" if p > 0.5 else "Dire"
        mid = f"{fav} +{edge:.1f} pts over a coin flip"
    return (
        '<div class="dd-win"><div class="dd-win-row">'
        '<div class="dd-win-side"><span class="dd-win-label dd-radiant-c">Radiant</span>'
        f'<span class="dd-win-pct">{p:.1%}</span></div>'
        f'<div class="dd-win-mid">{mid}</div>'
        '<div class="dd-win-side dd-right"><span class="dd-win-label dd-dire-c">Dire</span>'
        f'<span class="dd-win-pct">{1 - p:.1%}</span></div></div>'
        f'<div class="dd-bar"><span class="dd-radiant-bg" style="width:{p * 100:.2f}%"></span>'
        f'<span class="dd-dire-bg" style="width:{(1 - p) * 100:.2f}%"></span>'
        '<i class="dd-bar-mid"></i></div>'
        f'<div class="dd-win-foot">{n_radiant}/5 vs {n_dire}/5 picks &middot; '
        'predicted P(win) from the draft only</div></div>'
    )


def team(side: str, heroes: Sequence[int], names: Mapping[int, str],
         assets: Mapping[int, Mapping[str, Any]]) -> str:
    """Team header plus five portrait slots (empty slots drawn as dashed placeholders)."""
    slots = []
    for i in range(5):
        if i < len(heroes):
            h = heroes[i]
            name = names.get(h, f"hero {h}")
            slots.append(f'<div class="dd-slot">{portrait(name, assets.get(h, {}).get("img"))}'
                         f'<div class="dd-slot-name" title="{_e(name)}">{_e(name)}</div></div>')
        else:
            slots.append('<div class="dd-slot dd-empty"><div class="dd-portrait"></div>'
                         f'<div class="dd-slot-name">Pick {i + 1}</div></div>')
    return (f'<div class="dd-{side}"><div class="dd-team-head">'
            f'<span class="dd-team-name dd-{side}-c">{side.capitalize()}</span>'
            f'<span class="dd-count">{len(heroes)}/5</span></div>'
            f'<div class="dd-slots">{"".join(slots)}</div></div>')


def bans(heroes: Sequence[int], names: Mapping[int, str],
         assets: Mapping[int, Mapping[str, Any]]) -> str:
    """Strip of greyed, struck-through ban portraits."""
    if not heroes:
        return '<div class="dd-bans"><span class="dd-muted">No bans.</span></div>'
    items = "".join(
        f'<div title="{_e(names.get(h, h))}">'
        f'{portrait(names.get(h, str(h)), assets.get(h, {}).get("icon"), "dd-ban")}</div>'
        for h in heroes)
    return f'<div class="dd-bans">{items}</div>'


def delta_span(delta: float) -> str:
    """Signed change in percentage points, coloured by direction."""
    pts = delta * 100
    cls = "dd-flat" if abs(pts) < 0.05 else ("dd-up" if pts > 0 else "dd-down")
    return f'<span class="dd-delta {cls}">{pts:+.1f} pts</span>'


def hero_meta(asset: Mapping[str, Any]) -> str:
    """``"Agility, melee"`` (empty when no asset metadata); roles go in the tooltip."""
    bits = [ATTR_LABEL.get(asset.get("primary_attr") or "", ""),
            str(asset.get("attack_type") or "").lower()]
    return _e(", ".join(b for b in bits if b))


def rec_card(rank: int, name: str, asset: Mapping[str, Any], win_prob: float,
             delta: float) -> str:
    """Recommendation card: portrait with rank badge, name, meta, win prob, change."""
    badge = f'<span class="dd-rec-rank">#{rank}</span>'
    meta = hero_meta(asset) or "&nbsp;"
    roles = _e(", ".join(asset.get("roles") or []))
    return (f'<div class="dd-rec" title="{roles}">{portrait(name, asset.get("img"), extra=badge)}'
            f'<div class="dd-rec-body"><div class="dd-rec-name" title="{_e(name)}">{_e(name)}</div>'
            f'<div class="dd-rec-meta">{meta}</div>'
            f'<div class="dd-rec-nums"><span class="dd-rec-p">{win_prob:.1%}</span>'
            f'{delta_span(delta)}</div></div></div>')


def _hero_cell(name: str, asset: Mapping[str, Any], side: str | None = None) -> str:
    dot = (f'<span class="dd-dot dd-{side}-bg"></span>' if side else "")
    return (f'<div class="dd-hero">{dot}{portrait(name, asset.get("icon"), "dd-icon")}'
            f'<span style="white-space:nowrap">{_e(name)}</span></div>')


def ci_bar(wr: float, lo: float, hi: float, span: float = 0.10) -> str:
    """Interval bar on a ``0.5 ± span`` axis with a tick at 50%."""
    def x(v: float) -> float:
        return min(max((v - 0.5 + span) / (2 * span), 0.0), 1.0) * 100

    return (f'<div class="dd-ci" title="{wr:.1%} (95% CI {lo:.1%}-{hi:.1%})">'
            '<i class="dd-ci-axis"></i><i class="dd-ci-mid"></i>'
            f'<i class="dd-ci-band" style="left:{x(lo):.1f}%;width:{x(hi) - x(lo):.1f}%"></i>'
            f'<i class="dd-ci-pt" style="left:{x(wr):.1f}%"></i></div>')


def hero_stats_table(rows: Sequence[Mapping[str, Any]], names: Mapping[int, str],
                     assets: Mapping[int, Mapping[str, Any]]) -> str:
    """Selected heroes with train-split win rate, Wilson 95% CI and games."""
    body = "".join(
        f'<tr><td>{_hero_cell(names.get(r["hero_id"], str(r["hero_id"])), assets.get(r["hero_id"], {}), r["side"])}</td>'
        f'<td class="dd-num">{r["wr"]:.1%}</td>'
        f'<td style="width:34%">{ci_bar(r["wr"], r["ci_lo"], r["ci_hi"])}</td>'
        f'<td class="dd-num dd-muted">{int(r["games"]):,}</td></tr>'
        for r in rows)
    return ('<div class="dd-scroll"><table class="dd-table"><thead><tr><th>Hero</th>'
            '<th style="text-align:right">Win rate</th>'
            '<th>95% CI (40-60%)</th><th style="text-align:right">Games</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def pairs_table(rows: Sequence[Mapping[str, Any]], names: Mapping[int, str],
                assets: Mapping[int, Mapping[str, Any]]) -> str:
    """Pair effects present in the draft: which side each one favours, by how much.

    Rows carry ``kind`` (synergy within ``team`` / counter Radiant ``a`` vs Dire ``b``),
    ``value`` in log-odds for that team (counter: for Radiant) and ``games``.
    "Favours (pts)" is the linearised effect at 50%: ``|value| / 4`` in points.
    """
    body = []
    for r in rows:
        a, b = names.get(r["a"], str(r["a"])), names.get(r["b"], str(r["b"]))
        syn = r["kind"] == "synergy"
        side_a = r["team"] if syn else "radiant"
        v = float(r["value"]) if (not syn or r["team"] == "radiant") else -float(r["value"])
        fav = "radiant" if v >= 0 else "dire"  # v is now in Radiant log-odds
        icons = (portrait(a, assets.get(r["a"], {}).get("icon"), "dd-icon")
                 + portrait(b, assets.get(r["b"], {}).get("icon"), "dd-icon"))
        body.append(
            f'<tr><td><div class="dd-hero"><span class="dd-dot dd-{side_a}-bg"></span>{icons}'
            f'<span>{_e(a)} <span class="dd-muted">{"with" if syn else "vs"}</span> {_e(b)}'
            '</span></div></td>'
            f'<td class="dd-num" title="{abs(v):.3f} log-odds"><span class="dd-delta dd-{fav}-c">'
            f'{fav.capitalize()} +{abs(v) * 25:.1f}</span></td>'
            f'<td class="dd-num dd-muted">{int(r["games"]):,}</td></tr>')
    return ('<div class="dd-scroll"><table class="dd-table"><thead><tr><th>Pair</th>'
            '<th style="text-align:right">Favours (pts)</th>'
            '<th style="text-align:right">Games</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def section(title: str, aside: str = "") -> str:
    """Section heading with an optional muted line underneath (trusted HTML)."""
    sub = f'<p class="dd-muted">{aside}</p>' if aside else ""
    return f'<div class="dd-section"><h3>{_e(title)}</h3>{sub}</div>'


def footer() -> str:
    """Valve attribution and repo link."""
    return ('<div class="dd-footer"><span>Dota 2 and hero images are property of Valve '
            'Corporation. This project is not affiliated with or endorsed by Valve.</span>'
            f'<span>Match data: OpenDota &middot; <a href="{REPO_URL}" target="_blank" '
            'rel="noopener">Source on GitHub</a></span></div>')
