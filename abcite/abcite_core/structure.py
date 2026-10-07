"""2D residue-map snapshots (SVG). Swap renderer for py3Dmol/PyMOL PNG."""

from __future__ import annotations

from pathlib import Path


def residue_map_svg(
    pdb_id: str,
    chain: str,
    length: int,
    highlights: list[int],
    out_path: Path,
) -> Path:
    """Documented local renderer: circular residue beads, epitope in amber."""
    w = max(640, length * 3)
    h = 120
    marked = set(highlights)
    beads = []
    for i in range(1, length + 1):
        x = 16 + (i - 1) * (w - 32) / max(1, length - 1)
        y = 60
        fill = "#d97706" if i in marked else "#1f3a4d"
        r = 5 if i in marked else 3
        beads.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r}" fill="{fill}"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="#0b1220"/>
  <text x="16" y="24" fill="#9fb3c8" font-family="ui-sans-serif" font-size="14">{pdb_id} chain {chain} residue map</text>
  {''.join(beads)}
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def py3dmol_html(pdb_id: str, residues: list[str]) -> str:
    res_js = ",".join(f'"{r}"' for r in residues)
    return f"""<!DOCTYPE html>
<html><head><script src="https://unpkg.com/3dmol/build/3Dmol-min.js"></script></head>
<body>
<div id="v" style="width:100%;height:360px;background:#0b1220"></div>
<script>
const viewer = $3Dmol.createViewer("v", {{backgroundColor: "#0b1220"}});
$3Dmol.download("pdb:{pdb_id}", viewer, {{}}, function() {{
  viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
  const residues = [{res_js}];
  residues.forEach(r => {{
    const n = parseInt(r.replace(/[^0-9]/g, ""), 10);
    viewer.addStyle({{resi: n}}, {{stick: {{color: "#d97706"}}}});
  }});
  viewer.zoomTo();
  viewer.render();
}});
</script>
</body></html>
"""
