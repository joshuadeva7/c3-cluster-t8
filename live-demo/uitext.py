"""Anti-aliased UI text (DejaVu Sans, bundled in assets/fonts: renders the real omega glyph) drawn into BGR numpy canvases.
Colours are BGR tuples like the rest of the canvas code."""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
S = 1   # canvas pixels per layout unit (2 on a HiDPI canvas): every x / y / size below is in layout units (1280 x 720 picture)
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts"); _FC = {}
def font(sz, bold=False):
    k = (int(sz), bool(bold))
    if k not in _FC: _FC[k] = ImageFont.truetype(os.path.join(_D, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), int(sz))
    return _FC[k]
def _pw(s, sz, bold=False): return font(sz * S, bold).getlength(s)          # width in canvas pixels
def width(s, sz, bold=False): return _pw(s, sz, bold) / S                   # width in layout units
def segs_width(segs, sz): return sum(width(s, sz, b) for s, b in segs)
def draw(img, segs, x, y, sz=14, col=(255, 255, 255), alpha=1.0, anchor="l"):
    """segs: str or [(text, bold)]. (x, y) = left (or centre / right with anchor 'c' / 'r') of the text's top line box. Returns the bbox."""
    if isinstance(segs, str): segs = [(segs, False)]
    if alpha <= 0.01: return None
    x, y, szp = x * S, y * S, sz * S   # -> canvas pixels
    tw = sum(_pw(s, sz, b) for s, b in segs); x0 = x - (tw / 2 if anchor == "c" else (tw if anchor == "r" else 0))
    H, W = img.shape[:2]; pad = 4 * S; bx0, by0 = int(max(0, x0 - pad)), int(max(0, y - pad)); bx1, by1 = int(min(W, x0 + tw + pad)), int(min(H, y + szp * 1.35 + pad))
    if bx1 <= bx0 or by1 <= by0: return None
    crop = np.ascontiguousarray(img[by0:by1, bx0:bx1]); pim = Image.fromarray(crop); d = ImageDraw.Draw(pim); xx = x0 - bx0
    for s, b in segs: d.text((xx, y - by0), s, font=font(szp, b), fill=tuple(int(c) for c in col)); xx += _pw(s, sz, b)
    out = np.asarray(pim)
    img[by0:by1, bx0:bx1] = out if alpha >= 0.99 else (crop * (1 - alpha) + out * alpha).astype(np.uint8)
    return (bx0 / S, by0 / S, bx1 / S, by1 / S) if S != 1 else (bx0, by0, bx1, by1)   # bbox in layout units
def wrap(s, sz, maxw, bold=False):
    lines, cur = [], ""
    for w in s.split(" "):
        t = (cur + " " + w).strip()
        if width(t, sz, bold) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines
