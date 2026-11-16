"""Single-window UI for the live demo (no OpenCV GUI): one pyglet window owns the only GL context; the scene renders
offscreen into FBOs in that same context, the canvas (numpy) is blitted as a texture, and the rate-selection pop-up's
slider and GO button are drawn into the canvas and driven by pyglet mouse/key events. Headless mode uses a standalone
GL context. Works on macOS (Cocoa, GL 3.3 core) and Linux/X11.
The picture fills the whole window (no control band). The only on-screen controls are the rate-selection pop-up
(mode "select": the TOPEX Angular Rate Selection pop-up, omega-y slider + GO, GO greyed out until it is valid). Keys: Enter/S start, Space skip the
cutscene, G = GO, P = pause, Esc/Q = quit; hidden test keys X (pose error), L (rotate the sun), I (visible camera fails -> infrared)."""
import numpy as np, moderngl, cv2
import gl_render, uitext as UT

F = cv2.FONT_HERSHEY_SIMPLEX
S = 1   # canvas pixels per layout unit (set by demo.py; 2 = HiDPI canvas). Widget geometry stays in 1280 x 720 layout units.
def P(v): return int(round(v * S))

class Slider:
    def __init__(self, name, x, y, w, lo, hi, val): self.name, self.x, self.y, self.w, self.lo, self.hi, self.val = name, x, y, w, lo, hi, val
    def hit(self, mx, my): return self.x - 14 <= mx <= self.x + self.w + 14 and abs(my - self.y) <= 22
    def set_from_x(self, mx): u = np.clip((mx - self.x) / self.w, 0, 1); self.val = round(float(self.lo + u * (self.hi - self.lo)), 3)
    def draw(self, img, enabled=True):
        col = (235, 235, 235) if enabled else (110, 110, 110)
        cv2.line(img, (P(self.x), P(self.y)), (P(self.x + self.w), P(self.y)), (90, 90, 90), 3 * S, cv2.LINE_AA)
        x0 = self.x + int((0 - self.lo) / (self.hi - self.lo) * self.w); cv2.line(img, (P(x0), P(self.y - 7)), (P(x0), P(self.y + 7)), (110, 110, 110), S)
        u = (self.val - self.lo) / (self.hi - self.lo); cx = self.x + int(u * self.w)
        cv2.circle(img, (P(cx), P(self.y)), P(9), (255, 255, 255) if enabled else (80, 80, 80), -1, cv2.LINE_AA)   # white knob (Daniel 2026-11-12)
        lab = {"wx": "\u03c9x", "wy": "\u03c9y", "wz": "\u03c9z"}.get(self.name, self.name)          # real omega glyph
        UT.draw(img, [(lab + " ", False), (f"{self.val:+.3f}", True), (" rad/s", False)], self.x + self.w + 18, self.y - 10, 16, col)

class Button:
    def __init__(self, label, key, x, y, w=100, h=24): self.label, self.key, self.x, self.y, self.w, self.h = label, key, x, y, w, h
    def hit(self, mx, my): return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h
    def draw(self, img, hover=False, enabled=True):
        bg = ((50, 130, 70) if not hover else (70, 170, 95)) if enabled else (48, 48, 48)
        cv2.rectangle(img, (P(self.x), P(self.y)), (P(self.x + self.w), P(self.y + self.h)), bg, -1)
        cv2.rectangle(img, (P(self.x), P(self.y)), (P(self.x + self.w), P(self.y + self.h)), (230, 230, 230) if enabled else (90, 90, 90), S)
        (tw, th), _ = cv2.getTextSize(self.label, F, 0.6, 2 if enabled else 1)
        cv2.putText(img, self.label, (P(self.x + (self.w - tw) // 2), P(self.y + (self.h + th) // 2)), F, 0.6 * S, (250, 250, 250) if enabled else (110, 110, 110), (2 if enabled else 1) * S, cv2.LINE_AA)

QUAD_VS = "#version 330\nin vec2 p; out vec2 uv; void main(){ uv = vec2((p.x+1.0)*0.5, (1.0-p.y)*0.5); gl_Position = vec4(p,0,1); }"
QUAD_FS = "#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 f; void main(){ f = vec4(texture(tex, uv).rgb, 1.0); }"

class Display:
    """show(canvas_bgr) -> list of events: ('key', ch) / ('slider', name, value)."""
    PANEL = (300, 130, 980, 590)   # TOPEX angular-rate selection panel rect (x0, y0, x1, y1), centred on the 1280 x 720 picture
    def __init__(self, W, H, headless, title="SWARM live demo", wmax=1.0, w0=(0.0, 0.661, 0.0)):
        self.W, self.H, self.headless = W, H, headless; self.events = []; self.drag = None; self.closed = False; self.mode = "none"; self.hover = False
        self.go_enabled = False
        x0, y0, x1, y1 = self.PANEL
        wm = wmax if isinstance(wmax, (tuple, list)) else (wmax,) * 3   # per-axis slider range
        axes = [(n, v, m) for n, v, m in zip(("wx", "wy", "wz"), w0, wm) if m > 0]   # an axis capped at 0 gets no slider (Daniel: wx = wz = 0)
        self.sliders = [Slider(n, x0 + 40, y0 + 300 + 40 * i, 400, -m, m, v) for i, (n, v, m) in enumerate(axes)]
        self.go = Button("GO", "g", (x0 + x1) // 2 - 80, y1 - 62, 160, 40)
        if headless:
            self.ctx = gl_render.make_context(True); self.win = None; return
        import pyglet
        cfg = pyglet.gl.Config(major_version=3, minor_version=3, forward_compatible=True, double_buffer=True, depth_size=24)
        try: self.win = pyglet.window.Window(W, H, caption=title, config=cfg, resizable=True, vsync=False)
        except Exception: self.win = pyglet.window.Window(W, H, caption=title, resizable=True, vsync=False)
        self.win.set_minimum_size(320, 180)   # resizable: drag any edge / the green button goes fullscreen; the picture keeps 16:9 (black bars)
        self.ctx = moderngl.create_context()
        self.tex = None; self.tex_wh = None
        prog = self.ctx.program(vertex_shader=QUAD_VS, fragment_shader=QUAD_FS)
        vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.quad = self.ctx.vertex_array(prog, [(vbo, "2f", "p")])
        w = self.win
        def to_canvas(x, y):   # window point (pyglet: points, origin bottom-left) -> canvas units (1280 x 720), through the letterbox
            vx, vy, vw, vh = self.fit(w.width, w.height); return (x - vx) * W / vw, (vy + vh - y) * H / vh
        @w.event
        def on_key_press(sym, mod):
            import pyglet.window.key as k
            if sym == k.F: w.set_fullscreen(not w.fullscreen); return True   # F also toggles fullscreen
            m = {k.G: "g", k.X: "x", k.L: "l", k.I: "i", k.Q: "q", k.ESCAPE: "q", k.SPACE: " ", k.R: "r", k.ENTER: "\r", k.RETURN: "\r", k.S: "\r", k.P: "p"}
            if sym in m: self.events.append(("key", m[sym]))
            if self.mode == "select" and sym in (k.LEFT, k.RIGHT, k.UP, k.DOWN) and self.sliders:   # arrows nudge wy by 0.01 rad/s
                s_ = self.sliders[0]; s_.val = round(float(np.clip(s_.val + (0.01 if sym in (k.RIGHT, k.UP) else -0.01), s_.lo, s_.hi)), 3)
            return True                                        # don't let pyglet close on ESC by itself
        @w.event
        def on_mouse_press(x, y, b, mod):
            if self.mode != "select": return
            cx, cy = to_canvas(x, y)
            for s in self.sliders:
                if s.hit(cx, cy): self.drag = s; s.set_from_x(cx); self.events.append(("slider", s.name, s.val)); return
            if self.go.hit(cx, cy) and self.go_enabled: self.events.append(("key", "g"))
        def follow(x, y):   # a held slider follows the pointer whichever event macOS delivers (drag or plain motion)
            if self.drag is not None: cx, _ = to_canvas(x, y); self.drag.set_from_x(cx); self.events.append(("slider", self.drag.name, self.drag.val))
        @w.event
        def on_mouse_motion(x, y, dx, dy):
            cx, cy = to_canvas(x, y); self.hover = self.mode == "select" and self.go.hit(cx, cy); follow(x, y)
        @w.event
        def on_mouse_drag(x, y, dx, dy, b, mod): follow(x, y)
        @w.event
        def on_mouse_release(x, y, b, mod): self.drag = None
        @w.event
        def on_close(): self.closed = True; self.events.append(("key", "q"))
    SENTENCE = ("This live demo will dock for a range of yaw rates on TOPEX. Introduce disturbances like lighting (press L), noise in the "
                "imager estimates (press X), or full visual imager failure, resulting in a back up to infrared (press I) to test the robustness "
                "of the computer vision software (running live in the physics sim) and autonomy algorithm.")   # Daniel 2026-11-12, verbatim
    def draw_select(self, img, go_enabled, lines):
        """the TOPEX angular-rate selection pop-up: white title, Daniel's sentence, the omega-y slider, the GO prompt, GO (greyed out until valid)."""
        self.mode = "select"; self.go_enabled = go_enabled; x0, y0, x1, y1 = self.PANEL
        sub = img[P(y0):P(y1) + 1, P(x0):P(x1) + 1]; ov = sub.copy(); ov[:] = (14, 12, 10); cv2.addWeighted(ov, 0.88, sub, 0.12, 0, sub)   # == full-frame rectangle + blend
        cv2.rectangle(img, (P(x0), P(y0)), (P(x1), P(y1)), (120, 120, 120), S)
        UT.draw(img, [("TOPEX Angular Rate Selection", True)], x0 + 40, y0 + 26, 26, (255, 255, 255))
        for j, ln in enumerate(UT.wrap(self.SENTENCE, 16, x1 - x0 - 80)): UT.draw(img, ln, x0 + 40, y0 + 82 + 24 * j, 16, (225, 225, 225))
        for s in self.sliders: s.draw(img, True)
        for j, ln in enumerate([l for l in lines if l]): UT.draw(img, ln, x0 + 40, y0 + 326 + 20 * j, 14, (120, 160, 255))
        UT.draw(img, "Press GO to initiate TOPEX spin and autonomous settling", (x0 + x1) // 2, y1 - 104, 17, (255, 255, 255), anchor="c")
        self.go.draw(img, hover=self.hover and go_enabled, enabled=go_enabled)
    def fit(self, fw, fh):
        s = min(fw / self.W, fh / self.H); vw, vh = int(round(self.W * s)), int(round(self.H * s))
        return ((fw - vw) // 2, (fh - vh) // 2, vw, vh)
    def framebuffer_px(self):   # the window's drawable size in device pixels (Retina: 2x its point size)
        return self.win.get_framebuffer_size() if self.win is not None else (self.W, self.H)
    def show(self, canvas_bgr, mode=None):
        if mode is not None: self.mode = mode
        if self.win is None: return []
        self.win.switch_to(); self.win.dispatch_events()
        h_, w_ = canvas_bgr.shape[:2]
        if self.tex_wh != (w_, h_):   # the canvas may be drawn at a multiple of 1280 x 720 (HiDPI)
            self.tex = self.ctx.texture((w_, h_), 3); self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); self.tex_wh = (w_, h_)
        self.tex.write(np.ascontiguousarray(canvas_bgr[:, :, ::-1]).tobytes()); self.tex.build_mipmaps()   # clean when the window is smaller than the canvas
        self.ctx.screen.use(); fw, fh = self.win.get_framebuffer_size(); self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0); self.ctx.viewport = self.fit(fw, fh)   # letterbox: largest 16:9 box, centred
        self.ctx.disable(moderngl.DEPTH_TEST); self.tex.use(0); self.quad.render(moderngl.TRIANGLE_STRIP)
        self.win.flip(); ev = self.events; self.events = []; return ev
