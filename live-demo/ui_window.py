"""Single-window UI for the live demo (no OpenCV GUI): one pyglet window owns the only GL context; the scene renders
offscreen into FBOs in that same context, the canvas (numpy) is blitted as a texture, and the docker-selection panel's
sliders and GO button are drawn into the canvas and driven by pyglet mouse/key events. Headless mode uses a standalone
GL context. Works on macOS (Cocoa, GL 3.3 core) and Linux/X11.
The picture fills the whole window (no control band). The only on-screen controls are the docker-selection panel
(mode "select": 3 TOPEX body-rate sliders + GO, GO greyed out until it is valid). Keys: Enter/S start, Space skip the
cutscene, G = GO, P = pause, Esc/Q = quit; hidden test keys X (tracking loss) and L (rotate the sun)."""
import numpy as np, moderngl, cv2
import gl_render

F = cv2.FONT_HERSHEY_SIMPLEX

class Slider:
    def __init__(self, name, x, y, w, lo, hi, val): self.name, self.x, self.y, self.w, self.lo, self.hi, self.val = name, x, y, w, lo, hi, val
    def hit(self, mx, my): return self.x - 8 <= mx <= self.x + self.w + 8 and abs(my - self.y) <= 12
    def set_from_x(self, mx): u = np.clip((mx - self.x) / self.w, 0, 1); self.val = round(float(self.lo + u * (self.hi - self.lo)), 3)
    def draw(self, img, enabled=True):
        col = (235, 235, 235) if enabled else (110, 110, 110)
        cv2.line(img, (self.x, self.y), (self.x + self.w, self.y), (90, 90, 90), 3, cv2.LINE_AA)
        x0 = self.x + int((0 - self.lo) / (self.hi - self.lo) * self.w); cv2.line(img, (x0, self.y - 7), (x0, self.y + 7), (110, 110, 110), 1)
        u = (self.val - self.lo) / (self.hi - self.lo); cx = self.x + int(u * self.w)
        cv2.circle(img, (cx, self.y), 8, (60, 170, 255) if enabled else (80, 80, 80), -1, cv2.LINE_AA)
        cv2.putText(img, f"{self.name} {self.val:+.3f} rad/s", (self.x + self.w + 16, self.y + 5), F, 0.48, col, 1, cv2.LINE_AA)

class Button:
    def __init__(self, label, key, x, y, w=100, h=24): self.label, self.key, self.x, self.y, self.w, self.h = label, key, x, y, w, h
    def hit(self, mx, my): return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h
    def draw(self, img, hover=False, enabled=True):
        bg = ((50, 130, 70) if not hover else (70, 170, 95)) if enabled else (48, 48, 48)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (230, 230, 230) if enabled else (90, 90, 90), 1)
        (tw, th), _ = cv2.getTextSize(self.label, F, 0.6, 2 if enabled else 1)
        cv2.putText(img, self.label, (self.x + (self.w - tw) // 2, self.y + (self.h + th) // 2), F, 0.6, (250, 250, 250) if enabled else (110, 110, 110), 2 if enabled else 1, cv2.LINE_AA)

QUAD_VS = "#version 330\nin vec2 p; out vec2 uv; void main(){ uv = vec2((p.x+1.0)*0.5, (1.0-p.y)*0.5); gl_Position = vec4(p,0,1); }"
QUAD_FS = "#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 f; void main(){ f = vec4(texture(tex, uv).rgb, 1.0); }"

class Display:
    """show(canvas_bgr) -> list of events: ('key', ch) / ('slider', name, value)."""
    PANEL = (370, 170, 910, 530)   # docker-selection panel rect (x0, y0, x1, y1), centred on the 1280 x 720 picture
    def __init__(self, W, H, headless, title="SWARM live demo", wmax=1.0, w0=(0.0, 0.661, 0.0)):
        self.W, self.H, self.headless = W, H, headless; self.events = []; self.drag = None; self.closed = False; self.mode = "none"; self.hover = False
        self.go_enabled = False
        x0, y0, x1, y1 = self.PANEL
        wm = wmax if isinstance(wmax, (tuple, list)) else (wmax,) * 3   # per-axis slider range
        self.sliders = [Slider(n, x0 + 40, y0 + 170 + 40 * i, 280, -m, m, v) for i, (n, v, m) in enumerate(zip(("wx", "wy", "wz"), w0, wm))]
        self.go = Button("GO", "g", (x0 + x1) // 2 - 80, y1 - 62, 160, 40)
        if headless:
            self.ctx = gl_render.make_context(True); self.win = None; return
        import pyglet
        cfg = pyglet.gl.Config(major_version=3, minor_version=3, forward_compatible=True, double_buffer=True, depth_size=24)
        try: self.win = pyglet.window.Window(W, H, caption=title, config=cfg, resizable=False, vsync=False)
        except Exception: self.win = pyglet.window.Window(W, H, caption=title, resizable=False, vsync=False)
        self.ctx = moderngl.create_context()
        self.tex = self.ctx.texture((W, H), 3); self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        prog = self.ctx.program(vertex_shader=QUAD_VS, fragment_shader=QUAD_FS)
        vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.quad = self.ctx.vertex_array(prog, [(vbo, "2f", "p")])
        w = self.win
        def to_canvas(x, y): return x * W / w.width, (w.height - y) * H / w.height
        @w.event
        def on_key_press(sym, mod):
            import pyglet.window.key as k
            m = {k.G: "g", k.X: "x", k.L: "l", k.Q: "q", k.ESCAPE: "q", k.SPACE: " ", k.R: "r", k.ENTER: "\r", k.RETURN: "\r", k.S: "\r", k.P: "p"}
            if sym in m: self.events.append(("key", m[sym]))
            return True                                        # don't let pyglet close on ESC by itself
        @w.event
        def on_mouse_press(x, y, b, mod):
            if self.mode != "select": return
            cx, cy = to_canvas(x, y)
            for s in self.sliders:
                if s.hit(cx, cy): self.drag = s; s.set_from_x(cx); self.events.append(("slider", s.name, s.val)); return
            if self.go.hit(cx, cy) and self.go_enabled: self.events.append(("key", "g"))
        @w.event
        def on_mouse_motion(x, y, dx, dy):
            cx, cy = to_canvas(x, y); self.hover = self.mode == "select" and self.go.hit(cx, cy)
        @w.event
        def on_mouse_drag(x, y, dx, dy, b, mod):
            if self.drag is not None: cx, _ = to_canvas(x, y); self.drag.set_from_x(cx); self.events.append(("slider", self.drag.name, self.drag.val))
        @w.event
        def on_mouse_release(x, y, b, mod): self.drag = None
        @w.event
        def on_close(): self.closed = True; self.events.append(("key", "q"))
    def draw_select(self, img, go_enabled, lines):
        """the docker-selection panel: title + context lines, the three TOPEX body-rate sliders, GO (greyed out until valid)."""
        self.mode = "select"; self.go_enabled = go_enabled; x0, y0, x1, y1 = self.PANEL
        ov = img.copy(); cv2.rectangle(ov, (x0, y0), (x1, y1), (14, 12, 10), -1); cv2.addWeighted(ov, 0.86, img, 0.14, 0, img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (120, 120, 120), 1)
        cv2.putText(img, "DOCKER SELECTION", (x0 + 40, y0 + 46), F, 0.85, (90, 175, 245), 2, cv2.LINE_AA)
        for j, ln in enumerate(lines): cv2.putText(img, ln, (x0 + 40, y0 + 80 + 22 * j), F, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
        cv2.putText(img, "TOPEX body rates (spin starts on GO)", (x0 + 40, y0 + 140), F, 0.5, (170, 200, 230), 1, cv2.LINE_AA)
        for s in self.sliders: s.draw(img, True)
        self.go.draw(img, hover=self.hover and go_enabled, enabled=go_enabled)
    def show(self, canvas_bgr, mode=None):
        if mode is not None: self.mode = mode
        if self.win is None: return []
        self.win.switch_to(); self.win.dispatch_events()
        self.tex.write(np.ascontiguousarray(canvas_bgr[:, :, ::-1]).tobytes())
        self.ctx.screen.use(); fw, fh = self.win.get_framebuffer_size(); self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.disable(moderngl.DEPTH_TEST); self.tex.use(0); self.quad.render(moderngl.TRIANGLE_STRIP)
        self.win.flip(); ev = self.events; self.events = []; return ev
