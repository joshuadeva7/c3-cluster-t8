"""Single-window UI for the live demo (no OpenCV GUI): one pyglet window owns the only GL context; the scene renders
offscreen into FBOs in that same context, the dashboard image (numpy) is blitted as a texture, and the sliders/buttons
are drawn into the dashboard and driven by pyglet mouse/key events. Headless mode uses a standalone GL context.
Works on macOS (Cocoa, GL 3.3 core) and Linux/X11."""
import numpy as np, moderngl, cv2
import gl_render

class Slider:
    def __init__(self, name, x, y, w, lo, hi, val): self.name, self.x, self.y, self.w, self.lo, self.hi, self.val = name, x, y, w, lo, hi, val
    def hit(self, mx, my): return self.x - 8 <= mx <= self.x + self.w + 8 and abs(my - self.y) <= 12
    def set_from_x(self, mx): u = np.clip((mx - self.x) / self.w, 0, 1); self.val = round(float(self.lo + u * (self.hi - self.lo)), 2)
    def draw(self, img, enabled=True):
        col = (230, 230, 230) if enabled else (110, 110, 110)
        cv2.line(img, (self.x, self.y), (self.x + self.w, self.y), (90, 90, 90), 3, cv2.LINE_AA)
        x0 = self.x + int((0 - self.lo) / (self.hi - self.lo) * self.w); cv2.line(img, (x0, self.y - 6), (x0, self.y + 6), (90, 90, 90), 1)
        u = (self.val - self.lo) / (self.hi - self.lo); cx = self.x + int(u * self.w)
        cv2.circle(img, (cx, self.y), 7, (60, 170, 255) if enabled else (80, 80, 80), -1, cv2.LINE_AA)
        cv2.putText(img, f"{self.name} {self.val:+.2f} rad/s", (self.x + self.w + 14, self.y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

class Button:
    def __init__(self, label, key, x, y, w=70, h=24): self.label, self.key, self.x, self.y, self.w, self.h = label, key, x, y, w, h
    def hit(self, mx, my): return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h
    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (70, 70, 70), -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)
        cv2.putText(img, self.label, (self.x + 8, self.y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

QUAD_VS = "#version 330\nin vec2 p; out vec2 uv; void main(){ uv = vec2((p.x+1.0)*0.5, (1.0-p.y)*0.5); gl_Position = vec4(p,0,1); }"
QUAD_FS = "#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 f; void main(){ f = vec4(texture(tex, uv).rgb, 1.0); }"

class Display:
    """show(canvas_bgr) -> list of events: ('key', ch) / ('slider', name, value) / ('button', key) / ('quit',)."""
    def __init__(self, W, H, headless, title="SWARM live demo", wmax=1.0, w0=(0.0, 0.66, 0.0)):
        self.W, self.H, self.headless = W, H, headless; self.events = []; self.drag = None; self.closed = False
        ys = H - 70
        self.sliders = [Slider(n, 20, ys + 22 * i, 300, -wmax, wmax, v) for i, (n, v) in enumerate(zip(("wx", "wy", "wz"), w0))]
        self.buttons = [Button("GO", "g", 470, ys - 10), Button("LOSS", "x", 550, ys - 10), Button("SUN", "l", 630, ys - 10),
                        Button("SKIP", " ", 470, ys + 22), Button("QUIT", "q", 550, ys + 22)]
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
            m = {k.G: "g", k.X: "x", k.L: "l", k.Q: "q", k.ESCAPE: "q", k.SPACE: " ", k.R: "r"}
            if sym in m: self.events.append(("key", m[sym]))
            return True                                        # don't let pyglet close on ESC by itself
        @w.event
        def on_mouse_press(x, y, b, mod):
            cx, cy = to_canvas(x, y)
            for s in self.sliders:
                if s.hit(cx, cy): self.drag = s; s.set_from_x(cx); self.events.append(("slider", s.name, s.val)); return
            for bt in self.buttons:
                if bt.hit(cx, cy): self.events.append(("key", bt.key)); return
        @w.event
        def on_mouse_drag(x, y, dx, dy, b, mod):
            if self.drag is not None: cx, _ = to_canvas(x, y); self.drag.set_from_x(cx); self.events.append(("slider", self.drag.name, self.drag.val))
        @w.event
        def on_mouse_release(x, y, b, mod): self.drag = None
        @w.event
        def on_close(): self.closed = True; self.events.append(("key", "q"))
    def draw_controls(self, img, sliders_enabled=True):
        for s in self.sliders: s.draw(img, sliders_enabled)
        for b in self.buttons: b.draw(img)
    def show(self, canvas_bgr):
        if self.win is None: return []
        self.win.switch_to(); self.win.dispatch_events()
        self.tex.write(np.ascontiguousarray(canvas_bgr[:, :, ::-1]).tobytes())
        self.ctx.screen.use(); fw, fh = self.win.get_framebuffer_size(); self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.disable(moderngl.DEPTH_TEST); self.tex.use(0); self.quad.render(moderngl.TRIANGLE_STRIP)
        self.win.flip(); ev = self.events; self.events = []; return ev
