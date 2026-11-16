"""Minimal moderngl renderer for the live demo: replaces pyrender (whose macOS path needs a pyglet window + screen).
One GL context for everything: a standalone context (EGL on Linux, CGL on macOS) when headless, or the context of the
pyglet window in window mode. Flat-shaded per-face colours, one directional sun + ambient, two-sided lighting.
render() returns RGB uint8 (H,W,3) and metric depth float32 (H,W) (0 = background), like pyrender.
Camera convention: OpenCV (x right, y down, z forward); camera pose given as camera->world rotation and position."""
import sys, numpy as np, moderngl, cv2

VS = """#version 330
uniform mat4 M; uniform mat3 Mn; uniform mat4 V; uniform mat4 P;
in vec3 in_p; in vec3 in_n; in vec3 in_c;
out vec3 n_w; out vec3 c; out float zc; out vec3 p_w;
void main(){ vec4 pw = M*vec4(in_p,1.0); p_w = pw.xyz; vec4 pc = V*pw; zc = pc.z; n_w = Mn*in_n; c = in_c; gl_Position = P*pc; }"""
FS = """#version 330
uniform vec3 sun; uniform float amb; uniform float kd; uniform vec3 cam_fwd_w; uniform vec3 cam_pos_w; uniform float kh; uniform float hd0; uniform float v2look;
in vec3 n_w; in vec3 c; in float zc; in vec3 p_w;
out vec4 f_col;
void main(){ vec3 n = normalize(n_w); if (!gl_FrontFacing) n = -n;
  if (v2look > 0.5) {   // the v2 video's flat shading: face colour (sRGB) x (0.32 + 0.78 sun + 0.22 earthshine from -x)
    if (dot(n, cam_pos_w - p_w) < 0.0) n = -n;
    float sh = 0.32 + 0.78 * max(dot(n, sun), 0.0) + 0.22 * max(-n.x, 0.0);
    f_col = vec4(clamp(pow(c, vec3(1.0/2.2)) * sh, 0.0, 1.0), 1.0); return; }
  float l = max(dot(n, sun), 0.0);
  // illuminator: point spotlight at the camera, inverse-square-like falloff and a soft 30 deg cone (no flat grey wash)
  vec3 L = cam_pos_w - p_w; float d = length(L); L /= max(d, 1e-4);
  float cone = smoothstep(0.80, 0.97, dot(-L, cam_fwd_w));
  float h = max(dot(n, L), 0.0) * cone / (1.0 + (d / hd0) * (d / hd0));
  vec3 col = c*(amb + kd*l + kh*h); f_col = vec4(pow(clamp(col,0.0,1.0), vec3(1.0/2.2)), 1.0); }"""

def _find_glflush():   # glFlush without PyOpenGL (the Mac venv has none): straight from the system GL library
    import ctypes, ctypes.util, sys
    names = ["/System/Library/Frameworks/OpenGL.framework/OpenGL"] if sys.platform == "darwin" else [ctypes.util.find_library("GL"), "libGL.so.1", "libOpenGL.so.0"]
    for nm in names:
        try:
            if nm: return ctypes.CDLL(nm).glFlush
        except Exception: pass
    return None
_glFlush = _find_glflush()
def make_context(standalone=True):
    if not standalone: return moderngl.create_context()
    if sys.platform.startswith("linux"): return moderngl.create_standalone_context(backend="egl")
    return moderngl.create_standalone_context()          # macOS: CGL, no window server screen needed

class GLScene:
    def __init__(self, ctx=None, ambient=0.035, kd=1.25):
        self.ctx = ctx or make_context(True); self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.objs = []; self.amb = ambient; self.kd = kd; self.fbos = {}; self.sun = np.array([0, 0, 1.0])
    def add(self, mesh, pose=np.eye(4)):
        v = np.asarray(mesh.vertices, np.float32)[np.asarray(mesh.faces)].reshape(-1, 3)
        cn = mesh.metadata.get("cn") if hasattr(mesh, "metadata") else None   # optional (F,3,3) smoothed corner normals (display meshes)
        n = np.asarray(cn, np.float32).reshape(-1, 3) if cn is not None else np.repeat(np.asarray(mesh.face_normals, np.float32), 3, axis=0)
        fc = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.float32) / 255.0
        c = np.repeat(fc ** 2.2, 3, axis=0)                 # sRGB -> linear albedo
        vbo = self.ctx.buffer(np.c_[v, n, c].astype(np.float32).tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_p", "in_n", "in_c")])
        lo, hi = v.min(0).astype(float), v.max(0).astype(float)   # local AABB corners: bound the screen rect a display render can touch
        box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self.objs.append(dict(vao=vao, pose=np.array(pose, float), visible=True, box=box)); return len(self.objs) - 1
    def add_hq(self, i, mesh):
        """alternate (display-quality) geometry for object i, drawn instead of it by render(..., hq=True); shares its pose/visibility."""
        j = self.add(mesh); o = self.objs.pop(j); self.objs[i]["vao_hq"] = o["vao"]
        b = np.r_[self.objs[i]["box"], o["box"]]; lo, hi = b.min(0), b.max(0)
        self.objs[i]["box"] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    def set_pose(self, i, T): self.objs[i]["pose"] = np.array(T, float)
    def set_visible(self, i, vis): self.objs[i]["visible"] = bool(vis)
    def _fbo_new(self, w, h):
        col = self.ctx.renderbuffer((w, h), 4); d = self.ctx.depth_texture((w, h))
        return self.ctx.framebuffer(color_attachments=[col], depth_attachment=d)
    def _fbo_ms(self, w, h, samples):
        k = ("ms", w, h, samples)
        if k not in self.fbos:   # multisampled colour + depth, resolved into a plain RGBA8 target (alpha = pixel coverage)
            ms = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4, samples=samples)],
                                      depth_attachment=self.ctx.depth_renderbuffer((w, h), samples=samples))
            rs = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4)])
            self.fbos[k] = (ms, rs)
        return self.fbos[k]
    def _fbo(self, w, h, slot=0):   # slot > 0: a separate target, so several submitted renders can be fetched later
        if slot: k = (w, h, slot); self.fbos[k] = self.fbos.get(k) or self._fbo_new(w, h); return self.fbos[k]
        if (w, h) not in self.fbos:
            # colour RGBA8 + 32-bit float depth TEXTURE; metric depth is recovered from the standard depth buffer
            # (no float colour attachment / MRT: those read back as zeros on some drivers, e.g. Apple's GL-on-Metal)
            col = self.ctx.renderbuffer((w, h), 4); d = self.ctx.depth_texture((w, h))
            self.fbos[(w, h)] = self.ctx.framebuffer(color_attachments=[col], depth_attachment=d)
        return self.fbos[(w, h)]
    def _screen_rect(self, K, W, H, V, n):
        """pixel rect [x0,x1) x [y0,y1) (image rows top-down) holding every visible object's projected AABB, padded 3 px; None if
        any corner is at/behind the near plane (then read the whole frame)."""
        x0 = y0 = 1e18; x1 = y1 = -1e18; any_ = False
        for o in self.objs:
            if not o["visible"]: continue
            M = V @ o["pose"]; c = o["box"] @ M[:3, :3].T + M[:3, 3]; z = c[:, 2]
            if (z <= n).any(): return None
            u = K[0, 0] * c[:, 0] / z + K[0, 2]; w = K[1, 1] * c[:, 1] / z + K[1, 2]
            x0 = min(x0, u.min()); x1 = max(x1, u.max()); y0 = min(y0, w.min()); y1 = max(y1, w.max()); any_ = True
        if not any_: return (0, 0, 0, 0)
        x0 = int(max(0, min(W, np.floor(x0) - 3))); x1 = int(max(0, min(W, np.ceil(x1) + 4)))
        y0 = int(max(0, min(H, np.floor(y0) - 3))); y1 = int(max(0, min(H, np.ceil(y1) + 4)))
        return (x0, y0, x1, y1)
    def render(self, *a, **k): return self.submit(*a, **k)()
    def submit(self, K, W, H, R_IC, p_IC, sun_w, znear=0.3, zfar=2000.0, headlight=0.0, amb_override=None, v2look=False, depth=True, msaa=0, hq=False, slot=0):
        """issue the draw calls now (the GPU runs them asynchronously) and return fetch(): the readback, = render()'s result.
        Calling fetch() later lets CPU work overlap the GPU frame instead of stalling on it. Nothing else may use this
        renderer's framebuffers between submit() and fetch()."""
        """depth=True: (RGB, metric depth). depth=False: (RGB, None). msaa=N (display views only): N-sample anti-aliased edges,
        returns (RGB premultiplied over black, coverage alpha uint8 0..255) -- composite as img*(1-a/255) + rgb."""
        n, f = znear, zfar
        P = np.array([[2 * K[0, 0] / W, 0, 2 * K[0, 2] / W - 1, 0],
                      [0, -2 * K[1, 1] / H, 1 - 2 * K[1, 2] / H, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, 1, 0]], np.float64)
        if msaa: P[1] = [0, 2 * K[1, 1] / H, 2 * K[1, 2] / H - 1, 0]   # display views: image row 0 -> GL row 0, readback already top-down (no flip copy)
        V = np.eye(4); V[:3, :3] = R_IC.T; V[:3, 3] = -R_IC.T @ p_IC
        if msaa: fms, frs = self._fbo_ms(W, H, msaa); fbo = fms
        else: fbo = self._fbo(W, H, slot)
        fbo.use(); fbo.clear(0, 0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.disable(moderngl.CULL_FACE); self.ctx.front_face = "cw" if msaa else "ccw"   # y flipped -> winding flipped
        s = np.asarray(sun_w, float); s = s / np.linalg.norm(s)
        pr = self.prog; pr["V"].write(V.T.astype("f4").tobytes()); pr["P"].write(P.T.astype("f4").tobytes())
        pr["sun"].value = tuple(s); pr["amb"].value = self.amb if amb_override is None else float(amb_override); pr["kd"].value = self.kd
        pr["kh"].value = float(headlight); pr["cam_fwd_w"].value = tuple(R_IC[:, 2])
        pr["cam_pos_w"].value = tuple(np.asarray(p_IC, float)); pr["hd0"].value = 2.5; pr["v2look"].value = 1.0 if v2look else 0.0
        for o in self.objs:
            if not o["visible"]: continue
            M = o["pose"]; pr["M"].write(M.T.astype("f4").tobytes())
            Mn = np.linalg.inv(M[:3, :3]).T; pr["Mn"].write(Mn.T.astype("f4").tobytes())
            (o.get("vao_hq", o["vao"]) if hq else o["vao"]).render(moderngl.TRIANGLES)
        self.ctx.front_face = "ccw"
        r = self._screen_rect(K, W, H, V, n)   # (poses as drawn) only this rect can differ from the clear values
        if msaa: self.ctx.copy_framebuffer(frs, fms)
        if _glFlush is not None: _glFlush()   # start the GPU on it now, so a deferred fetch() overlaps CPU work
        return lambda: self._fetch(fbo, frs if msaa else None, W, H, n, f, depth, msaa, r)
    def _fetch(self, fbo, frs, W, H, n, f, depth, msaa, r):
        if msaa:   # returns (RGB premultiplied, coverage alpha uint8 0..255); reads back only rect r (outside it alpha and rgb are 0)
            if r is None:
                rgba = np.empty((H, W, 4), np.uint8); frs.read_into(rgba, components=4, attachment=0)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB), np.ascontiguousarray(rgba[:, :, 3])
            x0, y0, x1, y1 = r; rgb = np.zeros((H, W, 3), np.uint8); al = np.zeros((H, W), np.uint8)
            if x1 > x0 and y1 > y0:
                sub = np.empty((y1 - y0, x1 - x0, 4), np.uint8); frs.read_into(sub, viewport=(x0, y0, x1 - x0, y1 - y0), components=4, attachment=0)
                rgb[y0:y1, x0:x1] = sub[:, :, :3]; al[y0:y1, x0:x1] = sub[:, :, 3]
            return rgb, al
        if r is not None:   # read only rect r: outside it colour stays the clear 0 and depth the clear 1.0 (-> metric 0)
            x0, y0, x1, y1 = r; col = np.zeros((H, W, 3), np.uint8); zf = np.zeros((H, W), np.float32) if depth else None
            if x1 > x0 and y1 > y0:   # image rows are flipped vs GL rows: image row v = GL row H-1-v
                vp = (x0, H - y1, x1 - x0, y1 - y0)
                sub = np.empty((y1 - y0, x1 - x0, 3), np.uint8); fbo.read_into(sub, viewport=vp, components=3, attachment=0)
                col[y0:y1, x0:x1] = sub[::-1]
                if depth:
                    d = np.empty((y1 - y0, x1 - x0), np.float32); fbo.read_into(d, viewport=vp, components=1, attachment=-1, dtype="f4")
                    zf[y0:y1, x0:x1] = self._metric(d[::-1].astype(np.float64), n, f)
            return col, zf
        col = np.empty((H, W, 3), np.uint8); fbo.read_into(col, components=3, attachment=0); col = cv2.flip(col, 0)
        if not depth: return col, None
        d = np.empty((H, W), np.float32); fbo.read_into(d, components=1, attachment=-1, dtype="f4")
        return col, self._metric(d[::-1].astype(np.float64), n, f)
    @staticmethod
    def _metric(d, n, f):
        # z = 2fn / ((f+n) - (2d-1)(f-n)), the same float64 operations in the same order, done in place (bit-identical, fewer passes)
        z = d * 2.0; z -= 1.0; z *= (f - n); np.subtract(f + n, z, out=z); np.divide(2.0 * f * n, z, out=z); z[d >= 1.0 - 1e-7] = 0.0
        return z.astype(np.float32)
