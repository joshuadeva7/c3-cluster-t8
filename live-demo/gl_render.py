"""Minimal moderngl renderer for the live demo: replaces pyrender (whose macOS path needs a pyglet window + screen).
One GL context for everything: a standalone context (EGL on Linux, CGL on macOS) when headless, or the context of the
pyglet window in window mode. Flat-shaded per-face colours, one directional sun + ambient, two-sided lighting.
render() returns RGB uint8 (H,W,3) and metric depth float32 (H,W) (0 = background), like pyrender.
Camera convention: OpenCV (x right, y down, z forward); camera pose given as camera->world rotation and position."""
import sys, numpy as np, moderngl

VS = """#version 330
uniform mat4 M; uniform mat3 Mn; uniform mat4 V; uniform mat4 P;
in vec3 in_p; in vec3 in_n; in vec3 in_c;
out vec3 n_w; out vec3 c; out float zc;
void main(){ vec4 pw = M*vec4(in_p,1.0); vec4 pc = V*pw; zc = pc.z; n_w = Mn*in_n; c = in_c; gl_Position = P*pc; }"""
FS = """#version 330
uniform vec3 sun; uniform float amb; uniform float kd; uniform vec3 cam_fwd_w; uniform float kh;
in vec3 n_w; in vec3 c; in float zc;
out vec4 f_col;
void main(){ vec3 n = normalize(n_w); if (!gl_FrontFacing) n = -n;
  float l = max(dot(n, sun), 0.0); float h = max(dot(n, -cam_fwd_w), 0.0);
  vec3 col = c*(amb + kd*l + kh*h); f_col = vec4(pow(clamp(col,0.0,1.0), vec3(1.0/2.2)), 1.0); }"""

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
        n = np.repeat(np.asarray(mesh.face_normals, np.float32), 3, axis=0)
        fc = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.float32) / 255.0
        c = np.repeat(fc ** 2.2, 3, axis=0)                 # sRGB -> linear albedo
        vbo = self.ctx.buffer(np.c_[v, n, c].astype(np.float32).tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_p", "in_n", "in_c")])
        self.objs.append(dict(vao=vao, pose=np.array(pose, float), visible=True)); return len(self.objs) - 1
    def set_pose(self, i, T): self.objs[i]["pose"] = np.array(T, float)
    def set_visible(self, i, vis): self.objs[i]["visible"] = bool(vis)
    def _fbo(self, w, h):
        if (w, h) not in self.fbos:
            # colour RGBA8 + 32-bit float depth TEXTURE; metric depth is recovered from the standard depth buffer
            # (no float colour attachment / MRT: those read back as zeros on some drivers, e.g. Apple's GL-on-Metal)
            col = self.ctx.renderbuffer((w, h), 4); d = self.ctx.depth_texture((w, h))
            self.fbos[(w, h)] = self.ctx.framebuffer(color_attachments=[col], depth_attachment=d)
        return self.fbos[(w, h)]
    def render(self, K, W, H, R_IC, p_IC, sun_w, znear=0.3, zfar=2000.0, headlight=0.0):
        n, f = znear, zfar
        P = np.array([[2 * K[0, 0] / W, 0, 2 * K[0, 2] / W - 1, 0],
                      [0, -2 * K[1, 1] / H, 1 - 2 * K[1, 2] / H, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, 1, 0]], np.float64)
        V = np.eye(4); V[:3, :3] = R_IC.T; V[:3, 3] = -R_IC.T @ p_IC
        fbo = self._fbo(W, H); fbo.use(); fbo.clear(0, 0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.disable(moderngl.CULL_FACE)
        s = np.asarray(sun_w, float); s = s / np.linalg.norm(s)
        pr = self.prog; pr["V"].write(V.T.astype("f4").tobytes()); pr["P"].write(P.T.astype("f4").tobytes())
        pr["sun"].value = tuple(s); pr["amb"].value = self.amb; pr["kd"].value = self.kd
        pr["kh"].value = float(headlight); pr["cam_fwd_w"].value = tuple(R_IC[:, 2])
        for o in self.objs:
            if not o["visible"]: continue
            M = o["pose"]; pr["M"].write(M.T.astype("f4").tobytes())
            Mn = np.linalg.inv(M[:3, :3]).T; pr["Mn"].write(Mn.T.astype("f4").tobytes())
            o["vao"].render(moderngl.TRIANGLES)
        col = np.frombuffer(fbo.read(components=3, attachment=0), np.uint8).reshape(H, W, 3)[::-1]
        d = np.frombuffer(fbo.read(components=1, attachment=-1, dtype="f4"), np.float32).reshape(H, W)[::-1].astype(np.float64)
        zn = 2.0 * d - 1.0; z = (2.0 * f * n) / ((f + n) - zn * (f - n)); z[d >= 1.0 - 1e-7] = 0.0
        return np.ascontiguousarray(col), np.ascontiguousarray(z.astype(np.float32))
