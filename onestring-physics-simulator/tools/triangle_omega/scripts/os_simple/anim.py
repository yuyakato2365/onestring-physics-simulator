"""Frame rendering and video assembly shared by all stages (matplotlib, Agg)."""
from __future__ import annotations
import os, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection, LineCollection

LIGHT = np.array([0.35, 0.8, 0.5]); LIGHT /= np.linalg.norm(LIGHT)
INK = "#1d2733"; SEAM = "#c8324f"; ACC = "#1f5fbf"; PANEL = "#dfe6ee"


def shade(V, polys, base=(0.80, 0.84, 0.90), colors=None):
    P = [V[p] for p in polys]
    out = []
    for i, q in enumerate(P):
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if len(q) == 4:
            n = n + np.cross(q[2] - q[0], q[3] - q[0])
        nn = np.linalg.norm(n)
        lam = 0.35 + 0.65 * abs(float(np.dot(n / nn, LIGHT))) if nn > 0 else 0.6
        b = np.array(colors[i] if colors is not None else base)
        out.append(np.clip(b * lam, 0, 1))
    return P, out


def setup3d(ax, bounds, elev=12, azim=-62):
    lo, hi = bounds
    c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.02
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(c[1] - r, c[1] + r)
    ax.view_init(elev=elev, azim=azim); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def yup(V):
    V = np.asarray(V, float)
    return np.c_[V[:, 0], V[:, 2], V[:, 1]]           # model is y-up; matplotlib is z-up


def draw_mesh3d(ax, V, polys, colors=None, edge=None, lw=0.15, base=(0.80, 0.84, 0.90), alpha=1.0, lines=None):
    W = yup(V)
    P, col = shade(W, polys, base=base, colors=colors)
    pc = Poly3DCollection(P, facecolors=col, edgecolors=(edge if edge is not None else col), linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)
    if lines is not None:
        for seg, c, w in lines:
            seg = yup(seg)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, lw=w, zorder=10)


class Recorder:
    """Writes numbered PNG frames and turns them into an MP4 (and a light GIF)."""

    def __init__(self, folder, name):
        self.disabled = os.environ.get("TRI_SKIP_VIDEOS") == "1"
        if self.disabled: return
        self.dir = os.path.join(folder, name + "_frames"); self.name = name; self.folder = folder
        shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.n = 0

    def add(self, fig):
        if self.disabled:
            plt.close(fig)
            return
        fig.savefig(os.path.join(self.dir, f"f{self.n:05d}.png"), dpi=fig.dpi)
        plt.close(fig); self.n += 1

    def finish(self, fps=12, hold_last=1.5):
        if self.disabled or self.n == 0:
            return None
        last = os.path.join(self.dir, f"f{self.n - 1:05d}.png")
        for k in range(int(hold_last * fps)):
            shutil.copy(last, os.path.join(self.dir, f"f{self.n + k:05d}.png"))
        mp4 = os.path.join(self.folder, self.name + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(self.dir, "f%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "23", mp4], check=True)
        shutil.copy(last, os.path.join(self.folder, self.name + "_final.png"))
        return mp4


def pick_frames(n_total, n_max):
    """Indices of at most n_max frames spread over n_total, always keeping the first and the last."""
    if os.environ.get("TRI_SKIP_VIDEOS") == "1": return []
    if n_total <= n_max:
        return list(range(n_total))
    idx = np.unique(np.round(np.geomspace(1, n_total, n_max)).astype(int) - 1)
    return sorted(set(idx.tolist()) | {0, n_total - 1})
