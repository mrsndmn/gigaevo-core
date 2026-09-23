"""Conformal geometric algebra of the plane, Cl(3,1), on 16-vectors.

A multivector is a numpy array of shape (16,) indexed by the bitmask over the
basis vectors (e1, e2, e+, e-) with signature (+, +, +, -). The null vectors
e_inf = e- + e+ and e_0 = (e- - e+)/2 satisfy e_inf . e_0 = -1, e_inf^2 =
e_0^2 = 0, which gives the usual conformal model:

    point   X = x + |x|^2/2 e_inf + e_0              (IPNS, null)
    circle  s = C - r^2/2 e_inf                      (IPNS)
    line    l = n + d e_inf                          (IPNS, |n| = 1, x.n = d)

Similarities act by versor sandwiches and commute with the outer product, so
constructions written with wedge/dual/meet are equivariant by construction.
Everything here is plain numpy; no dependency on a GA library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

N_BASIS = 4
DIM = 1 << N_BASIS
SIGNATURE = (1.0, 1.0, 1.0, -1.0)


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _blade_product(a: int, b: int) -> tuple[float, int]:
    """Sign and index of the geometric product of two basis blades."""
    sign = 1.0
    shifted = a >> 1
    while shifted:
        if _popcount(shifted & b) % 2:
            sign = -sign
        shifted >>= 1
    for i in range(N_BASIS):
        if (a & b) >> i & 1:
            sign *= SIGNATURE[i]
    return sign, a ^ b


def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cayley = np.zeros((DIM, DIM, DIM))
    outer = np.zeros((DIM, DIM, DIM))
    lcontr = np.zeros((DIM, DIM, DIM))
    for a in range(DIM):
        for b in range(DIM):
            sign, c = _blade_product(a, b)
            cayley[a, b, c] = sign
            if a & b == 0:
                outer[a, b, c] = sign
            if a & ~b == 0:  # a is a sub-blade of b
                lcontr[a, b, c] = sign
    return cayley, outer, lcontr


CAYLEY, OUTER, LCONTR = _tables()
GRADES = np.array([_popcount(i) for i in range(DIM)])
REVERSE_SIGN = np.array([(-1.0) ** (g * (g - 1) // 2) for g in GRADES])


def gp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("i,j,ijk->k", a, b, CAYLEY)


def wedge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("i,j,ijk->k", a, b, OUTER)


def lcontract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("i,j,ijk->k", a, b, LCONTR)


def reverse(a: np.ndarray) -> np.ndarray:
    return a * REVERSE_SIGN


def grade(a: np.ndarray, k: int) -> np.ndarray:
    out = a.copy()
    out[GRADES != k] = 0.0
    return out


def scalar(a: np.ndarray) -> float:
    return float(a[0])


def blade(mask: int, coeff: float = 1.0) -> np.ndarray:
    v = np.zeros(DIM)
    v[mask] = coeff
    return v


ONE = blade(0)
E1 = blade(0b0001)
E2 = blade(0b0010)
EP = blade(0b0100)
EM = blade(0b1000)
E12 = blade(0b0011)
EINF = EM + EP
E0 = 0.5 * (EM - EP)
I4 = blade(0b1111)
I4_INV = -I4  # I^2 = -1 in Cl(3,1)


def dual(a: np.ndarray) -> np.ndarray:
    return gp(a, I4_INV)


# Coefficients of e_0 and e_inf in a vector written on the (e+, e-) basis:
# v = a e1 + b e2 + alpha e_inf + beta e_0  =>  e+ : alpha - beta/2,
# e- : alpha + beta/2.
def e0_coeff(v: np.ndarray) -> float:
    return float(v[8] - v[4])


def einf_coeff(v: np.ndarray) -> float:
    return 0.5 * float(v[4] + v[8])


# ---------------------------------------------------------------------------
# Conformal objects
# ---------------------------------------------------------------------------


def point(x: float, y: float) -> np.ndarray:
    return x * E1 + y * E2 + 0.5 * (x * x + y * y) * EINF + E0


def normalize_point(P: np.ndarray) -> np.ndarray:
    return P / e0_coeff(P)


def to_xy(P: np.ndarray) -> tuple[float, float]:
    b = e0_coeff(P)
    return float(P[1] / b), float(P[2] / b)


def circle(cx: float, cy: float, r: float) -> np.ndarray:
    return point(cx, cy) - 0.5 * r * r * EINF


def normalize_circle(S: np.ndarray) -> np.ndarray:
    return S / e0_coeff(S)


def circle_params(S: np.ndarray) -> tuple[float, float, float]:
    S = normalize_circle(S)
    cx, cy = float(S[1]), float(S[2])
    r2 = cx * cx + cy * cy - 2.0 * einf_coeff(S)
    return cx, cy, math.sqrt(max(r2, 0.0))


def line(nx: float, ny: float, d: float) -> np.ndarray:
    return nx * E1 + ny * E2 + d * EINF


def normalize_line(L: np.ndarray) -> np.ndarray:
    return L / math.hypot(float(L[1]), float(L[2]))


def line_params(L: np.ndarray) -> tuple[float, float, float]:
    L = normalize_line(L)
    return float(L[1]), float(L[2]), einf_coeff(L)


def line_direction(L: np.ndarray) -> tuple[float, float]:
    """Unit direction: the normal rotated by -90 deg (line(P, Q) runs P -> Q)."""
    nx, ny, _ = line_params(L)
    return ny, -nx


def oriented_line(L: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Normalise and choose the sign so that the direction is (tx, ty)."""
    L = normalize_line(L)
    nx, ny = -ty, tx
    if float(L[1]) * nx + float(L[2]) * ny < 0:
        L = -L
    return L


def line_through(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """IPNS line through two points, direction P -> Q."""
    L = dual(wedge(wedge(P, Q), EINF))
    px, py = to_xy(P)
    qx, qy = to_xy(Q)
    return oriented_line(L, qx - px, qy - py)


def circle_through(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """IPNS circle centred at C passing through P."""
    cx, cy = to_xy(C)
    px, py = to_xy(P)
    return circle(cx, cy, math.hypot(px - cx, py - cy))


# ---------------------------------------------------------------------------
# Meet
# ---------------------------------------------------------------------------


def meet_pair(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """OPNS point pair of two IPNS objects and its square T.T.

    T.T > 0: two real points; = 0: tangent (coincident pair); < 0: no real
    intersection. For two lines T is a flat point (contains e_inf) and T.T = 1
    regardless, so callers dispatch on the argument kinds.
    """
    T = dual(wedge(A, B))
    return T, scalar(gp(T, T))


def pair_points(T: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """The two (normalised) points of a real point pair T = lambda P^Q.

    With F = e_inf _| T = lambda (P - Q) and sqrt(T.T) = |lambda| |P.Q|:
    P = (sqrt(T.T) - T) F^-1 and Q = -(T + sqrt(T.T)) F^-1, up to the swap
    that a negative lambda induces (the caller orders the pair anyway).
    """
    F = lcontract(EINF, T)
    Finv = F / scalar(gp(F, F))
    r = math.sqrt(max(s, 0.0))
    P = gp(r * ONE - T, Finv)
    Q = gp(-(T + r * ONE), Finv)
    return normalize_point(P), normalize_point(Q)


def flat_point_xy(T: np.ndarray) -> tuple[float, float] | None:
    """Euclidean position of a flat point T = lambda X^e_inf; None if at infinity."""
    v = lcontract(E0, T)
    b = e0_coeff(v)
    norm = float(np.abs(T).max())
    if norm == 0.0 or abs(b) <= 1e-12 * norm:
        return None
    return float(v[1] / b), float(v[2] / b)


# ---------------------------------------------------------------------------
# Versors
# ---------------------------------------------------------------------------


def translator(tx: float, ty: float) -> np.ndarray:
    return ONE - 0.5 * gp(tx * E1 + ty * E2, EINF)


def rotor(theta: float) -> np.ndarray:
    """Counter-clockwise rotation by theta about the origin."""
    return math.cos(theta / 2) * ONE - math.sin(theta / 2) * E12


def dilator(s: float) -> np.ndarray:
    """Uniform scaling by s > 0 about the origin: exp(ln(s)/2 e_0^e_inf)."""
    lam = math.log(s)
    E = wedge(E0, EINF)
    return math.cosh(lam / 2) * ONE + math.sinh(lam / 2) * E


def reflector_x_axis() -> np.ndarray:
    """Reflection across the x axis: the (odd) versor e2."""
    return E2.copy()


def apply(V: np.ndarray, X: np.ndarray, odd: bool = False) -> np.ndarray:
    Y = gp(gp(V, X), reverse(V))
    return -Y if odd else Y


@dataclass(frozen=True)
class Similarity:
    """x -> s R (F x) + t, with F the optional x-axis reflection applied first."""

    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0
    scale: float = 1.0
    reflect: bool = False

    def versor(self) -> np.ndarray:
        V = gp(gp(translator(self.tx, self.ty), dilator(self.scale)), rotor(self.theta))
        if self.reflect:
            V = gp(V, reflector_x_axis())
        return V

    def transform(self, X: np.ndarray) -> np.ndarray:
        return apply(self.versor(), X, odd=self.reflect)

    def linear(self, vx: float, vy: float) -> tuple[float, float]:
        """The linear part without scale (rotation and optional reflection)."""
        if self.reflect:
            vy = -vy
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * vx - s * vy, s * vx + c * vy

    def xy(self, x: float, y: float) -> tuple[float, float]:
        ux, uy = self.linear(x, y)
        return self.scale * ux + self.tx, self.scale * uy + self.ty

    def inverse_xy(self, x: float, y: float) -> tuple[float, float]:
        ux, uy = (x - self.tx) / self.scale, (y - self.ty) / self.scale
        c, s = math.cos(self.theta), math.sin(self.theta)
        vx, vy = c * ux + s * uy, -s * ux + c * uy
        if self.reflect:
            vy = -vy
        return vx, vy

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "tx": self.tx,
            "ty": self.ty,
            "theta": self.theta,
            "scale": self.scale,
            "reflect": self.reflect,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Similarity:
        return cls(
            tx=float(d["tx"]),
            ty=float(d["ty"]),
            theta=float(d["theta"]),
            scale=float(d["scale"]),
            reflect=bool(d.get("reflect", False)),
        )


IDENTITY = Similarity()
