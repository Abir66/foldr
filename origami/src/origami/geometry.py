"""Small 2D geometry helpers. Points are numpy arrays of shape (2,), polygons (n, 2).

Affine maps are 3x3 homogeneous matrices. All faces in the simulator are convex
(they come from cutting a square with straight lines), so the convex-only
algorithms here are enough.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-7


def cross(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def allclose(a, b, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """np.allclose(a, b, atol=atol) for small arrays without its per-call overhead (same rule,
    |a - b| <= atol + rtol * |b| elementwise; stops at the first element that differs)."""
    A = np.ravel(a).tolist()
    B = np.ravel(b).tolist()
    if len(A) != len(B):
        return bool(np.allclose(a, b, atol=atol, rtol=rtol))
    for x, y in zip(A, B):
        if not abs(x - y) <= atol + rtol * abs(y):
            return False
    return True


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("zero-length direction")
    return v / n


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine map to one point (2,) or many (n, 2)."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:2, :2].T + T[:2, 2]


def reflection(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Affine map reflecting across the line through point a with direction u."""
    u = unit(np.asarray(u, dtype=float))
    R = 2.0 * np.outer(u, u) - np.eye(2)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = a - R @ a
    return T


def rotation(center: np.ndarray, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = center - R @ center
    return T


def mirror(center: np.ndarray, axis: str = "vertical") -> np.ndarray:
    """Mirror across a vertical (x -> -x) or horizontal (y -> -y) axis through center."""
    u = np.array([0.0, 1.0]) if axis == "vertical" else np.array([1.0, 0.0])
    return reflection(np.asarray(center, dtype=float), u)


def signed_area(poly: np.ndarray) -> float:
    # plain floats: polygons here have 3-10 vertices, numpy call overhead dominated (np.roll)
    pts = poly.tolist() if isinstance(poly, np.ndarray) else [list(p) for p in poly]
    n = len(pts)
    if n < 3:
        return 0.0
    s1 = s2 = 0.0
    px, py = pts[-1]
    for x, y in pts:
        s1 += px * y
        s2 += py * x
        px, py = x, y
    return 0.5 * (s1 - s2)


def area(poly: np.ndarray) -> float:
    return abs(signed_area(poly))


def ccw(poly: np.ndarray) -> np.ndarray:
    return poly if signed_area(poly) >= 0 else poly[::-1].copy()


def side(a: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
    """Positive if p is left of the directed line (a, u), negative if right."""
    return cross(u, np.asarray(p) - a)


def split_convex(poly: np.ndarray, a: np.ndarray, u: np.ndarray):
    """Cut a convex polygon by a line. Returns (left_piece, right_piece); either may be None."""
    u = unit(u)
    a = np.asarray(a, dtype=float)
    d = u[0] * (poly[:, 1] - a[1]) - u[1] * (poly[:, 0] - a[0])  # = side(a, u, p) for every p
    if not (d < -EPS).any():
        return poly, None
    if not (d > EPS).any():
        return None, poly
    left, right = [], []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = d[i], d[(i + 1) % n]
        if dp >= -EPS:
            left.append(p)
        if dp <= EPS:
            right.append(p)
        if (dp > EPS and dq < -EPS) or (dp < -EPS and dq > EPS):
            x = p + (q - p) * (dp / (dp - dq))
            left.append(x)
            right.append(x)
    return np.array(left), np.array(right)


def clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: intersection of two convex polygons (may be empty).
    Plain-float loop (same arithmetic as split_convex; numpy overhead dominated for tiny polygons)."""
    out = ccw(subject).tolist()
    cl = ccw(clipper).tolist()
    n = len(cl)
    sqrt = math.sqrt
    for i in range(n):
        ax, ay = cl[i]
        bx, by = cl[(i + 1) % n]
        ux, uy = bx - ax, by - ay
        L = sqrt(ux * ux + uy * uy)
        if L < EPS:
            raise ValueError("zero-length direction")
        ux, uy = ux / L, uy / L
        d = [ux * (py - ay) - uy * (px - ax) for px, py in out]
        if min(d) >= -EPS:
            continue  # all of it is inside this edge
        if max(d) <= EPS:
            return np.zeros((0, 2))
        left = []
        m = len(out)
        for j in range(m):
            p, dp = out[j], d[j]
            k = (j + 1) % m
            dq = d[k]
            if dp >= -EPS:
                left.append(p)
            if (dp > EPS and dq < -EPS) or (dp < -EPS and dq > EPS):
                q = out[k]
                t = dp / (dp - dq)
                left.append([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t])
        if len(left) < 3:
            return np.zeros((0, 2))
        out = left
    return np.array(out, dtype=float)


def overlap_area(p: np.ndarray, q: np.ndarray) -> float:
    inter = clip_convex(p, q)
    return area(inter) if len(inter) >= 3 else 0.0


def contains(poly: np.ndarray, pt: np.ndarray, tol: float = 1e-6) -> bool:
    """Point in convex polygon, boundary included."""
    P = ccw(poly).tolist()
    x, y = float(pt[0]), float(pt[1])
    n = len(P)
    for i in range(n):
        ax, ay = P[i]
        bx, by = P[(i + 1) % n]
        ux, uy = bx - ax, by - ay
        L = math.sqrt(ux * ux + uy * uy)
        if L < EPS:
            raise ValueError("zero-length direction")
        if (ux / L) * (y - ay) - (uy / L) * (x - ax) < -tol:
            return False
    return True


_SHARED: dict = {}


def shared_edge(p: np.ndarray, q: np.ndarray, tol: float = 1e-6):
    """Longest common boundary segment of two polygons, as (start, end), or None.
    Memoised on the exact polygon bytes (pure function; the menu asks the same pairs many times)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    key = (p.tobytes(), q.tobytes(), p.shape[0], tol)
    r = _SHARED.get(key, False)
    if r is not False:
        return None if r is None else (r[0].copy(), r[1].copy())
    r = _shared_edge(p, q, tol)
    if len(_SHARED) > 200_000:
        _SHARED.clear()
    _SHARED[key] = r
    return None if r is None else (r[0].copy(), r[1].copy())


def _shared_edge(p: np.ndarray, q: np.ndarray, tol: float = 1e-6):
    P, Q = p.tolist(), q.tolist()
    best, best_len = None, tol
    n, m = len(P), len(Q)
    for i in range(n):
        ax, ay = P[i]
        bx, by = P[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.sqrt(dx * dx + dy * dy)
        if L < tol:
            continue
        ux, uy = dx / L, dy / L
        for j in range(m):
            cx, cy = Q[j]
            ex, ey = Q[(j + 1) % m]
            if abs(ux * (cy - ay) - uy * (cx - ax)) > tol or abs(ux * (ey - ay) - uy * (ex - ax)) > tol:
                continue
            t0 = (cx - ax) * ux + (cy - ay) * uy
            t1 = (ex - ax) * ux + (ey - ay) * uy
            if t0 > t1:
                t0, t1 = t1, t0
            lo, hi = max(0.0, t0), min(L, t1)
            if hi - lo > best_len:
                best, best_len = (np.array([ax + lo * ux, ay + lo * uy]),
                                  np.array([ax + hi * ux, ay + hi * uy])), hi - lo
    return best


def clip_segment(p: np.ndarray, q: np.ndarray, poly: np.ndarray):
    """Part of segment pq inside a convex polygon (Cyrus-Beck), or None."""
    poly = ccw(poly)
    t0, t1 = 0.0, 1.0
    d = q - p
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        num = cross(e, p - a)  # >= 0 means inside for this edge
        den = cross(e, d)
        if abs(den) < 1e-12:
            if num < -1e-9:
                return None
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1 - 1e-9:
            return None
    return p + t0 * d, p + t1 * d


def line_intersection(a1, a2, b1, b2) -> np.ndarray:
    a1, a2, b1, b2 = (np.asarray(x, dtype=float) for x in (a1, a2, b1, b2))
    r, s = a2 - a1, b2 - b1
    den = cross(r, s)
    if abs(den) < 1e-12:
        raise ValueError("lines are parallel")
    t = cross(b1 - a1, s) / den
    return a1 + t * r
