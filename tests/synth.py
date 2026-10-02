"""Synthetic rooms with known dimensions, to check the geometry code itself (not the sensor)."""
import numpy as np


def rotate_yaw(pts: np.ndarray, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    out = pts.copy()
    out[:, 0] = c * pts[:, 0] - s * pts[:, 2]
    out[:, 2] = s * pts[:, 0] + c * pts[:, 2]
    return out


def cast_hits(cam_xz, segments, max_range=5.0, step_deg=0.3):
    """2D ray casting from a camera against wall segments ((x0, z0), (x1, z1)); returns the nearest hit per ray."""
    out = []
    ox, oz = cam_xz
    for a in np.radians(np.arange(0, 360, step_deg)):
        dx, dz = np.cos(a), np.sin(a)
        best = max_range
        for (x0, z0), (x1, z1) in segments:
            ex, ez = x1 - x0, z1 - z0
            den = dx * ez - dz * ex
            if abs(den) < 1e-12:
                continue
            t = ((x0 - ox) * ez - (z0 - oz) * ex) / den
            u = ((x0 - ox) * dz - (z0 - oz) * dx) / den
            if 0.0 < t < best and 0.0 <= u <= 1.0:
                best = t
        if best < max_range:
            out.append((ox + best * dx, oz + best * dz))
    return np.array(out)


def _plane(u0, u1, v0, v1, fixed, axes, step):
    u, v = np.meshgrid(np.arange(u0, u1, step), np.arange(v0, v1, step))
    pts = np.empty((u.size, 3))
    pts[:, axes[0]], pts[:, axes[1]], pts[:, axes[2]] = u.ravel(), v.ravel(), fixed
    return pts


def cast_rays_3d(cam_xyz, segments, height, floor_y=-1.4, openings=(), max_range=5.0, az_step=0.3, el_range=(-55, 35), el_step=0.6,
                 ceiling=True, ceilings=None, seg_heights=None):
    """3D ray casting against vertical walls (segments), a floor and a ceiling; returns the hits (N, 3).

    `openings` is a list of (segment_index, a0, a1, h0, h1): along-segment distance from its start and height above the
    floor; a ray crossing a wall inside an opening continues. Rays that find nothing within `max_range` give no hit.
    `ceiling=False` leaves the room without a ceiling (rays going up find nothing); `ceilings` is a list of (x0, x1, H) giving the
    ceiling height by x; `seg_heights` gives one wall height per segment.
    """
    ox, oy, oz = cam_xyz
    az = np.radians(np.arange(0, 360, az_step))
    el = np.radians(np.arange(el_range[0], el_range[1] + 1e-9, el_step))
    A, E = np.meshgrid(az, el)
    dx, dz, m = (np.cos(A) * 1.0).ravel(), (np.sin(A) * 1.0).ravel(), np.tan(E).ravel()  # horizontal unit dir, slope dy/dt
    rng_t = max_range * np.cos(E).ravel()
    best = np.full(dx.shape, np.inf)
    y0 = oy - floor_y
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(m < 0, -y0 / m, np.inf)
        tc = np.full(dx.shape, np.inf)
        if ceiling:
            if ceilings is None:
                tc = np.where(m > 0, (height - y0) / m, np.inf)
            else:
                for xa, xb, H in ceilings:
                    tch = np.where(m > 0, (H - y0) / m, np.inf)
                    xh = ox + dx * tch
                    tc = np.where((xh >= xa) & (xh < xb) & (tch < tc), tch, tc)
    best = np.minimum(best, np.minimum(tf, tc))
    for k, ((x0, z0), (x1, z1)) in enumerate(segments):
        ex, ez = x1 - x0, z1 - z0
        den = dx * ez - dz * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x0 - ox) * ez - (z0 - oz) * ex) / den
            u = ((x0 - ox) * dz - (z0 - oz) * dx) / den
        y = y0 + m * t
        hk = height if seg_heights is None else seg_heights[k]
        ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1) & (y >= 0) & (y <= hk)
        L = float(np.hypot(ex, ez))
        for seg, a0, a1, h0, h1 in openings:
            if seg == k:
                ok &= ~((u * L >= a0) & (u * L <= a1) & (y >= h0) & (y <= h1))
        best = np.where(ok & (t < best), t, best)
    valid = best <= rng_t
    t = best[valid]
    return np.c_[ox + dx[valid] * t, floor_y + y0 + m[valid] * t, oz + dz[valid] * t]


def polygon_room(verts, height, floor_y=-1.4, step=0.02, noise=0.005, ceiling=True, seed=0, openings=()):
    """Rectilinear polygon room. `openings` is a list of (edge_index, a0, a1, h0, h1): along-edge distance from the edge
    start and height above the floor; the matching wall points are removed."""
    import shapely

    rng = np.random.default_rng(seed)
    poly = shapely.Polygon(verts)
    x0, z0, x1, z1 = poly.bounds
    X, Z = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step))
    inside = shapely.contains_xy(poly, X.ravel(), Z.ravel())
    parts = [np.c_[X.ravel()[inside], np.full(inside.sum(), floor_y), Z.ravel()[inside]]]
    if ceiling:
        parts.append(np.c_[X.ravel()[inside], np.full(inside.sum(), floor_y + height), Z.ravel()[inside]])
    n = len(verts)
    for i in range(n):
        (ax, az), (bx, bz) = verts[i], verts[(i + 1) % n]
        length = float(np.hypot(bx - ax, bz - az))
        t, hh = np.meshgrid(np.arange(0, length, step), np.arange(0, height, step))
        t, hh = t.ravel(), hh.ravel()
        for e, a0, a1, h0, h1 in openings:
            if e == i:
                keep = ~((t >= a0) & (t <= a1) & (hh >= h0) & (hh <= h1))
                t, hh = t[keep], hh[keep]
        u = t / length
        parts.append(np.c_[ax + u * (bx - ax), floor_y + hh, az + u * (bz - az)])
    pts = np.concatenate(parts)
    return pts + rng.normal(scale=noise, size=pts.shape)


def box_room(x0, z0, x1, z1, height, floor_y=-1.4, step=0.02, noise=0.005, ceiling=True, seed=0, skip=(), openings=()):
    """Floor, walls and optional ceiling of a rectangular room.

    Walls are named 'x0','x1','z0','z1'; `skip` omits walls. `openings` is a list of
    (wall, a0, a1, h0, h1): remove wall points whose coordinate along the wall is in [a0, a1] and whose height
    above the floor is in [h0, h1] (z is along x-walls, x is along z-walls).
    """
    rng = np.random.default_rng(seed)
    parts = [_plane(x0, x1, z0, z1, floor_y, (0, 2, 1), step)]
    if ceiling:
        parts.append(_plane(x0, x1, z0, z1, floor_y + height, (0, 2, 1), step))
    walls = {
        "x0": _plane(z0, z1, floor_y, floor_y + height, x0, (2, 1, 0), step),
        "x1": _plane(z0, z1, floor_y, floor_y + height, x1, (2, 1, 0), step),
        "z0": _plane(x0, x1, floor_y, floor_y + height, z0, (0, 1, 2), step),
        "z1": _plane(x0, x1, floor_y, floor_y + height, z1, (0, 1, 2), step),
    }
    for name, w in walls.items():
        if name in skip:
            continue
        for wall, a0, a1, h0, h1 in openings:
            if wall == name:
                along = w[:, 2] if name.startswith("x") else w[:, 0]
                hh = w[:, 1] - floor_y
                w = w[~((along >= a0) & (along <= a1) & (hh >= h0) & (hh <= h1))]
        parts.append(w)
    pts = np.concatenate(parts)
    return pts + rng.normal(scale=noise, size=pts.shape)
