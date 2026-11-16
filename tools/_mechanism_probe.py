"""Read-only diagnostic: measure, do not assume.

Answers the questions that decide whether the new mechanism is worth
building, using the real archive and nothing else.

A. What are ``CAP20..CAP70``?  Altitude levels or dBZ thresholds?
   Decisive test: if they are levels at 2..7 km, then wherever ``CAP70``
   has signal the echo top must be above 7 km.  The echo top unit is
   itself in question, so both readings are tested.
B. Does the radar -> rain relation depend on the regime?
   A global Z-R fit is compared against regime-conditional fits.  If the
   regimes explain a non-trivial share of the residual, a
   regime-routed operator is justified; if not, it is not.
C. Does the discarded NWP halo carry independent information?
   Three separate measurements:
     C1  the halo ring's signal level versus the centre's;
     C2  does the halo *anticipate* the centre, i.e. does a halo anomaly
         at hour t show up in the centre at hour t+1 more often than the
         centre's own persistence would explain;
     C3  the decisive one: leave-one-case-out regression R^2 for predicting
         the next hour of centre rainfall from
             centre features only / halo features only / both.
         ``delta_r2_halo_over_centre`` is the number that decides whether
         a halo context encoder is worth building.
D. Is the divergence field usable for a Helmholtz inversion?
   Scale, sign convention and the vertical profile over 200/850/925 hPa.

The probe never writes to the archive and never trains anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_ROOT = "/home/wf/radarNWP"
DATA_ROOT = "/home/wf/data/radarNWP/BJ_mmap"

RADAR_VARS = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)

#: NWP channels the halo study needs.
HALO_VARS = (
    "DVG850", "DVG925", "DVG200", "WS925", "PWAT", "TTdMean74",
    "CAPE", "LI500",
)

#: Upsampling factor between the native NWP grid and the radar grid.
NWP_SCALE = 3
NWP_NATIVE = 167
RADAR_N = 301
CROP = (NWP_NATIVE * NWP_SCALE - RADAR_N) // 2      # 100 px per side


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    nz = a[a > 0]
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "frac_positive": float((a > 0).mean()),
        "p50_positive": float(np.percentile(nz, 50)) if nz.size else 0.0,
        "p99_positive": float(np.percentile(nz, 99)) if nz.size else 0.0,
    }


def _ols_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit log y = a + b log x; return (a, b, r2) on the valid subset."""
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(x[m])
    ly = np.log(y[m])
    if lx.std() < 1e-9:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def _ols_train(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least squares with an intercept.  Returns (coef, intercept) or None."""
    if X.shape[0] < X.shape[1] + 5:
        return None
    A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return coef[1:], float(coef[0])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ((y_true - y_pred) ** 2).sum() / ss_tot)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------------------
# A. CAPPI semantics
# ---------------------------------------------------------------------------
def _cap_meaning(radar_cap: np.ndarray, et: np.ndarray, cr: np.ndarray) -> dict:
    """``radar_cap`` is (T, 6, H, W) for CAP20..CAP70."""
    cap = radar_cap
    out: dict = {"levels_km": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}

    # Coverage per level.  If these were dBZ thresholds the area would
    # shrink monotonically with the threshold.
    cover = [float((cap[:, i] > 0).mean()) for i in range(6)]
    out["frac_positive_per_level"] = cover
    out["monotone_decreasing"] = bool(
        all(cover[i] >= cover[i + 1] for i in range(5))
    )

    # Echo-top consistency, for both readings of the ET unit.
    for name, et_km in (("ET_in_km", et.astype(np.float64)),
                        ("ET_in_0.1km", et.astype(np.float64) / 10.0)):
        checks = {}
        for k, level in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
            has = cap[:, k] > 0
            above = et_km > level
            n_above = int(above.sum())
            checks[f"level_{level:.1f}km"] = {
                "n_echo_top_above": n_above,
                "frac_above_with_level_signal": float(has[above].mean()) if n_above else None,
                "frac_above_with_level_empty": float((~has[above]).mean()) if n_above else None,
            }
        out[f"echo_top_consistency_{name}"] = checks

    # Correlation of every level with CR, on shared support.
    corr = {}
    for k, level in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        x = cap[:, k].ravel()
        y = cr.ravel()
        m = (x != 0) & (y != 0)
        corr[f"level_{level:.1f}km"] = (
            float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 1000 else None
        )
    out["level_vs_cr_corr"] = corr

    # Column-wise vertical profile: does the profile have an interior
    # maximum (a melting band) rather than monotone decay?  Subsampled to
    # keep the memory bounded.
    step_t, step_p = 4, 2
    sub = cap[::step_t, :, ::step_p, ::step_p]
    prof = sub.transpose(0, 2, 3, 1).reshape(-1, 6)
    live = prof.max(axis=1) > 0
    p = prof[live]
    n_live = int(live.sum())
    if n_live > 100:
        argmax = p.argmax(axis=1)
        hist = np.bincount(argmax, minlength=6).tolist()
        monotone_dec = float(np.mean(np.all(np.diff(p, axis=1) <= 0.5, axis=1)))
        interior_max = float(np.mean((argmax > 0) & (argmax < 5)))
    else:
        hist, monotone_dec, interior_max = [], float("nan"), float("nan")
    out["vertical_profile"] = {
        "n_live_columns": n_live,
        "argmax_level_histogram": hist,
        "frac_monotone_nonincreasing": monotone_dec,
        "frac_interior_maximum": interior_max,
    }
    return out


# ---------------------------------------------------------------------------
# B. Regime dependent radar -> rain
# ---------------------------------------------------------------------------
def _regime_analysis(
    cr: np.ndarray,
    vil: np.ndarray,
    lab: np.ndarray,
    env_centre: dict[str, np.ndarray],
) -> dict:
    """Is the radar -> rain relation regime dependent?

    The regimes are environmental: the 850 hPa divergence (convergence
    vs divergence), the freezing level (TTdMean74).  No rainfall value
    enters the regime definition, so the split cannot degenerate into
    "route by intensity".
    """
    d_r = env_centre["DVG850"]
    tz_r = env_centre["TTdMean74"]

    regimes = {
        "convergent": d_r < np.percentile(d_r, 33.3),
        "divergent": d_r > np.percentile(d_r, 66.7),
    }
    out: dict = {
        "regime_definitions": {
            "convergent": "DVG850 in the lowest third",
            "divergent": "DVG850 in the highest third",
            "warm": "TTdMean74 above median (freezing level high)",
            "cold": "TTdMean74 below median",
        },
        "fits": {},
    }

    def fit(mask: np.ndarray, tag: str) -> None:
        x = cr[mask]
        y = lab[mask]
        a, b, r2 = _ols_loglog(x, y)
        out["fits"][tag] = {
            "n": int(mask.sum()),
            "log_intercept": a,
            "log_slope_b": b,
            "r2_loglog": r2,
            "z_r_a": float(np.exp(a)) if np.isfinite(a) else None,
        }

    wet = (cr > 0) & (lab > 0)
    fit(wet, "global")
    for name, mask in regimes.items():
        fit(wet & mask, f"DVG850_{name}")
    warm = tz_r > np.median(tz_r)
    fit(wet & warm, "warm_freezing_level_high")
    fit(wet & ~warm, "cold_freezing_level_low")
    fit(wet & regimes["convergent"] & warm, "convergent_warm")
    fit(wet & regimes["divergent"] & warm, "divergent_warm")

    # How much residual variance does the regime explain?  Compare the
    # residual sum of squares of one global fit against the per-regime
    # fits, on the same pixels.
    gp = {}
    for tag in out["fits"]:
        a = out["fits"][tag]["log_intercept"]
        b = out["fits"][tag]["log_slope_b"]
        if np.isfinite(a) and np.isfinite(b):
            gp[tag] = (a, b)

    def sse(mask: np.ndarray, params: dict[str, tuple[float, float]]) -> float:
        total = 0.0
        for tag, m in mask.items():
            if not m.any():
                continue
            a, b = params[tag]
            x = cr[m]
            y = lab[m]
            pred = np.exp(a) * np.power(x, b)
            total += float(((y - pred) ** 2).sum())
        return total

    masks = {
        "DVG850_convergent": wet & regimes["convergent"],
        "DVG850_divergent": wet & regimes["divergent"],
    }
    regime_sse = sse(masks, gp)
    common = masks["DVG850_convergent"] | masks["DVG850_divergent"]
    ga, gb, _ = _ols_loglog(cr[common], lab[common])
    global_on_common = float(
        ((lab[common] - np.exp(ga) * np.power(cr[common], gb)) ** 2).sum()
    )
    out["variance_explained_by_regime"] = {
        "n_common": int(common.sum()),
        "sse_global_on_common": global_on_common,
        "sse_per_regime": regime_sse,
        "fraction_reduced": (
            float(1.0 - regime_sse / global_on_common)
            if global_on_common > 0 else None
        ),
    }

    # VIL as a second radar variable: does it add anything over CR?
    a1, b1, r1 = _ols_loglog(cr[wet], lab[wet])
    out["vil_adds_over_cr"] = {
        "corr_cr_lab": float(np.corrcoef(cr[wet], lab[wet])[0, 1]),
        "corr_vil_lab": float(np.corrcoef(vil[wet], lab[wet])[0, 1]),
        "r2_cr_only": r1,
    }
    return out


# ---------------------------------------------------------------------------
# C. Halo geometry, signal, anticipation and predictive value
# ---------------------------------------------------------------------------
def _upsample_to_radar(field: np.ndarray) -> np.ndarray:
    """(T, 167, 167) -> (T, 301, 301) via x3 nearest repeat plus centre crop."""
    up = np.repeat(np.repeat(field, NWP_SCALE, axis=-2), NWP_SCALE, axis=-1)
    return up[:, CROP:CROP + RADAR_N, CROP:CROP + RADAR_N]


def _halo_masks() -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks on the 501x501 upsampled grid: centre and halo ring."""
    up_n = NWP_NATIVE * NWP_SCALE
    centre = np.zeros((up_n, up_n), dtype=bool)
    centre[CROP:CROP + RADAR_N, CROP:CROP + RADAR_N] = True
    halo = ~centre
    return centre, halo


def _halo_geometry() -> dict:
    up_n = NWP_NATIVE * NWP_SCALE
    centre_px = RADAR_N * RADAR_N
    total = up_n * up_n
    return {
        "native_shape": NWP_NATIVE,
        "native_degrees": NWP_NATIVE * 0.03,
        "upsampled_shape": up_n,
        "radar_shape": RADAR_N,
        "crop_px_per_side": CROP,
        "halo_native_cells_per_side": CROP / NWP_SCALE,
        "halo_degrees_per_side": (CROP / NWP_SCALE) * 0.03,
        "centre_area_fraction": centre_px / total,
        "halo_area_fraction": 1.0 - centre_px / total,
    }


def _masked(up: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a 2-D boolean mask to a (H, W) or (T, H, W) field."""
    if up.ndim == 3:
        return up[:, mask]
    return up[mask]


def _storm_motion(cr: np.ndarray, thresh: float = 20.0) -> dict:
    """Estimate the convective system's motion from the radar echo track.

    The mass-weighted centroid of the reflectivity field is fitted with a
    straight line in time; the slope is the motion in radar pixels per
    6-minute step.  The direction is what matters here: it tells which
    side of the halo the weather arrives from.
    """
    T, H, W = cr.shape
    mask = (cr > thresh).astype(np.float64)
    tot = mask.reshape(T, -1).sum(axis=1)
    rows = np.arange(H, dtype=np.float64)[None, :, None]
    cols = np.arange(W, dtype=np.float64)[None, None, :]
    cy = (mask * rows).reshape(T, -1).sum(axis=1) / np.maximum(tot, 1.0)
    cx = (mask * cols).reshape(T, -1).sum(axis=1) / np.maximum(tot, 1.0)
    valid = tot > 50
    if valid.sum() < 4:
        return {"vx_px_per_step": 0.0, "vy_px_per_step": 0.0,
                "frac_frames_with_echo": float(valid.mean()), "usable": False}
    idx = np.arange(T, dtype=np.float64)[valid]
    A = np.stack([idx, np.ones_like(idx)], axis=1)
    kx, *_ = np.linalg.lstsq(A, cx[valid], rcond=None)
    ky, *_ = np.linalg.lstsq(A, cy[valid], rcond=None)
    return {
        "vx_px_per_step": float(kx[0]),
        "vy_px_per_step": float(ky[0]),
        "speed_px_per_step": float(np.hypot(kx[0], ky[0])),
        "frac_frames_with_echo": float(valid.mean()),
        "usable": True,
    }


def _sector_masks(vx: float, vy: float, halo_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the halo ring into an upstream and a downstream half-plane.

    ``vx`` and ``vy`` are the eastward (column) and southward (row)
    components of the motion, so the upstream direction is ``(-vx, -vy)``.
    """
    n = halo_mask.shape[0]
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float64)
    dx = cc - (n - 1) / 2.0
    dy = rr - (n - 1) / 2.0
    proj = dx * (-vx) + dy * (-vy)
    up = halo_mask & (proj > 0)
    down = halo_mask & (proj <= 0)
    return up, down


def _halo_features(
    field_native: np.ndarray,
    centre_mask: np.ndarray,
    halo_mask: np.ndarray,
    up_mask: np.ndarray,
    down_mask: np.ndarray,
) -> dict[str, float]:
    """Scalar summary of one field over the centre crop and the halo ring."""
    up = np.repeat(np.repeat(field_native, NWP_SCALE, axis=-2), NWP_SCALE, axis=-1)
    c = _masked(up, centre_mask)
    h = _masked(up, halo_mask)
    u = _masked(up, up_mask)
    d = _masked(up, down_mask)
    out = {
        "centre_mean": float(c.mean()),
        "centre_p10": float(np.percentile(c, 10)),
        "centre_min": float(c.min()),
        "halo_mean": float(h.mean()),
        "halo_p10": float(np.percentile(h, 10)),
        "halo_min": float(h.min()),
    }
    if u.size:
        out["up_mean"] = float(u.mean())
        out["up_p10"] = float(np.percentile(u, 10))
        out["up_min"] = float(u.min())
    if d.size:
        out["down_mean"] = float(d.mean())
        out["down_p10"] = float(np.percentile(d, 10))
        out["down_min"] = float(d.min())
    return out


def _halo_signal(
    nwp: np.ndarray,
    nwp_vars: list[str],
    centre_mask: np.ndarray,
    halo_mask: np.ndarray,
) -> dict:
    per_var = {}
    for name in HALO_VARS:
        if name not in nwp_vars:
            continue
        f = nwp[:, nwp_vars.index(name)]
        up = np.repeat(np.repeat(f, NWP_SCALE, axis=-2), NWP_SCALE, axis=-1)
        c = up[:, centre_mask]
        h = up[:, halo_mask]
        per_var[name] = {
            "centre_abs_mean": float(np.abs(c).mean()),
            "halo_abs_mean": float(np.abs(h).mean()),
            "centre_std": float(c.std()),
            "halo_std": float(h.std()),
            "halo_over_centre_abs": float(
                np.abs(h).mean() / max(np.abs(c).mean(), 1e-12)
            ),
        }
    return {"per_variable": per_var}


def _halo_anticipation(
    nwp: np.ndarray,
    nwp_vars: list[str],
    centre_mask: np.ndarray,
    halo_mask: np.ndarray,
) -> dict:
    """Does a halo anomaly at hour t show up inside the centre at t+1?

    For every channel we build the two series ``halo_mean(t)`` and
    ``centre_mean(t)`` over the hourly NWP frames of one case and compare

        corr(halo(t),     centre(t+1))   <- anticipation
        corr(centre(t),   centre(t+1))   <- persistence (the reference)

    If the anticipation term beats the persistence term, the halo contains
    information about the centre's future state that the centre itself
    does not already carry.
    """
    out: dict = {}
    for name in ("DVG850", "DVG925", "DVG200", "PWAT", "TTdMean74"):
        if name not in nwp_vars:
            continue
        f = nwp[:, nwp_vars.index(name)]
        up = np.repeat(np.repeat(f, NWP_SCALE, axis=-2), NWP_SCALE, axis=-1)
        cm = up[:, centre_mask].mean(axis=1)
        hm = up[:, halo_mask].mean(axis=1)
        if cm.size < 6:
            continue
        pers = _safe_corr(cm[:-1], cm[1:])
        antic = _safe_corr(hm[:-1], cm[1:])
        out[name] = {
            "corr_centre_persistence": pers,
            "corr_halo_to_centre_next": antic,
            "anticipation_gain": (
                float(antic - pers) if np.isfinite(antic) and np.isfinite(pers)
                else None
            ),
        }
    return out


def _halo_predictive_table(
    nwp: np.ndarray,
    nwp_vars: list[str],
    label: np.ndarray,
    label_times: np.ndarray,
    nwp_times: np.ndarray,
    centre_mask: np.ndarray,
    halo_mask: np.ndarray,
    up_mask: np.ndarray,
    down_mask: np.ndarray,
) -> list[dict]:
    """One row per (case, NWP hour): features and the next-hour rain target.

    The predictor is the NWP slice valid at the hour boundary ``t``; the
    target is the accumulated ground rainfall over ``(t, t+60min]``.  This
    is stricter than the benchmark's own FORECAST alignment, which also
    shows the model the slice valid at the end of the window.
    """
    rows: list[dict] = []
    lab_map = {int(t): label[i] for i, t in enumerate(label_times)}
    lab_keys = np.array(sorted(lab_map.keys()))

    # MMDDHHMM is not linear in minutes across midnight, so the window end
    # is built with the upstream timestamp helper rather than by addition.
    from radar_nwp_bench.data.index import add_minutes  # noqa: PLC0415

    for j, t in enumerate(nwp_times):
        t = int(t)
        # label frames in (t, t+60]
        sel = [k for k in lab_keys if t < int(k) <= add_minutes(t, 60)]
        if len(sel) < 4:
            continue
        acc = np.zeros((RADAR_N, RADAR_N), dtype=np.float64)
        per_frame = []
        for k in sel:
            frame = np.asarray(lab_map[int(k)], dtype=np.float64)
            acc += frame
            per_frame.append(float(frame.sum()))
        target_total = float(acc.sum())
        target_max = float(max(per_frame))
        target_frac = float((acc > 0.05).mean())

        row: dict = {
            "nwp_index": j,
            "nwp_time": t,
            "n_label_frames": len(sel),
            "target_rain_total": target_total,
            "target_rain_max_6min": target_max,
            "target_rain_area_frac": target_frac,
        }
        for name in HALO_VARS:
            if name not in nwp_vars:
                continue
            f = nwp[j, nwp_vars.index(name)]
            feats = _halo_features(f, centre_mask, halo_mask, up_mask, down_mask)
            for k, v in feats.items():
                row[f"{name}__{k}"] = v
        rows.append(row)
    return rows


def _loco_r2(rows: list[dict], case_ids: list[str], feature_groups: list[str],
             target: str) -> dict:
    """Leave-one-case-out CV R^2 for one feature group.

    Missing values are imputed with the training-fold column mean, so a
    feature that is absent in one case does not void the whole fold.
    """
    y = np.array([r[target] for r in rows], dtype=np.float64)
    groups = np.array(case_ids)
    unique = sorted(set(case_ids))
    X = np.full((len(rows), len(feature_groups)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(feature_groups):
            X[i, j] = r.get(c, np.nan)
    preds = np.full_like(y, np.nan)
    for held in unique:
        tr = groups != held
        te = groups == held
        Xtr, ytr = X[tr], y[tr]
        with np.errstate(invalid="ignore"):
            col_mean = np.nanmean(Xtr, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, col_mean)
        Xte = np.where(np.isfinite(X[te]), X[te], col_mean)
        ok = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
        fitted = _ols_train(Xtr[ok], ytr[ok])
        if fitted is None:
            continue
        coef, b = fitted
        preds[te] = Xte @ coef + b
    m = np.isfinite(preds)
    if m.sum() < 20:
        return {"n": int(m.sum()), "r2": None}
    per_case = {}
    for c in unique:
        sel = groups == c
        sel = sel & m
        if sel.sum() >= 3:
            per_case[c] = _r2(y[sel], preds[sel])
    return {
        "n": int(m.sum()),
        "n_cases": len(unique),
        "n_features": len(feature_groups),
        "r2": _r2(y[m], preds[m]),
        "per_case_r2": per_case,
        "target_mean": float(y[m].mean()),
        "target_std": float(y[m].std()),
    }


# ---------------------------------------------------------------------------
# D. Divergence profile
# ---------------------------------------------------------------------------
def _divergence_profile(nwp: np.ndarray, nwp_vars: list[str]) -> dict:
    out = {}
    for name in ("DVG200", "DVG850", "DVG925"):
        if name not in nwp_vars:
            continue
        f = nwp[:, nwp_vars.index(name)]
        out[name] = {
            **_stats(f),
            "frac_negative": float((f < 0).mean()),
            "frac_positive": float((f > 0).mean()),
        }
    if all(v in nwp_vars for v in ("DVG200", "DVG850", "DVG925")):
        d200 = nwp[:, nwp_vars.index("DVG200")]
        d850 = nwp[:, nwp_vars.index("DVG850")]
        d925 = nwp[:, nwp_vars.index("DVG925")]
        out["profile_differences"] = {
            "d925_minus_d850_mean": float((d925 - d850).mean()),
            "d850_minus_d200_mean": float((d850 - d200).mean()),
            "d925_minus_d200_mean": float((d925 - d200).mean()),
            "interpretation": (
                "divergence decreasing with height over the same column "
                "implies mass convergence in the layer below, i.e. ascent"
            ),
        }
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--bench-root", default=BENCH_ROOT)
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--nwp-channels", type=int, default=0,
                    help="0 = all HALO_VARS present")
    args = ap.parse_args()

    sys.path.insert(0, args.bench_root)
    from radar_nwp_bench.data.pack import load_packed_mmap  # noqa: E402

    root = Path(args.data_root)
    cases = sorted(p for p in root.glob("CP_*") if p.is_dir())
    if not cases:
        raise SystemExit(f"no cases under {root}")

    centre_mask, halo_mask = _halo_masks()
    report: dict = {
        "n_cases": 0,
        "cases": [],
        "halo_geometry": _halo_geometry(),
        "table_rows": [],
        "table_cases": [],
    }
    acc_cap_cover: list[list[float]] = []
    acc_interior: list[float] = []
    acc_halo_ratio: dict[str, list[float]] = {}
    anticipation: dict[str, list[float]] = {}

    for path in cases[: args.n_cases]:
        case = load_packed_mmap(path)
        radar_mm = case.get("radar")
        if radar_mm is None:
            continue
        radar_vars = [str(v) for v in case["radar_vars"]]
        nwp_vars = [str(v) for v in case["nwp_vars"]]
        nwp_times = np.asarray(case["nwp_times"], dtype=np.int64)
        label_times = np.asarray(case["label_times"], dtype=np.int64)

        entry: dict = {
            "case": path.name,
            "n_radar_frames": int(radar_mm.shape[0]),
            "n_label_frames": int(case["label"].shape[0]),
            "n_nwp_frames": int(case["nwp"].shape[0]),
            "nwp_time_span": [int(nwp_times[0]), int(nwp_times[-1])] if nwp_times.size else None,
            "label_time_span": [int(label_times[0]), int(label_times[-1])] if label_times.size else None,
        }

        # -- A. CAPPI semantics -------------------------------------------
        try:
            cap_idx = [radar_vars.index(f"CAP{d}") for d in (20, 30, 40, 50, 60, 70)]
            cap = np.asarray(radar_mm[:, cap_idx])
            et = np.asarray(radar_mm[:, radar_vars.index("ET")])
            cr_full = np.asarray(radar_mm[:, radar_vars.index("CR")])
            entry["cap_meaning"] = _cap_meaning(cap, et, cr_full)
            acc_cap_cover.append(list(entry["cap_meaning"]["frac_positive_per_level"]))
            acc_interior.append(
                entry["cap_meaning"]["vertical_profile"]["frac_interior_maximum"]
            )
            del cap, et, cr_full
        except Exception as exc:  # noqa: BLE001
            entry["cap_meaning_error"] = f"{type(exc).__name__}: {exc}"

        # -- shared arrays -------------------------------------------------
        cr = None
        vil = None
        lab = None
        nwp = None
        try:
            cr = np.asarray(radar_mm[:, radar_vars.index("CR")]).astype(np.float64)
            vil = np.asarray(radar_mm[:, radar_vars.index("VIL")]).astype(np.float64)
            lab = np.asarray(case["label"][:, 0]).astype(np.float64)
            keep = [nwp_vars.index(v) for v in HALO_VARS if v in nwp_vars]
            nwp = np.asarray(case["nwp"][:, keep]).astype(np.float64)
            nwp_keep_vars = [v for v in HALO_VARS if v in nwp_vars]
        except Exception as exc:  # noqa: BLE001
            entry["load_error"] = f"{type(exc).__name__}: {exc}"
            report["cases"].append(entry)
            continue

        # -- B. regime dependent Z-R ---------------------------------------
        try:
            n = min(cr.shape[0], vil.shape[0], lab.shape[0], nwp.shape[0])
            env = {}
            for nm in ("DVG850", "TTdMean74"):
                if nm in nwp_keep_vars:
                    env[nm] = _upsample_to_radar(nwp[:n, nwp_keep_vars.index(nm)])
            entry["regime"] = _regime_analysis(cr[:n], vil[:n], lab[:n], env)
        except Exception as exc:  # noqa: BLE001
            entry["regime_error"] = f"{type(exc).__name__}: {exc}"

        # -- C. halo --------------------------------------------------------
        try:
            motion = _storm_motion(cr)
            entry["storm_motion"] = motion
            up_mask, down_mask = _sector_masks(
                motion["vx_px_per_step"], motion["vy_px_per_step"], halo_mask
            )
            entry["sector_pixels"] = {
                "upstream": int(up_mask.sum()),
                "downstream": int(down_mask.sum()),
            }
        except Exception as exc:  # noqa: BLE001
            entry["storm_motion_error"] = f"{type(exc).__name__}: {exc}"
            up_mask, down_mask = halo_mask, np.zeros_like(halo_mask)
        try:
            sig = _halo_signal(nwp, nwp_keep_vars, centre_mask, halo_mask)
            entry["halo"] = sig
            for k, v in sig["per_variable"].items():
                acc_halo_ratio.setdefault(k, []).append(v["halo_over_centre_abs"])
        except Exception as exc:  # noqa: BLE001
            entry["halo_error"] = f"{type(exc).__name__}: {exc}"
        try:
            ant = _halo_anticipation(nwp, nwp_keep_vars, centre_mask, halo_mask)
            entry["halo_anticipation"] = ant
            for k, v in ant.items():
                g = v.get("anticipation_gain")
                if g is not None and np.isfinite(g):
                    anticipation.setdefault(k, []).append(g)
        except Exception as exc:  # noqa: BLE001
            entry["halo_anticipation_error"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = _halo_predictive_table(
                nwp, nwp_keep_vars, np.asarray(case["label"][:, 0]),
                label_times, nwp_times, centre_mask, halo_mask,
                up_mask, down_mask,
            )
            for r in rows:
                r["case"] = path.name
                report["table_rows"].append(r)
                report["table_cases"].append(path.name)
            entry["n_table_rows"] = len(rows)
        except Exception as exc:  # noqa: BLE001
            entry["halo_table_error"] = f"{type(exc).__name__}: {exc}"

        # -- D. divergence profile -----------------------------------------
        try:
            div = _divergence_profile(nwp, nwp_keep_vars)
            entry["divergence"] = div
        except Exception as exc:  # noqa: BLE001
            entry["divergence_error"] = f"{type(exc).__name__}: {exc}"

        report["cases"].append(entry)
        report["n_cases"] += 1

    # -- aggregates --------------------------------------------------------
    if acc_cap_cover:
        cover = np.mean(np.asarray(acc_cap_cover, dtype=np.float64), axis=0)
        report["aggregate_cap"] = {
            "mean_frac_positive_per_level": cover.tolist(),
            "monotone_decreasing": bool(
                all(cover[i] >= cover[i + 1] for i in range(len(cover) - 1))
            ),
            "mean_frac_interior_maximum": (
                float(np.nanmean(acc_interior)) if acc_interior else None
            ),
        }
    if acc_halo_ratio:
        report["aggregate_halo_over_centre"] = {
            k: float(np.mean(v)) for k, v in acc_halo_ratio.items()
        }
    if anticipation:
        report["aggregate_anticipation_gain"] = {
            k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in anticipation.items()
        }

    fr = [
        c["regime"]["variance_explained_by_regime"]["fraction_reduced"]
        for c in report["cases"]
        if "regime" in c
        and c["regime"]["variance_explained_by_regime"]["fraction_reduced"] is not None
    ]
    if fr:
        report["aggregate_fraction_residual_reduced_by_regime"] = {
            "per_case": fr,
            "mean": float(np.mean(fr)),
            "min": float(np.min(fr)),
            "max": float(np.max(fr)),
        }

    # -- the decisive number ----------------------------------------------
    rows = report["table_rows"]
    if len(rows) >= 40:
        keys = set()
        for r in rows:
            keys.update(r.keys())

        def cols(suffix: str) -> list[str]:
            return sorted(c for c in keys if suffix in c)

        centre_cols = cols("__centre_")
        halo_cols = cols("__halo_")
        up_cols = cols("__up_")
        down_cols = cols("__down_")
        out = {}
        for target in ("target_rain_total", "target_rain_max_6min", "target_rain_area_frac"):
            r_centre = _loco_r2(rows, report["table_cases"], centre_cols, target)
            r_halo = _loco_r2(rows, report["table_cases"], halo_cols, target)
            r_up = _loco_r2(rows, report["table_cases"], up_cols, target)
            r_down = _loco_r2(rows, report["table_cases"], down_cols, target)
            r_both = _loco_r2(rows, report["table_cases"], centre_cols + halo_cols, target)
            r_cu = _loco_r2(rows, report["table_cases"], centre_cols + up_cols, target)
            r_cd = _loco_r2(rows, report["table_cases"], centre_cols + down_cols, target)

            def delta(a: dict, b: dict) -> float | None:
                if a.get("r2") is None or b.get("r2") is None:
                    return None
                return float(a["r2"] - b["r2"])

            out[target] = {
                "centre_only": r_centre,
                "halo_only": r_halo,
                "upstream_only": r_up,
                "downstream_only": r_down,
                "centre_plus_halo": r_both,
                "centre_plus_upstream": r_cu,
                "centre_plus_downstream": r_cd,
                "delta_r2_halo_over_centre": delta(r_both, r_centre),
                "delta_r2_upstream_over_centre": delta(r_cu, r_centre),
                "delta_r2_downstream_over_centre": delta(r_cd, r_centre),
                "delta_r2_upstream_minus_downstream": delta(r_cu, r_cd),
            }
            # Case-level sign test: how many held-out cases actually
            # improve when the upstream sector is added.
            pc_c = r_centre.get("per_case_r2", {})
            pc_u = r_cu.get("per_case_r2", {})
            pc_d = r_cd.get("per_case_r2", {})
            wins = losses = 0
            for c in pc_c:
                if c not in pc_u or c not in pc_d:
                    continue
                if pc_u[c] > pc_d[c]:
                    wins += 1
                elif pc_u[c] < pc_d[c]:
                    losses += 1
            out[target]["upstream_beats_downstream_cases"] = {
                "wins": wins,
                "losses": losses,
                "ties": len(pc_c) - wins - losses,
            }
        report["halo_predictive_value"] = out
        report["halo_feature_groups"] = {
            "centre": centre_cols,
            "halo": halo_cols,
            "upstream": up_cols,
            "downstream": down_cols,
        }
    else:
        report["halo_predictive_value"] = {
            "error": f"only {len(rows)} table rows, need >= 40"
        }

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {args.out}")
    for key in ("aggregate_cap", "aggregate_halo_over_centre",
                "aggregate_anticipation_gain",
                "aggregate_fraction_residual_reduced_by_regime",
                "halo_predictive_value"):
        if key in report:
            print(json.dumps({key: report[key]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
