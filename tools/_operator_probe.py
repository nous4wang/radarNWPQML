"""Decisive experiment: rebuild the radar -> rain relation, then test NWP.

Everything here is measured on the archive; nothing is assumed.

Protocol
--------
The upstream seam is fixed: the model sees 11 history frames and produces 10
future frames of 6-minute rainfall.  The upstream single-frame QPE arm
(``physical_qpe``) predicts the rain field at the issue time ``t0`` and copies
it across the ten future frames, scoring CSI@2.0 = 0.1226 against a
persistence floor of 0.0924 and an oracle ceiling of 0.1653.  This probe uses
exactly that protocol, so its numbers are comparable with the upstream ones:

    * one QPE prediction per pixel, regressed from the radar fields at ``t0``
      against the label at ``t0``;
    * the prediction is copied over the ten future frames;
    * CSI is accumulated on the real future labels.

Two questions are answered.

Q1  Does the ten-element radar composite, and in particular the six-level
    CAPPI vertical structure, carry information about the rain that the
    composite reflectivity alone does not?  The global Z-R relation explains
    only R^2 = 0.061 of the log-log variance, so there is room; the question
    is where the room is.

Q2  Does NWP add anything on top?  Two ways of adding it are tested: the
    centre crop (what the benchmark uses today) and the outer ring that the
    crop discards, split into an upstream and a downstream half by the
    radar-derived storm motion.  A noise control with the same number of
    columns calibrates how much of any change is overfitting.

The objective is not MSE.  Upstream measured that MSE-optimal rescaling of
the Z-R field drives CSI@2.0 to 0.0009 (MASTER section 4.17), because MSE
and CSI are different objectives.  Rainfall is zero-inflated and heavy
tailed, so the operator here is a two-part model: a classifier for "is it
wet" and a Poisson regressor for the amount, combined as
``p_wet * E[y | wet]``.

This probe never writes to the archive.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BENCH_ROOT = "/home/wf/radarNWP"
DATA_ROOT = "/home/wf/data/radarNWP/BJ_mmap"

RADAR_VARS = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)

NWP_SCALE = 3
NWP_NATIVE = 167
RADAR_N = 301
CROP = (NWP_NATIVE * NWP_SCALE - RADAR_N) // 2       # 100 px per side

#: Consecutive thresholds, mm per 6-minute accumulation.
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

#: The thresholds the operator is scored at.  2.0 mm is the load-bearing
#: one upstream: the persistence floor is 0.0924 and the oracle ceiling is
#: 0.1653 at that threshold.
SCORE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

#: The columns of the design matrix, in order.
COLUMNS = [
    # -- raw radar elements (10)
    "CR", "VIL", "ET", "HBR",
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    # -- CAPPI vertical structure (7)
    "cap_sum", "cap_max", "cap_top_idx", "cap_argmax_idx",
    "cap_bright", "cap_slope", "cap_nz_levels",
    # -- marginal transforms (3)
    "log1p_cr", "log1p_vil", "et_minus_captop",
    # -- NWP frame-level context (10)
    "nwp_dvg850_centre_mean", "nwp_dvg850_up_mean", "nwp_dvg850_down_mean",
    "nwp_cape_centre_mean", "nwp_cape_up_mean",
    "nwp_pwat_centre_mean", "nwp_pwat_up_mean",
    "nwp_ttd_centre_mean", "nwp_ttd_up_mean",
    "nwp_ws925_centre_mean",
]

N_COLS = len(COLUMNS)
N_NOISE = 8
#: The noise control columns sit right after the physical ones.
NOISE_COLUMNS: list[str] = [f"noise_{i}" for i in range(N_NOISE)]

RADAR_BLOCK = COLUMNS[:20]                     # every element + the CAPPI structure
ARMS: dict[str, list[str]] = {
    # A: composite reflectivity only -- the learned Z-R.
    "A_cr_only": ["CR"],
    # B: reflectivity plus vertically integrated liquid.
    "B_cr_vil": ["CR", "VIL"],
    # C: every element except the CAPPI levels.
    "C_no_cappi": ["CR", "VIL", "ET", "HBR"],
    # D: all ten elements plus the vertical structure.
    "D_all_plus_cappi": RADAR_BLOCK,
    # E: D plus the NWP centre crop (what the benchmark uses today).
    "E_d_plus_nwp_centre": (
        RADAR_BLOCK
        + ["nwp_dvg850_centre_mean", "nwp_cape_centre_mean", "nwp_pwat_centre_mean",
           "nwp_ttd_centre_mean", "nwp_ws925_centre_mean"]
    ),
    # F: D plus the *upstream* half of the discarded outer ring.
    "F_d_plus_nwp_upstream": (
        RADAR_BLOCK
        + ["nwp_dvg850_up_mean", "nwp_cape_up_mean", "nwp_pwat_up_mean", "nwp_ttd_up_mean"]
    ),
    # G: D plus the *downstream* half -- the mirror control for F.
    "G_d_plus_nwp_downstream": (
        RADAR_BLOCK
        + ["nwp_dvg850_down_mean", "nwp_cape_up_mean", "nwp_pwat_up_mean", "nwp_ttd_up_mean"]
    ),
    # H: D plus pure noise -- the overfitting control for E and F.
    "H_d_plus_noise": RADAR_BLOCK + [f"noise_{i}" for i in range(N_NOISE)],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _storm_motion(cr: np.ndarray, thresh: float = 20.0) -> tuple[float, float]:
    """Echo-centroid motion in pixels per 6-minute step (east, south)."""
    T, H, W = cr.shape
    mask = (cr > thresh).astype(np.float64)
    tot = mask.reshape(T, -1).sum(axis=1)
    rows = np.arange(H, dtype=np.float64)[None, :, None]
    cols = np.arange(W, dtype=np.float64)[None, None, :]
    cy = (mask * rows).reshape(T, -1).sum(axis=1) / np.maximum(tot, 1.0)
    cx = (mask * cols).reshape(T, -1).sum(axis=1) / np.maximum(tot, 1.0)
    valid = tot > 50
    if valid.sum() < 4:
        return 0.0, 0.0
    idx = np.arange(T, dtype=np.float64)[valid]
    A = np.stack([idx, np.ones_like(idx)], axis=1)
    kx, *_ = np.linalg.lstsq(A, cx[valid], rcond=None)
    ky, *_ = np.linalg.lstsq(A, cy[valid], rcond=None)
    return float(kx[0]), float(ky[0])


def _sector_masks(vx: float, vy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre crop, upstream half of the outer ring, downstream half."""
    n = NWP_NATIVE * NWP_SCALE
    centre = np.zeros((n, n), dtype=bool)
    centre[CROP:CROP + RADAR_N, CROP:CROP + RADAR_N] = True
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float64)
    dx = cc - (n - 1) / 2.0
    dy = rr - (n - 1) / 2.0
    proj = dx * (-vx) + dy * (-vy)
    halo = ~centre
    up = halo & (proj > 0)
    down = halo & (proj <= 0)
    return centre, up, down


def _cap_features(cap: np.ndarray) -> dict[str, np.ndarray]:
    """Vertical structure of the six CAPPI levels for one frame.

    ``cap`` is (6, H, W) with the levels at 2..7 km.
    """
    out: dict[str, np.ndarray] = {}
    levels = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    nz = cap > 0
    out["cap_sum"] = np.where(nz, cap, 0.0).sum(axis=0)
    out["cap_max"] = cap.max(axis=0)
    out["cap_nz_levels"] = nz.sum(axis=0).astype(np.float32)
    idx = np.arange(6, dtype=np.float32)[:, None, None]
    out["cap_top_idx"] = np.where(nz, idx, -1.0).max(axis=0)
    out["cap_argmax_idx"] = cap.argmax(axis=0).astype(np.float32)
    safe = np.where(nz, cap, 0.0)
    centred = levels - levels.mean()
    denom = float((centred ** 2).sum())
    out["cap_slope"] = (safe * centred[:, None, None]).sum(axis=0) / denom
    out["cap_bright"] = safe.max(axis=0) - safe[0]
    return out


# ---------------------------------------------------------------------------
# per-case feature construction
# ---------------------------------------------------------------------------
def _build_case(path: Path, n_px: int, issue_stride: int, seed: int) -> dict | None:
    """Build the design matrix, the QPE target and the future labels."""
    from radar_nwp_bench.data.pack import load_packed_mmap
    from radar_nwp_bench.data.index import add_minutes, floor_hour

    case = load_packed_mmap(path)
    radar_mm = case.get("radar")
    if radar_mm is None:
        return None
    radar_vars = [str(v) for v in case["radar_vars"]]
    nwp_vars = [str(v) for v in case["nwp_vars"]]
    radar_times = [int(t) for t in case["radar_times"]]
    label_times = [int(t) for t in case["label_times"]]
    nwp_times = [int(t) for t in case["nwp_times"]]

    n_radar = int(radar_mm.shape[0])
    if n_radar < 21:
        return None

    label_pos = {t: i for i, t in enumerate(label_times)}

    # -- storm motion and the halo sectors ---------------------------------
    cr_all = np.asarray(radar_mm[:, radar_vars.index("CR")])
    vx, vy = _storm_motion(cr_all)
    centre_mask, up_mask, down_mask = _sector_masks(vx, vy)

    # -- NWP frame-level context -------------------------------------------
    nwp_context: dict[int, dict[str, float]] = {}
    nwp_mm = case.get("nwp")
    if nwp_mm is not None:
        want = (("DVG850", "dvg850"), ("CAPE", "cape"), ("PWAT", "pwat"),
                ("TTdMean74", "ttd"), ("WS925", "ws925"))
        keep = [(tag, nwp_vars.index(name)) for name, tag in want if name in nwp_vars]
        for j, t in enumerate(nwp_times):
            row: dict[str, float] = {}
            for tag, idx in keep:
                f = np.asarray(nwp_mm[j, idx]).astype(np.float64)
                up = np.repeat(np.repeat(f, NWP_SCALE, axis=-2), NWP_SCALE, axis=-1)
                row[f"nwp_{tag}_centre_mean"] = float(up[centre_mask].mean())
                row[f"nwp_{tag}_up_mean"] = (
                    float(up[up_mask].mean()) if up_mask.any() else float("nan")
                )
                row[f"nwp_{tag}_down_mean"] = (
                    float(up[down_mask].mean()) if down_mask.any() else float("nan")
                )
            nwp_context[t] = row

    # -- radar indices ------------------------------------------------------
    cap_idx = [radar_vars.index(f"CAP{d}") for d in (20, 30, 40, 50, 60, 70)]
    cr_idx = radar_vars.index("CR")
    vil_idx = radar_vars.index("VIL")
    et_idx = radar_vars.index("ET")
    hbr_idx = radar_vars.index("HBR")

    # the pixel-level band the halo could physically reach
    up_dx, up_dy = -vx, -vy
    rr, cc = np.mgrid[0:RADAR_N, 0:RADAR_N].astype(np.float32)
    band = ((cc - (RADAR_N - 1) / 2.0) * up_dx + (rr - (RADAR_N - 1) / 2.0) * up_dy) > 0

    rng = np.random.default_rng(seed)
    Xs, Yq, Yf, Gs, Bs = [], [], [], [], []
    group_meta: list[dict] = []
    gid = 0

    for r in range(10, n_radar - 10, issue_stride):
        issue_time = radar_times[r]
        if issue_time not in label_pos:
            continue
        future = [add_minutes(issue_time, k * 6) for k in range(1, 11)]
        if not all(t in label_pos for t in future):
            continue
        fut_idx = [label_pos[t] for t in future]

        t0 = np.asarray(radar_mm[r]).astype(np.float32)
        cr, vil, et, hbr = t0[cr_idx], t0[vil_idx], t0[et_idx], t0[hbr_idx]
        cap = t0[cap_idx]
        cf = _cap_features(cap)

        px = rng.choice(RADAR_N * RADAR_N, size=n_px, replace=False)
        py, pxx = np.divmod(px, RADAR_N)

        cols = {
            "CR": cr[py, pxx], "VIL": vil[py, pxx], "ET": et[py, pxx], "HBR": hbr[py, pxx],
            "CAP20": cap[0][py, pxx], "CAP30": cap[1][py, pxx], "CAP40": cap[2][py, pxx],
            "CAP50": cap[3][py, pxx], "CAP60": cap[4][py, pxx], "CAP70": cap[5][py, pxx],
            "cap_sum": cf["cap_sum"][py, pxx], "cap_max": cf["cap_max"][py, pxx],
            "cap_top_idx": cf["cap_top_idx"][py, pxx],
            "cap_argmax_idx": cf["cap_argmax_idx"][py, pxx],
            "cap_bright": cf["cap_bright"][py, pxx],
            "cap_slope": cf["cap_slope"][py, pxx],
            "cap_nz_levels": cf["cap_nz_levels"][py, pxx],
            "log1p_cr": np.log1p(np.maximum(cr[py, pxx], 0.0)),
            "log1p_vil": np.log1p(np.maximum(vil[py, pxx], 0.0)),
            "et_minus_captop": et[py, pxx] - (cf["cap_top_idx"][py, pxx] + 1.0),
        }
        ctx = nwp_context.get(floor_hour(issue_time), {})
        for tag in ("dvg850", "cape", "pwat", "ttd", "ws925"):
            for zone in ("centre", "up", "down"):
                name = f"nwp_{tag}_{zone}_mean"
                cols[name] = np.full(n_px, ctx.get(name, float("nan")), dtype=np.float32)
        for i in range(N_NOISE):
            cols[f"noise_{i}"] = rng.standard_normal(n_px).astype(np.float32)

        Xs.append(np.stack([cols[c] for c in COLUMNS + NOISE_COLUMNS],
                           axis=1).astype(np.float32))
        Yq.append(np.asarray(case["label"][label_pos[issue_time], 0], dtype=np.float32)[py, pxx])
        Yf.append(np.asarray(case["label"][fut_idx, 0], dtype=np.float32)[:, py, pxx].T)
        Gs.append(np.full(n_px, gid, dtype=np.int32))
        Bs.append(band[py, pxx])
        group_meta.append({"case": path.name, "issue": issue_time, "group": gid})
        gid += 1

    if not Xs:
        return None
    return {
        "X": np.concatenate(Xs, axis=0),
        "Yq": np.concatenate(Yq, axis=0),
        "Yf": np.concatenate(Yf, axis=0),
        "g": np.concatenate(Gs),
        "band": np.concatenate(Bs),
        "groups": group_meta,
        "motion": {"vx": vx, "vy": vy},
    }


# ---------------------------------------------------------------------------
def _fit_event_classifier(Xtr, ytr, thr, Xte, seed: int = 0, max_iter: int = 100):
    """P(y >= thr) from the radar (and NWP) columns.

    The operator is a *discriminator*, not a mean regressor.  A mean
    regressor predicts E[y | x], which smooths the field; thresholding a
    smoothed field at 2 mm almost never fires, and the measured bias drops
    to a few per cent of the observed rate.  That is the same objective
    mismatch upstream measured for MSE (MASTER section 4.17), so the probe
    models the event directly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=False, max_bins=128, random_state=seed,
    )
    y = (ytr >= thr).astype(np.int32)
    clf.fit(Xtr, y)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    return clf, p_tr, p_te


def _csi_at(p: np.ndarray, Y: np.ndarray, thr: float, cut: float) -> dict:
    """Contingency counts for the decision rule ``p >= cut``."""
    t = Y >= thr
    pos = p[:, None] >= cut
    tp = int(np.count_nonzero(pos & t))
    fp = int(np.count_nonzero(pos & ~t))
    fn = int(np.count_nonzero(~pos & t))
    cn = int(np.count_nonzero(~pos & ~t))
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    bias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "cn": cn,
            "csi": csi, "pod": pod, "far": far, "bias": bias, "cut": float(cut)}


def _best_cut(p: np.ndarray, Y: np.ndarray, thr: float) -> float:
    """The decision cut that maximises CSI on the *training* rows.

    The cut is chosen on the training folds only; the held-out fold then
    inherits it, so no test information leaks into the score.
    """
    best_cut, best_csi = 0.5, -1.0
    for cut in np.unique(np.round(np.quantile(p, np.linspace(0.01, 0.99, 49)), 5)):
        s = _csi_at(p, Y, thr, float(cut))
        if np.isfinite(s["csi"]) and s["csi"] > best_csi:
            best_csi, best_cut = s["csi"], float(cut)
    return best_cut


def _score_persistence(Yq: np.ndarray, Yf: np.ndarray, thr: float) -> dict:
    """The calibration anchor: copy the true t0 rain forward.

    This is the upstream persistence floor, so the probe's numbers are on
    the same scale as the published ones.
    """
    t = Yf >= thr
    pos = (Yq >= thr)[:, None]
    tp = int(np.count_nonzero(pos & t))
    fp = int(np.count_nonzero(pos & ~t))
    fn = int(np.count_nonzero(~pos & t))
    cn = int(np.count_nonzero(~pos & ~t))
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    bias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "cn": cn,
            "csi": csi, "pod": pod, "far": far, "bias": bias, "cut": None}


def _score_zr(cr: np.ndarray, Yf: np.ndarray, thr: float, a: float, b: float) -> dict:
    """The analytical Z-R anchor: the global log-log power law."""
    pred = np.exp(a) * np.power(np.maximum(cr, 1e-6), b)
    t = Yf >= thr
    pos = pred[:, None] >= thr
    tp = int(np.count_nonzero(pos & t))
    fp = int(np.count_nonzero(pos & ~t))
    fn = int(np.count_nonzero(~pos & t))
    cn = int(np.count_nonzero(~pos & ~t))
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    bias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "cn": cn,
            "csi": csi, "pod": pod, "far": far, "bias": bias, "cut": None}


def _fit_zr(cr: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log y = a + b log CR on the wet training pixels."""
    m = (cr > 0) & (y > 0)
    if m.sum() < 100:
        return 0.0, 1.0
    b, a = np.polyfit(np.log(cr[m]), np.log(y[m]), 1)
    return float(a), float(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--bench-root", default=BENCH_ROOT)
    ap.add_argument("--n-cases", type=int, default=62)
    ap.add_argument("--n-px", type=int, default=1000)
    ap.add_argument("--issue-stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, args.bench_root)
    root = Path(args.data_root)
    cases = sorted(p for p in root.glob("CP_*") if p.is_dir())
    if not cases:
        raise SystemExit(f"no cases under {root}")

    report: dict = {"n_cases": 0, "arms": {}, "meta": {}}
    Xs, Yqs, Yfs, Gs, Bs, case_of_group = [], [], [], [], [], []
    group_offset = 0
    t_start = time.time()
    for i, path in enumerate(cases[: args.n_cases]):
        built = _build_case(path, args.n_px, args.issue_stride, seed=1234 + i)
        if built is None:
            continue
        Xs.append(built["X"])
        Yqs.append(built["Yq"])
        Yfs.append(built["Yf"])
        Gs.append(built["g"] + group_offset)
        Bs.append(built["band"])
        case_of_group.extend([m["case"] for m in built["groups"]])
        group_offset += len(built["groups"])
        report["n_cases"] += 1
        print(f"[{i+1}/{min(len(cases), args.n_cases)}] {path.name} "
              f"rows={built['X'].shape[0]} groups={len(built['groups'])} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

    if not Xs:
        raise SystemExit("no usable cases")
    X = np.concatenate(Xs, axis=0)
    Yq = np.concatenate(Yqs, axis=0)
    Yf = np.concatenate(Yfs, axis=0)
    G = np.concatenate(Gs, axis=0)
    B = np.concatenate(Bs, axis=0)
    case_of_group = np.array(case_of_group)
    del Xs, Yqs, Yfs, Gs, Bs

    report["meta"] = {
        "n_rows": int(X.shape[0]),
        "n_groups": int(G.max() + 1),
        "n_cases": report["n_cases"],
        "n_px_per_issue": args.n_px,
        "issue_stride": args.issue_stride,
        "columns": COLUMNS + NOISE_COLUMNS,
        "wet_fraction_t0": float((Yq > 0.05).mean()),
        "wet_fraction_future": float((Yf > 0.05).mean()),
    }
    print(f"[build] rows={X.shape[0]} groups={G.max()+1} "
          f"wet_t0={report['meta']['wet_fraction_t0']:.4f}", flush=True)

    # ---- fold assignment by case -----------------------------------------
    uniq = sorted(set(case_of_group.tolist()))
    fold_of_case = {c: i % args.folds for i, c in enumerate(uniq)}
    fold_of_group = np.array([fold_of_case[c] for c in case_of_group])
    group_fold = fold_of_group[G]

    def empty_counts() -> dict:
        return {f"thr_{t}": {"tp": 0, "fp": 0, "fn": 0, "cn": 0} for t in THRESHOLDS}

    want_arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]

    def blank() -> dict:
        return {f"thr_{t}": {"tp": 0, "fp": 0, "fn": 0, "cn": 0} for t in SCORE_THRESHOLDS}

    def accumulate(dst: dict, src: dict) -> None:
        for k, v in src.items():
            for f in ("tp", "fp", "fn", "cn"):
                dst[k][f] += v[f]

    def finalise(counts: dict) -> dict:
        out: dict = {}
        for thr in SCORE_THRESHOLDS:
            t = counts[f"thr_{thr}"]
            csi = t["tp"] / (t["tp"] + t["fp"] + t["fn"]) if (t["tp"] + t["fp"] + t["fn"]) else float("nan")
            pod = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else float("nan")
            far = t["fp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else float("nan")
            bias = (t["tp"] + t["fp"]) / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else float("nan")
            out[f"thr_{thr}"] = {**t, "csi": csi, "pod": pod, "far": far, "bias": bias}
        return out

    # ---- calibration anchors ---------------------------------------------
    # The persistence anchor copies the true t0 rain forward; it is the
    # upstream floor, so it tells us whether this probe is on the same
    # scale as the published numbers.
    anchors: dict[str, dict] = {}
    for thr in SCORE_THRESHOLDS:
        anchors.setdefault("Z_persistence", {})[f"thr_{thr}"] = _score_persistence(Yq, Yf, thr)
        anchors.setdefault("Z_persistence_band", {})[f"thr_{thr}"] = _score_persistence(
            Yq[B], Yf[B], thr
        )
    report["anchors"] = anchors

    cr_col = COLUMNS.index("CR")
    zr_counts = blank()
    zr_counts_band = blank()
    for f in range(args.folds):
        tr = group_fold != f
        te = group_fold == f
        a, b = _fit_zr(X[tr, cr_col], Yq[tr])
        for thr in SCORE_THRESHOLDS:
            accumulate(zr_counts, {"thr_" + str(thr): _score_zr(X[te, cr_col], Yf[te], thr, a, b)})
            accumulate(zr_counts_band, {"thr_" + str(thr): _score_zr(
                X[te][B[te], cr_col], Yf[te][B[te]], thr, a, b)})
    report["anchors"]["Z_zr_loglog"] = finalise(zr_counts)
    report["anchors"]["Z_zr_loglog_band"] = finalise(zr_counts_band)

    # ---- the arms --------------------------------------------------------
    for arm in want_arms:
        names = ARMS[arm]
        cols = [COLUMNS.index(c) if c in COLUMNS else N_COLS + int(c.split("_")[1])
                for c in names]
        Xa = X[:, cols]
        t_arm = time.time()
        counts = blank()
        counts_band = blank()
        cuts: dict[str, list[float]] = {f"thr_{t}": [] for t in SCORE_THRESHOLDS}
        for f in range(args.folds):
            tr = group_fold != f
            te = group_fold == f
            for thr in SCORE_THRESHOLDS:
                _, p_tr, p_te = _fit_event_classifier(
                    Xa[tr], Yq[tr], thr, Xa[te], max_iter=args.max_iter)
                cut = _best_cut(p_tr, Yf[tr], thr)
                cuts[f"thr_{thr}"].append(cut)
                accumulate(counts, {f"thr_{thr}": _csi_at(p_te, Yf[te], thr, cut)})
                sel = B[te]
                accumulate(counts_band, {f"thr_{thr}": _csi_at(
                    p_te[sel], Yf[te][sel], thr, cut)})
            print(f"  [{arm}] fold {f} done ({time.time()-t_arm:.0f}s)", flush=True)
        report["arms"][arm] = {
            "n_features": len(cols),
            "features": names,
            "full_domain": finalise(counts),
            "upstream_band": finalise(counts_band),
            "decision_cuts": cuts,
            "seconds": time.time() - t_arm,
        }
        csi2 = report["arms"][arm]["full_domain"]["thr_2.0"]["csi"]
        csi2b = report["arms"][arm]["upstream_band"]["thr_2.0"]["csi"]
        print(f"[arm] {arm}: CSI@2.0 full={csi2:.4f} band={csi2b:.4f} "
              f"({time.time()-t_arm:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
