"""Generate Software-section thumbnails for brhanufen.github.io.

Every thumbnail plots REAL results committed to the corresponding public repo.
Nothing here is synthetic or illustrative. Each tool gets a distinct visual form
so the eight read as a set of anchors rather than eight lookalike panels.

Design constraints (thumbnails display at 180px wide):
  - 4:3, exported at 1200x900 so they stay sharp on retina
  - no ticks, no tick labels, no legends, no axis furniture
  - thick strokes, large markers
  - at most 3 hues, from a CVD-validated categorical palette
"""
import io, json, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "https://raw.githubusercontent.com/brhanufen/{}/main/{}"
OUT = "thumbs"

# CVD-validated categorical slots (all-pairs clean: normal dE 24.0, deutan 9.2)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FAINT = "#1a1a1a", "#8a8a86", "#e6e4df"
SURFACE = "#ffffff"

W, H, DPI = 1200, 900, 200


def fetch(repo, path):
    return urllib.request.urlopen(RAW.format(repo, path), timeout=60).read()


def fig():
    f, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    f.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return f, ax


def save(f, name):
    f.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
    f.savefig(f"{OUT}/{name}.png", dpi=DPI, facecolor=SURFACE)
    plt.close(f)
    print(f"  wrote {OUT}/{name}.png")


# ---------------------------------------------------------------- TrustBinder
def trustbinder():
    """Unit grid: 100 PD-L1 designs by which oracle endorses them.

    AF2 endorsement threshold 0.53 is the paper's control-derived midpoint
    between the native complex (0.83) and scrambled negatives (0.23).
    Reproduces the paper exactly: 28 Boltz-strong, 12 AF2-endorsed, 2 both.
    """
    import csv
    rows = list(csv.DictReader(io.StringIO(
        fetch("trustbinder", "analysis/results_r1/dual_oracle_merged.csv").decode())))
    b = np.array([float(r["boltz_iptm"]) for r in rows])
    a = np.array([float(r["af2_iptm"]) for r in rows])
    both = (b >= 0.8) & (a >= 0.53)
    boltz_only = (b >= 0.8) & ~both
    af2_only = (a >= 0.53) & ~both
    print(f"    n={len(b)}  boltz-only {boltz_only.sum()}  af2-only {af2_only.sum()}  both {both.sum()}")

    f, ax = fig()
    order = np.lexsort((~af2_only, ~boltz_only, ~both))
    cols = np.where(both, ORANGE, np.where(boltz_only, BLUE,
                    np.where(af2_only, AQUA, FAINT)))[order]
    for i, c in enumerate(cols):
        r, cc = divmod(i, 10)
        ax.add_patch(plt.Circle((cc, -r), 0.37, color=c, lw=0))
    ax.set_xlim(-0.7, 9.7); ax.set_ylim(-9.7, 0.7); ax.set_aspect("equal")
    save(f, "sw_trustbinder")


# --------------------------------------------------------------- ProtDiff-ESM
def protdiff():
    """Guidance curve: ProteinGym mean rho rising with classifier-free guidance."""
    d = json.loads(fetch("protdiff-esm", "results/summary.json"))
    sv = d["scoring_variants"]
    pts = sorted((float(k.rsplit("_", 1)[1]), v["mean_rho"])
                 for k, v in sv.items() if k.startswith("option_B_gamma_"))
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    print(f"    gamma {x} -> rho {[round(v,3) for v in y]}")

    f, ax = fig()
    ax.plot(x, y, color=BLUE, lw=7, solid_capstyle="round", zorder=2)
    ax.scatter(x, y, s=340, color=BLUE, zorder=3, edgecolor=SURFACE, lw=4)
    ax.scatter([x[-1]], [y[-1]], s=520, color=ORANGE, zorder=4, edgecolor=SURFACE, lw=4)
    pad = (max(y) - min(y)) * 0.35
    ax.set_ylim(min(y) - pad, max(y) + pad)
    ax.set_xlim(min(x) - 0.35, max(x) + 0.35)
    save(f, "sw_protdiff")


# --------------------------------------------------------------- EnhancerDiff
def enhancerdiff():
    """Strip plot: per-design GC content for each of the three cell types."""
    d = json.loads(fetch("enhancerdiff", "webdemo/designs.json"))
    keys = list(d)[:3]
    groups = {k: [float(x["gc"]) for x in d[k]["designs"]] for k in keys}
    print(f"    {[(k, len(groups[k]), round(float(np.median(groups[k])), 3)) for k in keys]}")

    f, ax = fig()
    rng = np.random.default_rng(0)
    for i, (k, c) in enumerate(zip(keys, [BLUE, ORANGE, AQUA])):
        v = np.array(groups[k])
        ax.scatter(v, np.full(len(v), -i) + rng.uniform(-0.17, 0.17, len(v)),
                   s=95, color=c, alpha=0.55, lw=0)
        ax.plot([np.median(v)] * 2, [-i - 0.20, -i + 0.20], color=c, lw=5,
                solid_capstyle="round", alpha=0.95, zorder=5)
    ax.set_ylim(-len(keys) + 0.45, 0.55)
    save(f, "sw_enhancerdiff")


# ------------------------------------------------------------------ OracleGap
def oraclegap():
    """Scatter: every sequence scored by two independent oracles."""
    d = json.loads(fetch("oraclegap", "results/scores.json"))
    recs = d["records"]
    mal = np.array([float(r["malinois_k562"]) for r in recs])
    enf = np.array([float(r["enformer_k562"]) for r in recs])
    grp = np.array([r.get("group") for r in recs])
    keep = (grp == "design") | (grp == "natural_active")
    mal, enf, grp = mal[keep], enf[keep], grp[keep]
    isdes = grp == "design"
    print(f"    n={len(mal)} ({isdes.sum()} designs, {(~isdes).sum()} natural-active)"
          f"  pearson r={np.corrcoef(mal, enf)[0,1]:.3f}")

    f, ax = fig()
    ax.scatter(mal[~isdes], enf[~isdes], s=110, color=AQUA, alpha=0.5, lw=0, zorder=1)
    ax.scatter(mal[isdes], enf[isdes], s=130, color=BLUE, alpha=0.65, lw=0, zorder=2)
    m = (mal.max() - mal.min()) * 0.05
    ax.set_xlim(mal.min() - m, mal.max() + m)
    m2 = (enf.max() - enf.min()) * 0.05
    ax.set_ylim(enf.min() - m2, enf.max() + m2)
    save(f, "sw_oraclegap")


# ------------------------------------------------------------ SpliceConsensus
def spliceconsensus():
    """Lollipop: five predictors ranked by average precision."""
    d = json.loads(fetch("spliceconsensus", "results/benchmark_summary.json"))
    pm = sorted(d["per_method"], key=lambda r: r["AP"])
    names = [r["method"] for r in pm]; ap = [r["AP"] for r in pm]
    print(f"    {list(zip(names, [round(a,3) for a in ap]))}")

    f, ax = fig()
    for i, a in enumerate(ap):
        top = i == len(ap) - 1
        ax.plot([0, a], [i, i], color=ORANGE if top else FAINT, lw=9,
                solid_capstyle="round", zorder=2)
        ax.scatter([a], [i], s=560 if top else 380,
                   color=ORANGE if top else BLUE, zorder=3,
                   edgecolor=SURFACE, lw=4)
    ax.set_xlim(0, max(ap) * 1.18); ax.set_ylim(-0.7, len(ap) - 0.3)
    save(f, "sw_spliceconsensus")


# --------------------------------------------------------------------- AbStab
def abstab():
    """Grouped bars: accuracy under a random split vs a leakage-free split."""
    d = json.loads(fetch("abstab", "results/eval_jain.json"))
    sp = d["splits"]
    models = [k for k in sp if isinstance(sp[k], dict) and "random" in sp[k]]
    rnd = [sp[m]["random"]["mean"] for m in models]
    clu = [sp[m]["cluster_holdout"]["mean"] for m in models]
    print(f"    models {models}  random {[round(v,3) for v in rnd]}  grouped {[round(v,3) for v in clu]}")

    f, ax = fig()
    x = np.arange(len(models)); w = 0.34
    ax.bar(x - w / 2 - 0.012, rnd, w, color=BLUE, lw=0, zorder=2)
    ax.bar(x + w / 2 + 0.012, clu, w, color=ORANGE, lw=0, zorder=2)
    ax.axhline(0, color=INK, lw=3, zorder=3)
    hi = max(max(rnd), max(clu)); lo = min(min(rnd), min(clu))
    ax.set_ylim(lo - abs(lo) * 0.5 - 0.02, hi + abs(hi) * 0.35 + 0.02)
    ax.set_xlim(-0.6, len(models) - 0.4)
    save(f, "sw_abstab")


# ----------------------------------------------------------------- PerturbVAE
def perturbvae():
    """Slope chart: model vs mean-of-training baseline across three split designs."""
    d = json.loads(fetch("perturbvae", "results/cvae_norman_delta.json"))
    splits = [k for k in d if isinstance(d[k], dict) and "cvae" in d[k]]
    model = [d[k]["cvae"]["mean_pearson"] for k in splits]
    base = [d[k]["mean_of_training"]["mean_pearson"] for k in splits]
    print(f"    splits {splits}  cvae {[round(v,3) for v in model]}  baseline {[round(v,3) for v in base]}")

    f, ax = fig()
    for i, (m, b) in enumerate(zip(model, base)):
        ax.plot([0, 1], [b, m], color=FAINT, lw=7, solid_capstyle="round", zorder=1)
        ax.scatter([0], [b], s=420, color=BLUE, zorder=3, edgecolor=SURFACE, lw=4)
        ax.scatter([1], [m], s=420, color=ORANGE, zorder=3, edgecolor=SURFACE, lw=4)
    ax.set_xlim(-0.28, 1.28)
    allv = model + base; pad = (max(allv) - min(allv)) * 0.22
    ax.set_ylim(min(allv) - pad, max(allv) + pad)
    save(f, "sw_perturbvae")


# ---------------------------------------------------------------- NativeReady
def nativeready():
    """Histogram: out-of-fold predicted scores, split by true class."""
    raw = fetch("nativeready", "model/v3_robust_oof_predictions.npz")
    z = np.load(io.BytesIO(raw), allow_pickle=True)
    print(f"    npz keys: {list(z.keys())}")
    pk = next(k for k in z if any(s in k.lower() for s in ("prob", "pred", "score")))
    yk = next((k for k in z if any(s in k.lower() for s in ("y", "true", "label"))), None)
    p = np.asarray(z[pk], dtype=float).ravel()
    y = np.asarray(z[yk]).ravel() if yk else None
    print(f"    using '{pk}' (n={len(p)})" + (f" split by '{yk}'" if yk else ""))

    f, ax = fig()
    # Smooth density per class. A raw histogram is spiky at this n and unreadable
    # once shrunk to 180px; a KDE keeps the shape and loses the noise.
    def kde(v, grid, bw):
        d = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(1)
        return d / d.max()
    grid = np.linspace(np.nanmin(p), np.nanmax(p), 240)
    for cls, col in zip(np.unique(y), (BLUE, ORANGE)):
        v = p[y == cls]
        bw = 1.06 * v.std() * len(v) ** (-1 / 5) * 1.6      # Silverman, widened
        d = kde(v, grid, bw)
        ax.fill_between(grid, 0, d, color=col, alpha=0.45, lw=0)
        ax.plot(grid, d, color=col, lw=6, solid_capstyle="round")
    ax.set_ylim(0, 1.14)
    ax.set_xlim(grid[0], grid[-1])
    save(f, "sw_nativeready")


if __name__ == "__main__":
    import os, traceback
    os.makedirs(OUT, exist_ok=True)
    for fn in (trustbinder, protdiff, enhancerdiff, oraclegap,
               spliceconsensus, abstab, perturbvae, nativeready):
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception:
            print("  FAILED:"); traceback.print_exc(limit=2)
