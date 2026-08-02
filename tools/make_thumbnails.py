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
    """ESMFold pLDDT for random, generated and natural sequences.

    The repo's headline is an honest negative result: generated sequences sit
    beside the random baseline and far below natural. An earlier version of this
    thumbnail plotted the guidance sweep, which read as a success curve and
    inverted the finding.
    """
    d = json.loads(fetch("protdiff-esm", "results/foldability.json"))
    groups = [("random", d["random"], MUTED),
              ("generated", d["generated"], ORANGE),
              ("natural", d["natural"], BLUE)]
    print("    " + "  ".join(f"{k} {np.mean(v):.1f}" for k, v, _ in groups))

    f, ax = fig()
    rng = np.random.default_rng(0)
    for i, (k, v, c) in enumerate(groups):
        v = np.asarray(v, dtype=float)
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.19, 0.19, len(v)), v,
                   s=85, color=c, alpha=0.5, lw=0)
        ax.plot([i - 0.30, i + 0.30], [v.mean()] * 2, color=c, lw=6,
                solid_capstyle="round", zorder=5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    save(f, "sw_protdiff")


# --------------------------------------------------------------- EnhancerDiff
def enhancerdiff():
    """Independence gap by cell type, against a zero reference.

    The gap is target percentile minus independent percentile: the report's
    oracle-hacking signal, where lower is better and zero means the design
    transfers. K562 (+0.021) and HepG2 (-0.004) sit within noise of zero;
    SK-N-SH (+0.217) is where the model games its target oracle.

    Two earlier versions of this thumbnail were wrong: GC content said nothing
    about transfer, and raw target-vs-independent scores compared two oracles on
    incompatible scales, which made transfer look far worse than it is.
    """
    gaps = []
    for ct in (0, 1, 2):
        e = json.loads(fetch("enhancerdiff", f"results/eval_ct{ct}.json"))
        gaps.append(e["independence"]["mean_gap"])
    print(f"    independence gaps: {[round(x, 4) for x in gaps]}")

    f, ax = fig()
    lim = max(abs(min(gaps)), max(gaps)) * 1.45
    ax.axvline(0, color=MUTED, lw=4, zorder=1)
    for i, gp in enumerate(gaps):
        hot = gp > 0.1                       # clearly away from zero
        ax.plot([0, gp], [-i, -i], color=ORANGE if hot else FAINT, lw=9,
                solid_capstyle="round", zorder=2)
        ax.scatter([gp], [-i], s=520 if hot else 400,
                   color=ORANGE if hot else BLUE, zorder=3,
                   edgecolor=SURFACE, lw=4)
    ax.set_xlim(-lim * 0.35, lim)
    ax.set_ylim(-len(gaps) + 0.45, 0.55)
    save(f, "sw_enhancerdiff")


# ------------------------------------------------------------------ OracleGap
def oraclegap():
    """Scatter: every sequence scored by two independent oracles."""
    d = json.loads(fetch("oraclegap", "results/scores.json"))
    recs = d["records"]
    mal = np.array([float(r["malinois_k562"]) for r in recs])
    enf = np.array([float(r["enformer_k562"]) for r in recs])
    grp = np.array([r.get("group") for r in recs])
    isdes = grp == "design"
    from scipy.stats import rankdata, spearmanr
    print(f"    spearman over all {len(mal)} records = {spearmanr(mal, enf).statistic:.4f}"
          f"  (repo summary: 0.8649)")
    mal, enf = rankdata(mal), rankdata(enf)
    print(f"    n={len(mal)} ({isdes.sum()} designs, {(~isdes).sum()} natural-active)"
          f"  pearson r={np.corrcoef(mal, enf)[0,1]:.3f}")

    f, ax = fig()
    ax.scatter(mal[~isdes], enf[~isdes], s=70, color=AQUA, alpha=0.45, lw=0, zorder=1)
    ax.scatter(mal[isdes], enf[isdes], s=95, color=BLUE, alpha=0.7, lw=0, zorder=2)
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
    """Random split vs leakage-free split, for the two protein language models.

    The site's "0.24 -> 0.08-0.14" is ESM-2 650M and AntiBERTy, not the ridge /
    kNN probes an earlier version of this thumbnail plotted.
    """
    ev = {t: json.loads(fetch("abstab", f"results/eval_emb_{t}.json"))
          for t in ("esm2_650m", "antiberty")}
    rnd = [ev[t]["random"]["mean"] for t in ev]
    clu = [ev[t]["cluster_holdout"]["mean"] for t in ev]
    print(f"    {[(t, round(ev[t]['random']['mean'],3), round(ev[t]['cluster_holdout']['mean'],3)) for t in ev]}")

    f, ax = fig()
    x = np.arange(len(ev)); w = 0.34
    ax.bar(x - w / 2 - 0.012, rnd, w, color=BLUE, lw=0, zorder=2)
    ax.bar(x + w / 2 + 0.012, clu, w, color=ORANGE, lw=0, zorder=2)
    ax.axhline(0, color=INK, lw=3, zorder=3)
    ax.set_ylim(0, max(rnd) * 1.25)
    ax.set_xlim(-0.6, len(ev) - 0.4)
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
    pk = "v4_proba"          # V4 combined = the production model in the repo README
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
