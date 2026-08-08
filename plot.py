"""
Compare any number of RX logs, grouped by their (N, N_LEN) configuration.

Produces four panels:
    1. outcome counts per log (stacked)
    2. clean-detection rate with binomial standard error
    3. detection rate over time
    4. success rate against N_LEN, one line per N, so the two factors can be
       read separately rather than as a single ordering of configurations

Usage:  python compare_logs.py log1.csv log2.csv [log3.csv ...]
"""

import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

CATS = ["ok", "sync_fail", "ambiguous", "no_preamble"]
COLORS = ["#1D9E75", "#EF9F27", "#D85A30", "#888780"]
LINE = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#EF9F27", "#D4537E"]
WINDOW = 5.0            # seconds, sliding window for the rate curve


def read_log(path):
    """Parse the '# key=value' header lines, then the data rows."""
    cfg = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line[1:].strip().split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg, pd.read_csv(path, comment="#")


def outcome_mask(df):
    """
    Boolean series marking clean detections, whichever schema the log uses.

    A log without a status column contains only detected buffers, since the
    failure paths return before writing a row. The undetected count is then
    unknown, which the caller must flag rather than silently report as 100%.
    """
    if "status" in df.columns:
        return df["status"] == "ok", True
    if "sync_ok" in df.columns:
        return df["sync_ok"] == 1, False
    return pd.Series(True, index=df.index), False


def counts(df):
    if "status" in df.columns:
        c = df["status"].value_counts().to_dict()
        return {k: int(c.get(k, 0)) for k in CATS}
    ok, _ = outcome_mask(df)
    return {"ok": int(ok.sum()), "sync_fail": int((~ok).sum()),
            "ambiguous": 0, "no_preamble": 0}


def rate_series(df, window=WINDOW):
    """
    Clean detections per second, in a sliding window stepped by half a window.

    Wall-clock rate, so it mixes link quality with host processing speed. The
    final window is partly empty yet still divided by the full width, so its
    dip is an artefact -- read the trend, not the last point.
    """
    ok, _ = outcome_mask(df)
    t = df["t_s"].astype(float).to_numpy()
    ok = ok.to_numpy()
    if len(t) < 2:
        return [], []
    span = t[-1] - t[0]
    if span <= 0:
        return [], []
    step = window / 2
    centres = [t[0] + window / 2 + i * step
               for i in range(max(1, int(span / step)))]
    return centres, [ok[(t >= c - window/2) & (t < c + window/2)].sum() / window
                     for c in centres]


def rate_err(c):
    """Success fraction and its binomial standard error."""
    n = sum(c.values())
    if not n:
        return 0.0, 0.0
    p = c["ok"] / n
    return p, (p * (1 - p) / n) ** 0.5


def main(paths):
    logs = []
    for p in paths:
        cfg, df = read_log(p)
        _, complete = outcome_mask(df)
        n = int(cfg.get("N", 0))
        n_len = int(cfg.get("N_UP", 0))
        logs.append(dict(path=p, cfg=cfg, df=df, c=counts(df),
                         complete=complete, N=n, n_len=n_len,
                         tag=f"N={n}\nL={n_len}"))
    logs.sort(key=lambda L: (L["N"], L["n_len"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()
    x = range(len(logs))
    labels = [f"{L['tag']}\n({sum(L['c'].values())})" for L in logs]

    # --- panel 1: stacked outcome counts ---
    bottom = [0] * len(logs)
    for cat, col in zip(CATS, COLORS):
        vals = [L["c"][cat] for L in logs]
        ax1.bar(x, vals, 0.6, bottom=bottom, label=cat, color=col)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("buffers")
    ax1.set_title("Outcome counts")
    ax1.legend(fontsize=8)

    # --- panel 2: success rate with binomial standard error ---
    rates, errs = zip(*(rate_err(L["c"]) for L in logs))
    ax2.bar(x, [100*r for r in rates], 0.6,
            yerr=[100*e for e in errs], capsize=4, color="#378ADD")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("clean detections (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Success rate (bars = binomial s.e.)")
    for xi, r in zip(x, rates):
        ax2.text(xi, 100*r + 2, f"{100*r:.1f}", ha="center", fontsize=8)

    # --- panel 3: rate over time ---
    for i, L in enumerate(logs):
        c, r = rate_series(L["df"])
        if c:
            ax3.plot(c, r, lw=1.4, color=LINE[i % len(LINE)],
                     label=L["tag"].replace("\n", " "))
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("clean detections / s")
    ax3.set_title(f"Detection rate ({WINDOW:.0f} s sliding window)")
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8)

    # --- panel 4: success rate vs N_LEN, one line per N ---
    # Separating the two factors matters because N and N_LEN act through
    # different mechanisms: N sets the chirp duration and hence the frequency
    # resolution, while N_LEN sets how many slots are available to form a run.
    # Plotted as a single ordering they would be confounded.
    by_n = defaultdict(list)
    for L in logs:
        p, e = rate_err(L["c"])
        by_n[L["N"]].append((L["n_len"], 100*p, 100*e))
    for i, (n, pts) in enumerate(sorted(by_n.items())):
        pts.sort()
        xs, ys, es = zip(*pts)
        ax4.errorbar(xs, ys, yerr=es, marker="o", capsize=4, lw=1.4,
                     color=LINE[i % len(LINE)], label=f"N = {n}")
    ax4.set_xlabel("N_LEN (up/down chirps per region)")
    ax4.set_ylabel("clean detections (%)")
    ax4.set_ylim(0, 105)
    ax4.set_title("Effect of preamble length, by N")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    if not all(L["complete"] for L in logs):
        fig.text(0.5, 0.005,
                 "One or more logs have no status column: undetected buffers "
                 "were never written, so those rates are upper bounds.",
                 ha="center", fontsize=8, color="#A32D2D")

    plt.tight_layout()
    plt.savefig("log_comparison.png", dpi=150)
    print("written log_comparison.png")
    for L in logs:
        p, e = rate_err(L["c"])
        print(f"{L['path']}  N={L['N']} L={L['n_len']}  "
              f"{100*p:5.1f}% +/- {100*e:.1f}  {L['c']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python compare_logs.py log1.csv log2.csv [...]")
    main(sys.argv[1:])