"""Build all publication figures from regenerated machine-readable results."""
from __future__ import annotations


def build_main_figures() -> None:
    """Build the four Main figures from the current machine-readable results."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ROOT = Path(__file__).resolve().parents[1]
    R = ROOT / "results"
    OUT = ROOT / "figures" / "main"
    OUT.mkdir(parents=True, exist_ok=True)

    P = {
        "bubble": "#5C4A72", "high": "#B65C45", "low": "#3A6F7D",
        "olive": "#6C6A4E", "gold": "#A37B55", "guide": "#A8A8A8",
        "light": "#D8D8D8", "dmft": "#5C4A72", "controlled": "#B65C45",
        "aniso": "#6C6A4E",
    }
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.6,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2, "lines.linewidth": 1.55,
        "lines.markersize": 4.0, "mathtext.fontset": "stixsans", "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })


    def clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")


    def plabel(ax, label):
        ax.text(-0.14, 1.045, label, transform=ax.transAxes, fontsize=9.6,
                fontweight="bold", va="top", ha="left")


    def save(fig, stem):
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.09)
        fig.savefig(
            OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.09,
            metadata={"Creator": "Reproducible Matplotlib figure builder"},
        )
        plt.close(fig)


    def first_crossing(x, y, q=0.5):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        for i in range(len(x) - 1):
            if y[i] == q:
                return x[i]
            if (y[i] - q) * (y[i + 1] - q) <= 0 and y[i + 1] != y[i]:
                return x[i] + (q - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        return np.nan


    # Source data
    pole_markers = pd.read_csv(R / "pole_kinematics" / "exact_pole_family_markers.csv")
    pinning = pd.read_csv(R / "pole_kinematics" / "pinning_gate.csv")
    five_robust = pd.read_csv(R / "five_bath_ed_dmft" / "five_bath_readout_robustness.csv")
    five_markers = pd.read_csv(R / "five_bath_ed_dmft" / "five_bath_dmft_markers.csv")
    five_curves = pd.read_csv(R / "five_bath_ed_dmft" / "five_bath_dmft_curves.csv")
    two_markers = pd.read_csv(R / "two_site_dmft" / "two_site_dmft_markers.csv")
    two_curves = pd.read_csv(R / "two_site_dmft" / "two_site_dmft_curves.csv")
    two_spectra = pd.read_csv(R / "two_site_dmft" / "two_site_dmft_spectral_snapshots.csv")
    two_broadening = pd.read_csv(R / "two_site_dmft" / "two_site_dmft_broadening_robustness.csv")
    two_windows = pd.read_csv(R / "two_site_dmft" / "two_site_dmft_high_energy_window_robustness.csv")
    path_curves = pd.read_csv(R / "path_discrimination" / "matched_candidate_curves.csv")
    path_markers = pd.read_csv(R / "path_discrimination" / "matched_candidate_markers_dense.csv")

    # Fig. 1: parent mismatch -> mechanistic answer -> pole evolution -> nonminimal validation
    fig = plt.figure(figsize=(7.20, 5.70))
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.965, bottom=0.105, top=0.965, wspace=0.34, hspace=0.34)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    x = pinning.u_over_uc.to_numpy(); he = pinning.HE_completion.to_numpy(); le = pinning.LE_completion.to_numpy()
    a.plot(x, he, color=P["high"], marker="o", markevery=3, label="high-energy")
    a.plot(x, le, color=P["low"], marker="s", ls="--", markerfacecolor="white", markeredgewidth=0.9, markevery=3, label="low-energy")
    for q, col in [(0.25, "#668A9A"), (0.50, P["guide"]), (0.75, P["gold"])]:
        a.axhline(q, color=col, lw=0.8, ls=":")
    a.plot([first_crossing(x, he, 0.5)], [0.5], "o", color=P["high"], ms=4.6)
    a.plot([first_crossing(x, le, 0.5)], [0.5], marker="s", color=P["low"], markerfacecolor="white", markeredgewidth=1, ms=4.6)
    a.set(xlim=(-0.02, 1.10), ylim=(-0.03, 1.05), xlabel="U/Uc", ylabel="completion q")
    a.legend(frameon=False, loc="upper left"); plabel(a, "(a)"); clean(a)
    for q, col, mk, ls, lw, lab in [
        (0.25, "#668A9A", "o", "--", 1.2, "q=0.25"),
        (0.50, P["bubble"], "s", "-", 1.8, "q=0.50"),
        (0.75, P["gold"], "^", ":", 1.35, "q=0.75"),
    ]:
        s = pole_markers[np.isclose(pole_markers.fraction, q)].sort_values("alpha")
        b.plot(s.alpha, s.high_minus_low, color=col, marker=mk, ls=ls, lw=lw,
               markerfacecolor="white", markeredgewidth=0.9, label=lab)
    b.axhline(0, color=P["guide"], lw=0.9)
    b.set(xlim=(-0.02, 1.02), ylim=(-0.36, 0.66), xlabel="trajectory interpolation α", ylabel="HE−LE marker gap")
    b.set_xticks([0, 0.25, 0.5, 0.75, 1]); b.legend(frameon=False, loc="upper right"); plabel(b, "(b)"); clean(b)
    l1, = c.plot(x, pinning.pole_energy, color=P["olive"], marker="o", markevery=3, markerfacecolor="white", markeredgewidth=0.9, label="pole energy")
    c.set(xlim=(-0.02, 1.10), ylim=(-0.15, 6.35), xlabel="U/Uc", ylabel="pole energy / t")
    c.tick_params(axis="y", colors=P["olive"]); c.spines["left"].set_color(P["olive"])
    c2 = c.twinx()
    l2, = c2.plot(x, pinning.R_total, color=P["bubble"], marker="s", ls="--", markevery=3, markerfacecolor="white", markeredgewidth=0.9, label="total residue")
    c2.set_ylim(-1, 44); c2.set_ylabel(r"total residue / $t^2$", color=P["bubble"]); c2.tick_params(axis="y", colors=P["bubble"])
    c2.spines["top"].set_visible(False); c2.spines["right"].set_color(P["bubble"])
    c.legend([l1, l2], [l1.get_label(), l2.get_label()], frameon=False, loc="upper center"); plabel(c, "(c)"); clean(c)
    five_default = five_robust[
        np.isclose(five_robust.eta, 0.10)
        & np.isclose(five_robust.wmin, 4.2)
        & np.isclose(five_robust.wmax, 8.8)
        & five_robust.fraction.isin([0.25, 0.50, 0.75])
    ].sort_values("fraction")
    for _, row in five_default.iterrows():
        d.plot([row.U_HE, row.U_LE], [row.fraction, row.fraction], color=P["light"], lw=1, zorder=1)
    d.scatter(five_default.U_HE, five_default.fraction, color=P["high"], marker="o", s=28, label="high-energy", zorder=3)
    d.scatter(five_default.U_LE, five_default.fraction, facecolors="white", edgecolors=P["low"], linewidths=1.15, marker="s", s=30, label="low-energy", zorder=3)
    d.set(xlim=(2.5, 8.3), ylim=(0.15, 0.85), xlabel="interaction strength U/t", ylabel="completion fraction q")
    d.set_yticks([0.25, 0.50, 0.75]); d.legend(frameon=False, loc="upper left"); plabel(d, "(d)"); clean(d)
    save(fig, "fig1_pole_kinematics")

    # Fig. 2: two-site-DMFT mechanism
    fig = plt.figure(figsize=(7.25, 6.10))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.97, bottom=0.09, top=0.91, wspace=0.34, hspace=0.44)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    square = two_curves[two_curves.model == "two_site_square"]
    for r, col, mk, ls, lab in [
        ("transport_quadratic", P["bubble"], "D", "-", "bubble"),
        ("high_energy_spectral", P["high"], "o", "-", "high-energy"),
        ("low_energy_spectral", P["low"], "s", "--", "low-energy"),
    ]:
        s = square[square.readout == r].sort_values("u_over_uc")
        a.plot(s.u_over_uc, s.direct_completion, color=col, marker=mk, markevery=5, ls=ls,
               markerfacecolor="white" if r != "high_energy_spectral" else col, markeredgewidth=0.9, label=lab)
    for q in [0.25, 0.50, 0.75]: a.axhline(q, color=P["guide"], lw=0.6, ls=":")
    a.set(xlim=(-0.02, 1.02), ylim=(-0.03, 1.03), xlabel="U/Uc", ylabel="completion q")
    a.legend(frameon=False, loc="upper left"); plabel(a, "(a)"); clean(a)
    for model, ls, alpha in [("two_site_square", "-", 1.0), ("two_site_anisotropic", "--", 0.95)]:
        sub = two_markers[(two_markers.model == model) & (two_markers.validity == "primary")]
        for r, col, mk in [("transport_quadratic", P["bubble"], "D"), ("high_energy_spectral", P["high"], "o"), ("low_energy_spectral", P["low"], "s")]:
            s = sub[sub.readout == r].sort_values("fraction")
            b.plot(s.fraction, s.direct_marker, color=col, ls=ls, marker=mk,
                   markerfacecolor="white" if r != "high_energy_spectral" else col, markeredgewidth=0.9, alpha=alpha)
    leg = b.legend(handles=[
        Line2D([0], [0], color=P["bubble"], marker="D", lw=1.5, markerfacecolor="white", label="bubble"),
        Line2D([0], [0], color=P["high"], marker="o", lw=1.5, label="high-energy"),
        Line2D([0], [0], color=P["low"], marker="s", lw=1.5, ls="--", markerfacecolor="white", label="low-energy"),
    ], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, borderaxespad=0, columnspacing=0.9, handletextpad=0.4)
    b.add_artist(leg)
    b.legend(handles=[Line2D([0], [0], color="black", lw=1.5, ls="-", label="square"), Line2D([0], [0], color="black", lw=1.5, ls="--", label="anisotropic")], frameon=False, loc="lower right")
    b.set(xlim=(0.22, 0.78), ylim=(0.20, 0.98), xlabel="completion fraction q", ylabel="marker position U/Uc")
    b.set_xticks([0.25, 0.50, 0.75]); plabel(b, "(b)"); clean(b)
    for off, u in zip([0, 0.55, 1.10, 1.65], [0, 0.55, 0.85, 1.02]):
        s = two_spectra[np.isclose(two_spectra.u_over_uc, u)].sort_values("omega")
        y = s.local_dos.to_numpy(); y = y / max(y.max(), 1e-9) * 0.42
        c.plot(s.omega, y + off, color=P["olive"], lw=1.4); c.fill_between(s.omega, off, y + off, color=P["olive"], alpha=0.10)
        c.text(-11.7, off + 0.31, f"U/Uc={u:.2f}", fontsize=7.0, ha="left", va="center")
    c.axvline(0, color=P["guide"], lw=0.8, ls=":")
    c.set(xlim=(-12.2, 12.2), ylim=(-0.05, 2.15), xlabel="omega/t", ylabel="normalized local DOS (offset)")
    c.set_yticks([]); plabel(c, "(c)"); clean(c)
    # Reviewer-driven mechanism upgrade: replace redundant stage-gap panel with
    # independently verified on-self-consistent-branch derivative contributions.
    mech = pd.read_csv(R / "revision_validation" / "physical_mechanism_fraction_at_markers.csv")
    for readout, quantity, col, ls, marker, lab in [
        ("HE", "residue_percent", P["high"], "-", "o", "HE: residue growth"),
        ("LE", "motion_percent", P["low"], "--", "s", "LE: inward pole motion"),
    ]:
        row = mech[mech.readout == readout].sort_values("fraction")
        d.plot(row.fraction, row[quantity], color=col, marker=marker, ls=ls,
               markerfacecolor="white" if readout == "LE" else col,
               markeredgewidth=0.9, label=lab)
    d.axhline(50, color=P["guide"], ls=":", lw=0.8)
    d.set(xlim=(0.20, 0.80), ylim=(0, 105), xlabel="completion fraction q",
          ylabel="integrated contribution (%)")
    d.set_xticks([0.25,0.50,0.75])
    d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.015),
             ncol=1, borderaxespad=0, handlelength=2.1, fontsize=7.1)
    plabel(d, "(d)"); clean(d)
    save(fig, "fig2_two_site_dmft_mechanism")

    # Fig. 3: robustness
    fig = plt.figure(figsize=(7.25, 6.10))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.97, bottom=0.09, top=0.91, wspace=0.34, hspace=0.44)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for eta, ls, col in [(0.035, "--", "#668A9A"), (0.040, "-", P["bubble"]), (0.050, ":", P["gold"])]:
        s = two_broadening[np.isclose(two_broadening.lattice_broadening, eta)].sort_values("fraction")
        a.plot(s.fraction, s.low_energy_spectral - s.high_energy_spectral, color=col, ls=ls, marker="o", markerfacecolor="white", markeredgewidth=0.9)
        a.plot(s.fraction, s.high_energy_spectral - s.transport_quadratic, color=col, ls=ls, marker="s", markerfacecolor="white", markeredgewidth=0.9)
    leg = a.legend(handles=[Line2D([0], [0], color="#668A9A", lw=1.5, ls="--", label="η=0.035"), Line2D([0], [0], color=P["bubble"], lw=1.5, ls="-", label="η=0.040"), Line2D([0], [0], color=P["gold"], lw=1.5, ls=":", label="η=0.050")], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, borderaxespad=0, columnspacing=0.9, handletextpad=0.4)
    a.add_artist(leg)
    a.legend(handles=[Line2D([0], [0], color="black", marker="o", lw=1.4, markerfacecolor="white", label="LE-HE"), Line2D([0], [0], color="black", marker="s", lw=1.4, markerfacecolor="white", label="HE-bubble")], frameon=False, loc="upper right")
    a.axhline(0, color=P["guide"], lw=0.8); a.set(xlim=(0.22, 0.78), ylim=(0, 0.45), xlabel="completion fraction q", ylabel="square-model gap in U/Uc")
    a.set_xticks([0.25, 0.50, 0.75]); a.text(-0.17, 1.075, "(a)", transform=a.transAxes, fontsize=9.6, fontweight="bold", va="top", ha="left"); clean(a)
    for model, col, ls in [("two_site_square", P["dmft"], "-"), ("two_site_anisotropic", P["aniso"], "--")]:
        s = two_windows[two_windows.model == model].copy(); s["LE_HE"] = s.low_energy_spectral - s.high_energy_spectral; s["HE_T"] = s.high_energy_spectral - s.transport_quadratic
        base = s[s.window_id == "w105_260"].sort_values("fraction")
        b.plot(base.fraction, base.LE_HE, color=col, ls=ls, marker="o")
        b.plot(base.fraction, base.HE_T, color=col, ls=ls, marker="s", markerfacecolor="white", markeredgewidth=0.9)
        pv = s.pivot_table(index="fraction", values=["LE_HE", "HE_T"], aggfunc=["min", "max"]); q = pv.index.to_numpy()
        b.fill_between(q, pv[("min", "LE_HE")].to_numpy(), pv[("max", "LE_HE")].to_numpy(), color=col, alpha=0.08)
        b.fill_between(q, pv[("min", "HE_T")].to_numpy(), pv[("max", "HE_T")].to_numpy(), color=col, alpha=0.04)
    leg = b.legend(handles=[Line2D([0], [0], color=P["dmft"], lw=1.5, ls="-", label="square"), Line2D([0], [0], color=P["aniso"], lw=1.5, ls="--", label="anisotropic")], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, borderaxespad=0, columnspacing=0.9, handletextpad=0.4)
    b.add_artist(leg)
    b.legend(handles=[Line2D([0], [0], color="black", marker="o", lw=1.4, label="LE-HE"), Line2D([0], [0], color="black", marker="s", lw=1.4, markerfacecolor="white", label="HE-bubble")], frameon=False, loc="upper right")
    b.axhline(0, color=P["guide"], lw=0.8); b.set(xlim=(0.22, 0.78), ylim=(0, 0.56), xlabel="completion fraction q", ylabel="window-robust gap in U/Uc")
    b.set_xticks([0.25, 0.50, 0.75]); plabel(b, "(b)"); clean(b)
    anisotropic = two_curves[two_curves.model == "two_site_anisotropic"]
    for r, col, mk, ls, lab in [("transport_quadratic", P["bubble"], "D", "-", "bubble"), ("high_energy_spectral", P["high"], "o", "-", "high-energy"), ("low_energy_spectral", P["low"], "s", "--", "low-energy")]:
        s = anisotropic[anisotropic.readout == r].sort_values("u_over_uc")
        c.plot(s.u_over_uc, s.direct_completion, color=col, marker=mk, markevery=5, ls=ls, markerfacecolor="white" if r != "high_energy_spectral" else col, markeredgewidth=0.9, label=lab)
    for q in [0.25, 0.50, 0.75]: c.axhline(q, color=P["guide"], lw=0.6, ls=":")
    c.set(xlim=(-0.02, 1.02), ylim=(-0.03, 1.03), xlabel="U/Uc", ylabel="completion q")
    c.legend(frameon=False, loc="upper left"); plabel(c, "(c)"); clean(c)
    d.plot(five_curves.U, five_curves.HE_completion, color=P["high"], marker="o", label="high-energy")
    d.plot(five_curves.U, five_curves.LE_completion, color=P["low"], marker="s", ls="--", markerfacecolor="white", markeredgewidth=0.9, label="low-energy")
    for q in [0.25, 0.50, 0.75]: d.axhline(q, color=P["guide"], lw=0.6, ls=":")
    for _, row in five_markers.iterrows(): d.plot([row.U_HE, row.U_LE], [row.fraction, row.fraction], color=P["light"], lw=1, zorder=1)
    d.set(xlim=(-0.3, 8.3), ylim=(-0.03, 1.08), xlabel="interaction strength U/t", ylabel="completion q")
    d.legend(frameon=False, loc="upper left"); plabel(d, "(d)"); clean(d)
    save(fig, "fig3_robustness")

    # Fig. 4: matched self-energy-evolution comparison
    fig = plt.figure(figsize=(7.25, 6.05))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.97, bottom=0.09, top=0.92, wspace=0.34, hspace=0.42)
    a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    mapr = {"transport_quadratic": ("bubble", P["bubble"], "D", "-"), "high_energy_spectral": ("high-energy", P["high"], "o", "-"), "low_energy_spectral": ("low-energy", P["low"], "s", "--")}
    for ax, cand, lab in [(a, "controlled", "(a)"), (b, "DMFT", "(b)")]:
        sub = path_curves[path_curves.candidate == cand]
        for r in ["transport_quadratic", "high_energy_spectral", "low_energy_spectral"]:
            _, col, mk, ls = mapr[r]; s = sub[sub.readout == r].sort_values("lambda")
            ax.plot(s["lambda"], s.direct_completion, color=col, marker=mk, markevery=4, ls=ls, markerfacecolor="white" if r != "high_energy_spectral" else col, markeredgewidth=0.9)
        for q in [0.25, 0.50, 0.75]: ax.axhline(q, color=P["guide"], lw=0.6, ls=":")
        ax.set(xlim=(-0.01, 0.96), ylim=(-0.03, 1.03), xlabel="path coordinate λ", ylabel="completion q")
        plabel(ax, lab); clean(ax)
    fig.legend(handles=[Line2D([0], [0], color=P["bubble"], marker="D", lw=1.5, markerfacecolor="white", label="bubble"), Line2D([0], [0], color=P["high"], marker="o", lw=1.5, label="high-energy"), Line2D([0], [0], color=P["low"], marker="s", lw=1.5, ls="--", markerfacecolor="white", label="low-energy")], loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, columnspacing=1.2, handletextpad=0.5)
    for cand, col, lab, ls, mk in [("controlled", P["controlled"], "controlled", "-", "o"), ("DMFT", P["dmft"], "DMFT", "--", "s")]:
        pv = path_markers[path_markers.candidate == cand].pivot_table(index="fraction", columns="readout", values="direct_marker").sort_index()
        c.plot(pv.index, pv.high_energy_spectral - pv.low_energy_spectral, color=col, marker=mk, ls=ls, markerfacecolor="white", markeredgewidth=0.9, label=lab)
    c.axhline(0, color=P["guide"], lw=0.8)
    _cvals = np.concatenate([line.get_ydata() for line in c.lines if len(line.get_ydata()) > 1])
    _cmin, _cmax = float(np.nanmin(_cvals)), float(np.nanmax(_cvals)); _cspan = max(_cmax-_cmin, 0.1)
    c.set(xlim=(0.03, 0.97), ylim=(_cmin-0.10*_cspan, _cmax+0.10*_cspan), xlabel="completion fraction q", ylabel="HE−LE marker gap")
    c.legend(frameon=False, loc="upper right"); plabel(c, "(c)"); clean(c)
    for cand, col, lab, filled in [("controlled", P["controlled"], "controlled", True), ("DMFT", P["dmft"], "DMFT", False)]:
        pv = path_markers[path_markers.candidate == cand].pivot_table(index="fraction", columns="readout", values="direct_marker").sort_index(); face = col if filled else "white"
        d.plot(pv.index, pv.high_energy_spectral - pv.transport_quadratic, color=col, marker="s", ls="-", markerfacecolor=face, markeredgewidth=0.9, label=f"{lab}: HE-bubble")
        d.plot(pv.index, pv.low_energy_spectral - pv.high_energy_spectral, color=col, marker="o", ls="--", markerfacecolor=face, markeredgewidth=0.9, label=f"{lab}: LE-HE")
    d.axhline(0, color=P["guide"], lw=0.8)
    _dvals = np.concatenate([line.get_ydata() for line in d.lines if len(line.get_ydata()) > 1])
    _dmin, _dmax = float(np.nanmin(_dvals)), float(np.nanmax(_dvals)); _dspan = max(_dmax-_dmin, 0.1)
    d.set(xlim=(0.03, 0.97), ylim=(_dmin-0.10*_dspan, _dmax+0.10*_dspan), xlabel="completion fraction q", ylabel="stage separation in λ")
    d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, borderaxespad=0, columnspacing=0.9, handlelength=1.7, handletextpad=0.4)
    plabel(d, "(d)"); clean(d)
    save(fig, "fig4_matched_candidate_discrimination")

    print("built Main Figs. 1-4")


def build_supplementary_figures() -> None:
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ROOT = Path(__file__).resolve().parents[1]
    R = ROOT / 'results'
    OUT = ROOT / 'figures' / 'supplemental'
    OUT.mkdir(parents=True, exist_ok=True)

    P = {
        'low':'#3A6F7D', 'high':'#B65C45', 'omega':'#6C6A4E', 'residue':'#7B5E7D',
        'q025':'#668A9A', 'q050':'#5C4A72', 'q075':'#A37B55',
        'guide':'#A8A8A8', 'connector':'#BDBDBD', 'dark':'#444444'
    }

    mpl.rcParams.update({
        'figure.facecolor':'white', 'axes.facecolor':'white', 'font.family':'DejaVu Sans',
        'font.size':8.0, 'axes.labelsize':8.6, 'xtick.labelsize':7.2, 'ytick.labelsize':7.2,
        'legend.fontsize':6.7, 'axes.linewidth':0.8, 'xtick.major.width':0.8,
        'ytick.major.width':0.8, 'xtick.major.size':3.2, 'ytick.major.size':3.2,
        'lines.linewidth':1.55, 'lines.markersize':4.0, 'mathtext.fontset':'stixsans',
        'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'
    })

    def clean(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out')

    def panel(ax, lab):
        ax.text(-0.14, 1.055, lab, transform=ax.transAxes, fontsize=9.5,
                fontweight='bold', va='top', ha='left')

    def save(fig, stem):
        fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight', pad_inches=0.06)
        fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.06,
                    metadata={'Creator':'Reproducible Matplotlib figure builder'})
        plt.close(fig)

    # ---------------- S1 controlled pole-family depth ----------------
    markers = pd.read_csv(R/'pole_kinematics'/'exact_pole_family_markers.csv')
    comp = pd.read_csv(R/'pole_kinematics_detail'/'residue_motion_competition.csv')
    pinning = pd.read_csv(R/'pole_kinematics'/'pinning_gate.csv')
    fig = plt.figure(figsize=(7.3, 6.1))
    gs = fig.add_gridspec(2,2,left=0.09,right=0.97,bottom=0.10,top=0.96,wspace=0.38,hspace=0.42)
    a,b,c,d = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

    m50 = markers[np.isclose(markers.fraction,0.5)].sort_values('alpha')
    a.plot(m50.alpha,m50.transport_quadratic,color=P['residue'],marker='D',markerfacecolor='white',label='bubble')
    a.plot(m50.alpha,m50.high_energy_spectral,color=P['high'],marker='o',label='high-energy')
    a.plot(m50.alpha,m50.low_energy_spectral,color=P['low'],marker='s',ls='--',markerfacecolor='white',label='low-energy')
    a.set_xlabel('trajectory interpolation α'); a.set_ylabel('q=0.50 marker coordinate')
    a.set_xlim(-0.02,1.02); a.set_ylim(-0.02,0.95)
    a.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,borderaxespad=0.0,columnspacing=1.0,handletextpad=0.5)
    panel(a,'(a)'); clean(a)

    piv = markers.pivot_table(index='fraction',columns='alpha',values='high_minus_low').sort_index()
    vm = np.nanmax(np.abs(piv.values))
    im = b.imshow(piv.values,origin='lower',aspect='auto',extent=[piv.columns.min(),piv.columns.max(),piv.index.min(),piv.index.max()],cmap='coolwarm',vmin=-vm,vmax=vm)
    b.contour(piv.columns.to_numpy(),piv.index.to_numpy(),piv.values,levels=[0],colors='k',linewidths=0.9)
    b.set_xlabel('trajectory interpolation α'); b.set_ylabel('completion fraction q')
    # Non-color sign witnesses for grayscale readability.
    b.text(0.08,0.86,'+',fontsize=11,fontweight='bold',ha='center',va='center',color='black')
    b.text(0.86,0.86,'−',fontsize=11,fontweight='bold',ha='center',va='center',color='black')
    cb = fig.colorbar(im,ax=b,fraction=0.046,pad=0.03); cb.set_label('HE−LE marker gap',fontsize=8.2)
    panel(b,'(b)')

    cm = comp[np.isclose(comp.fraction,0.5)].pivot_table(index='p_motion',columns='p_residue',values='high_minus_low').sort_index()
    vm2 = np.nanmax(np.abs(cm.values))
    im2 = c.imshow(cm.values,origin='lower',aspect='auto',extent=[cm.columns.min()-0.25,cm.columns.max()+0.25,cm.index.min()-0.25,cm.index.max()+0.25],cmap='coolwarm',vmin=-vm2,vmax=vm2)
    c.contour(cm.columns.to_numpy(),cm.index.to_numpy(),cm.values,levels=[0],colors='k',linewidths=0.8)
    c.set_xticks(cm.columns); c.set_yticks(cm.index); c.set_xlabel('residue-growth exponent'); c.set_ylabel('pole-motion exponent')
    # Non-color sign witnesses for grayscale readability.
    c.text(3.75,0.65,'+',fontsize=11,fontweight='bold',ha='center',va='center',color='black')
    c.text(1.15,3.75,'−',fontsize=11,fontweight='bold',ha='center',va='center',color='black')
    cb2 = fig.colorbar(im2,ax=c,fraction=0.046,pad=0.03); cb2.set_label('HE−LE gap at q=0.50',fontsize=8.2)
    panel(c,'(c)')

    d.plot(pinning.u_over_uc,pinning.A0_dmft_rel,color=P['low'],marker='o',markevery=3,label='moving-pole path')
    d.plot(pinning.u_over_uc,pinning.A0_pinned_rel,color=P['high'],marker='s',ls='--',markerfacecolor='white',markevery=3,label='zero-energy pinned pole')
    d.axhline(0.5,color=P['guide'],lw=0.7,ls=':')
    d.set_xlabel('U/Uc'); d.set_ylabel('A(0)/A(0) at U=0'); d.set_xlim(-0.02,1.10); d.set_ylim(-0.02,1.04)
    d.legend(frameon=False,loc='upper right'); panel(d,'(d)'); clean(d)
    save(fig,'figS1_controlled_pole_family_detail')

    # ---------------- S2 two-site DMFT analytic/numerical validation ----------------
    conv = pd.read_csv(R/'two_site_dmft'/'two_site_dmft_convergence.csv')
    solver = pd.read_csv(R/'two_site_dmft'/'two_site_dmft_solver.csv')
    fig = plt.figure(figsize=(7.3,6.1))
    gs = fig.add_gridspec(2,2,left=0.09,right=0.97,bottom=0.10,top=0.96,wspace=0.38,hspace=0.42)
    a,b,c,d = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

    cc = conv[(conv.model=='two_site_square') & np.isclose(conv.fraction,0.5)]
    for ro,col,mk,lab in [('transport_quadratic',P['residue'],'D','bubble'),('high_energy_spectral',P['high'],'o','high-energy'),('low_energy_spectral',P['low'],'s','low-energy')]:
        ss = cc[cc.readout==ro].sort_values('nk')
        a.plot(ss.nk,ss.direct_marker,color=col,marker=mk,markerfacecolor='white' if ro!='high_energy_spectral' else col,markeredgewidth=0.9,label=lab)
    a.set_xlabel('k-grid linear size Nk'); a.set_ylabel('q=0.50 marker position U/Uc')
    a.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,borderaxespad=0.0,columnspacing=0.9,handletextpad=0.5)
    panel(a,'(a)'); clean(a)

    sq = solver[solver.model=='two_site_square'].sort_values('u_over_uc')
    b.plot(sq.u_over_uc,sq.analytic_quasiparticle_weight,color=P['low'],lw=1.7,label='analytic Z')
    b.plot(sq.u_over_uc,sq.numerical_quasiparticle_weight,color=P['high'],ls='--',marker='o',markevery=4,markerfacecolor='white',markeredgewidth=0.9,label='numerical Z')
    b.set_xlabel('U/Uc'); b.set_ylabel('quasiparticle weight Z'); b.set_xlim(-0.02,1.10); b.set_ylim(-0.03,1.04)
    b.legend(frameon=False,loc='upper right'); panel(b,'(b)'); clean(b)

    c.semilogy(sq.u_over_uc,np.maximum(np.abs(sq.analytic_z_error),1e-14),color=P['residue'],marker='s',markerfacecolor='white',markeredgewidth=0.9)
    c.set_xlabel('U/Uc'); c.set_ylabel('|Z numerical − Z analytic|'); c.set_xlim(-0.02,1.10)
    panel(c,'(c)'); clean(c)

    for model,col,ls,lab in [('two_site_square',P['residue'],'-','square'),('two_site_anisotropic',P['omega'],'--','anisotropic')]:
        ss=solver[solver.model==model].sort_values('u_over_uc')
        d.semilogy(ss.u_over_uc,ss.sum_rule_abs_error_max,color=col,ls=ls,marker='o',markevery=4,markerfacecolor='white',markeredgewidth=0.9,label=lab)
    d.set_xlabel('U/Uc'); d.set_ylabel('max spectral-sum error'); d.set_xlim(-0.02,1.10)
    d.legend(frameon=False,loc='upper left'); d.text(0.02,1.025,'(d)',transform=d.transAxes,fontsize=9.5,fontweight='bold',ha='left',va='bottom'); clean(d)
    save(fig,'figS2_two_site_dmft_validation')

    # ---------------- S3 five-bath ED-DMFT physics detail, no insets ----------------
    ed = pd.read_csv(R/'five_bath_ed_dmft'/'five_bath_dmft_curves.csv')
    spec = pd.read_csv(R/'five_bath_ed_dmft_detail'/'five_bath_dmft_spectra.csv')
    fig = plt.figure(figsize=(7.3,6.1))
    gs = fig.add_gridspec(2,2,left=0.09,right=0.97,bottom=0.10,top=0.96,wspace=0.38,hspace=0.42)
    a,b,c,d = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

    for u,off in zip([0.0,3.0,6.0,8.0],[0,0.33,0.66,0.99]):
        ss=spec[np.isclose(spec.U,u)].sort_values('omega')
        y=ss.A.to_numpy(); y=y/np.nanmax(y)*0.26
        a.plot(ss.omega,y+off,color=P['omega'],lw=1.35); a.fill_between(ss.omega,off,y+off,color=P['omega'],alpha=0.08)
        a.text(-11.5,off+0.21,f'U/t={u:g}',fontsize=6.8,va='center')
    a.axvline(0,color=P['guide'],lw=0.7,ls=':'); a.set_xlim(-12,12); a.set_ylim(-0.02,1.30); a.set_yticks([])
    a.set_xlabel('omega/t'); a.set_ylabel('local DOS (offset)'); panel(a,'(a)'); clean(a)

    b.plot(ed.U,ed.Z,color=P['low'],marker='o',label='Z')
    b.plot(ed.U,ed.A0/ed.A0.iloc[0],color=P['high'],marker='s',ls='--',markerfacecolor='white',markeredgewidth=0.9,label='A(0)/A0(0)')
    b.set_xlabel('interaction strength U/t'); b.set_ylabel('normalized low-energy measure'); b.set_xlim(-0.2,8.2); b.set_ylim(-0.03,1.04)
    b.legend(frameon=False,loc='upper right'); panel(b,'(b)'); clean(b)

    c.plot(ed.U,ed.pole_energy,color=P['omega'],marker='o',markerfacecolor='white',markeredgewidth=0.9)
    c.set_xlabel('interaction strength U/t'); c.set_ylabel('self-energy pole energy'); c.set_xlim(-0.2,8.2)
    panel(c,'(c)'); clean(c)

    d.semilogy(ed.U,np.maximum(ed.pole_height,1e-10),color=P['residue'],marker='s',ls='--',markerfacecolor='white',markeredgewidth=0.9)
    d.set_xlabel('interaction strength U/t'); d.set_ylabel('self-energy pole height'); d.set_xlim(-0.2,8.2)
    panel(d,'(d)'); clean(d)
    save(fig,'figS3_five_bath_dmft_detail')

    # ---------------- S4 complete DMFT robustness ----------------
    broad = pd.read_csv(R/'two_site_dmft'/'two_site_dmft_broadening_robustness.csv')
    win = pd.read_csv(R/'two_site_dmft'/'two_site_dmft_high_energy_window_robustness.csv')
    endp = pd.read_csv(R/'five_bath_ed_dmft_detail'/'endpoint_robustness.csv')
    edrob = pd.read_csv(R/'five_bath_ed_dmft'/'five_bath_readout_robustness.csv')
    fig = plt.figure(figsize=(7.3,8.7))
    gs = fig.add_gridspec(3,2,left=0.09,right=0.97,bottom=0.075,top=0.965,wspace=0.38,hspace=0.42)
    a,b,c,d,e,f = [fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]

    for eta,ls,col in [(0.035,'--',P['q025']),(0.040,'-',P['q050']),(0.050,':',P['q075'])]:
        ss=broad[np.isclose(broad.lattice_broadening,eta)].sort_values('fraction')
        a.plot(ss.fraction,ss.low_energy_spectral-ss.high_energy_spectral,color=col,ls=ls,marker='o',markerfacecolor='white',markeredgewidth=0.9,label=f'η={eta:.3f}')
        a.plot(ss.fraction,ss.high_energy_spectral-ss.transport_quadratic,color=col,ls=ls,marker='s',markerfacecolor='white',markeredgewidth=0.9)
    a.axhline(0,color=P['guide'],lw=0.8); a.set_xticks([0.25,0.5,0.75]); a.set_xlim(0.22,0.78); a.set_ylim(0.0,0.45)
    a.set_xlabel('completion fraction q'); a.set_ylabel('square-model gap in U/Uc')
    leg=a.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,borderaxespad=0.0,columnspacing=0.9,handletextpad=0.5); a.add_artist(leg)
    a.legend(handles=[Line2D([0],[0],color='black',marker='o',lw=1.4,markerfacecolor='white',label='LE−HE'),Line2D([0],[0],color='black',marker='s',lw=1.4,markerfacecolor='white',label='HE−bubble')],frameon=False,loc='upper right')
    # Keep the panel letter clear of the top y tick at publication size.
    a.text(-0.16, 1.09, '(a)', transform=a.transAxes, fontsize=9.5, fontweight='bold', va='top', ha='left')
    clean(a)

    for model,col,ls in [('two_site_square',P['residue'],'-'),('two_site_anisotropic',P['omega'],'--')]:
        ss=win[win.model==model].copy(); ss['LE_HE']=ss.low_energy_spectral-ss.high_energy_spectral; ss['HE_T']=ss.high_energy_spectral-ss.transport_quadratic
        base=ss[ss.window_id=='w105_260'].sort_values('fraction')
        b.plot(base.fraction,base.LE_HE,color=col,ls=ls,marker='o')
        b.plot(base.fraction,base.HE_T,color=col,ls=ls,marker='s',markerfacecolor='white',markeredgewidth=0.9)
        pv=ss.pivot_table(index='fraction',values=['LE_HE','HE_T'],aggfunc=['min','max']); q=pv.index.to_numpy()
        b.fill_between(q,pv[('min','LE_HE')].to_numpy(),pv[('max','LE_HE')].to_numpy(),color=col,alpha=0.08)
        b.fill_between(q,pv[('min','HE_T')].to_numpy(),pv[('max','HE_T')].to_numpy(),color=col,alpha=0.04)
    b.axhline(0,color=P['guide'],lw=0.8); b.set_xlim(0.22,0.78); b.set_xticks([0.25,0.5,0.75]); b.set_ylim(0.0,0.56)
    b.set_xlabel('completion fraction q'); b.set_ylabel('high-energy-window gap in U/Uc')
    leg=b.legend(handles=[Line2D([0],[0],color=P['residue'],lw=1.5,ls='-',label='square'),Line2D([0],[0],color=P['omega'],lw=1.5,ls='--',label='anisotropic')],frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=2,borderaxespad=0.0,columnspacing=1.0,handletextpad=0.5); b.add_artist(leg)
    b.legend(handles=[Line2D([0],[0],color='black',marker='o',lw=1.4,label='LE−HE'),Line2D([0],[0],color='black',marker='s',lw=1.4,markerfacecolor='white',label='HE−bubble')],frameon=False,loc='upper right')
    panel(b,'(b)'); clean(b)

    # 5-bath robustness: full window/eta points for each q, no aggregation into an inset.
    windows = list(dict.fromkeys(edrob['window'].tolist()))
    x = np.arange(len(windows))
    for ax,qv,lab in [(c,0.25,'(c)'),(d,0.50,'(d)'),(e,0.75,'(e)')]:
        ss=edrob[np.isclose(edrob.fraction,qv)]
        for eta,col,mk,ls in [(0.08,P['q025'],'o','--'),(0.10,P['q050'],'s','-'),(0.12,P['q075'],'^',':')]:
            ee=ss[np.isclose(ss.eta,eta)].set_index('window').loc[windows]
            ax.plot(x,ee.HE_minus_LE,color=col,marker=mk,ls=ls,markerfacecolor='white',markeredgewidth=0.9,label=f'η={eta:.2f}')
        ax.axhline(0,color=P['guide'],lw=0.8)
        ax.set_xticks(x,windows,rotation=28,ha='right'); ax.set_ylabel('U_HE − U_LE'); ax.set_xlabel('high-energy window id')
        ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,borderaxespad=0.0,columnspacing=0.8,handletextpad=0.4)
        panel(ax,lab); clean(ax)

    for q,col,mk,ls in [(0.25,P['q025'],'o','--'),(0.50,P['q050'],'s','-'),(0.75,P['q075'],'^',':')]:
        ss=endp[np.isclose(endp.fraction,q)].sort_values('Umax')
        f.plot(ss.Umax,ss.delta,color=col,marker=mk,ls=ls,markerfacecolor='white',markeredgewidth=0.9,label=f'q={q:.2f}')
    f.axhline(0,color=P['guide'],lw=0.8); f.set_xlabel('chosen endpoint Umax/t'); f.set_ylabel('U_HE − U_LE'); f.set_xlim(6.7,8.3)
    f.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,borderaxespad=0.0,columnspacing=0.9,handletextpad=0.5)
    panel(f,'(f)'); clean(f)
    save(fig,'figS4_dmft_robustness')

    # ---------------- S5 numerical/reproducibility audits ----------------
    solver = pd.read_csv(R/'two_site_dmft'/'two_site_dmft_solver.csv')
    five = pd.read_csv(R/'five_bath_ed_dmft'/'five_bath_dmft_curves.csv')
    five_diag = pd.read_csv(R/'five_bath_ed_dmft_detail'/'five_bath_solver_diagnostics.csv')
    fig = plt.figure(figsize=(7.3,6.1))
    gs = fig.add_gridspec(2,2,left=0.09,right=0.97,bottom=0.10,top=0.96,wspace=0.38,hspace=0.42)
    a,b,c,d = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

    # Two-site solver closure: the self-consistency residual stays at machine precision.
    for model,col,ls,lab in [('two_site_square',P['residue'],'-','square'),('two_site_anisotropic',P['omega'],'--','anisotropic')]:
        ss=solver[solver.model==model].sort_values('u_over_uc')
        a.semilogy(ss.u_over_uc,np.maximum(np.abs(ss.self_consistency_residual),1e-18),color=col,ls=ls,marker='o',markevery=4,markerfacecolor='white',markeredgewidth=0.9,label=lab)
    a.set_xlabel('U/Uc'); a.set_ylabel('|self-consistency residual|'); a.set_xlim(-0.02,1.10)
    a.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=2,borderaxespad=0.0,columnspacing=0.9,handletextpad=0.5); panel(a,'(a)'); clean(a)

    # Half filling of the two-site impurity solution.
    for model,col,ls,lab in [('two_site_square',P['residue'],'-','square'),('two_site_anisotropic',P['omega'],'--','anisotropic')]:
        ss=solver[solver.model==model].sort_values('u_over_uc')
        b.semilogy(ss.u_over_uc,np.maximum(np.abs(ss.impurity_occupancy-1.0),1e-18),color=col,ls=ls,marker='s',markevery=4,markerfacecolor='white',markeredgewidth=0.9,label=lab)
    b.set_xlabel('U/Uc'); b.set_ylabel('|impurity occupancy − 1|'); b.set_xlim(-0.02,1.10)
    b.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=2,borderaxespad=0.0,columnspacing=0.9,handletextpad=0.5); panel(b,'(b)'); clean(b)

    # Five-bath ED-DMFT bath fitting and self-consistency diagnostics.
    c.semilogy(five.U,np.maximum(five.fit_error,1e-16),color=P['high'],marker='o',label='bath-fit error')
    c.semilogy(five.U,np.maximum(five.sc_residual,1e-16),color=P['low'],marker='s',ls='--',markerfacecolor='white',markeredgewidth=0.9,label='self-consistency residual')
    c.set_xlabel('interaction strength U/t'); c.set_ylabel('five-bath solver residual'); c.set_xlim(-0.2,8.2)
    c.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=2,borderaxespad=0.0,columnspacing=0.8,handletextpad=0.4); c.text(-0.18,1.09,'(c)',transform=c.transAxes,fontsize=9.5,fontweight='bold',va='top',ha='left'); clean(c)

    # Finite-window spectral sum diagnostic for the five-bath validation.
    d.semilogy(five.U,np.maximum(np.abs(five.sumrule-1.0),1e-16),color=P['residue'],marker='o',label='regenerated ED path')
    d.set_xlabel('interaction strength U/t'); d.set_ylabel('|finite-window spectral sum − 1|'); d.set_xlim(-0.2,8.2)
    d.legend(frameon=False,loc='upper left'); d.text(0.02,1.025,'(d)',transform=d.transAxes,fontsize=9.5,fontweight='bold',ha='left',va='bottom'); clean(d)
    save(fig,'figS5_solver_diagnostics')

    print('built S1-S5')


def build_revision_supplementary_figures() -> None:
    """Revised SM evidence: physical tangent closure and explicit readout dependence.

    Inputs are generated by the review_validation phase using original primitive
    model definitions and saved five-bath ED input, not manually prepared plots.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root=Path(__file__).resolve().parents[1]
    data=root/'results'/'revision_validation'
    out=root/'figures'/'supplemental'
    out.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.0,
        'axes.labelsize':8.5,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
        'legend.fontsize':7.0,'axes.linewidth':0.8,'pdf.fonttype':42,
        'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
    C={'high':'#B65C45','low':'#3A6F7D','res':'#7B5E7D',
       'motion':'#6C6A4E','guide':'#B2B2B2','dark':'#484848'}
    def label(ax,lab):
        ax.text(-.145,1.055,lab,transform=ax.transAxes,fontweight='bold',
                fontsize=9.2,ha='left',va='top')
    def clean(ax):
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out')
    def save(fig,stem):
        fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',pad_inches=.11,
                    metadata={'Creator':'JPCM revision reproducible figures'})
        fig.savefig(out/(stem+'.png'),dpi=260,bbox_inches='tight',pad_inches=.11)
        plt.close(fig)

    df=pd.read_csv(data/'self_consistent_path_adaptive.csv')
    fig=plt.figure(figsize=(7.30,6.0))
    gs=fig.add_gridspec(2,2,left=.095,right=.965,bottom=.10,top=.95,
                       wspace=.35,hspace=.39)
    a,b,c,d=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    x=df['u'].to_numpy()
    for ax,obs,lab in ((a,'HE','(a)'),(b,'LE','(b)')):
        ax.plot(x,df[obs+'_res'],color=C['res'],lw=1.45,label='residue')
        ax.plot(x,df[obs+'_motion'],color=C['motion'],ls='--',lw=1.45,label='pole motion')
        ax.plot(x,df[obs+'_total'],color=C['dark'],ls=':',lw=1.25,label='sum')
        ax.axhline(0,color=C['guide'],lw=.8)
        ax.axvline(1,color=C['guide'],ls='--',lw=.75)
        ax.set(xlim=(0,1.08),xlabel=r'$U/U_c$',ylabel=r'$d R_{\mathrm{%s}}/dU$'%obs)
        ax.legend(frameon=False,loc='upper left' if obs=='HE' else 'lower left',ncol=1,fontsize=6.8)
        label(ax,lab);clean(ax)
    for obs,col in (('HE',C['high']),('LE',C['low'])):
        y=df[obs].to_numpy(); yrec=df[obs+'_fit'].to_numpy()
        c.plot(x,y,color=col,lw=1.6,label=obs+' direct')
        c.plot(x,yrec,color=col,lw=1.0,ls='--',label=obs+' reconstructed')
    c.axvline(1,color=C['guide'],ls='--',lw=.75)
    c.set(xlim=(0,1.08),xlabel=r'$U/U_c$',ylabel='spectral functional')
    c.legend(frameon=False,loc='upper left',fontsize=6.8)
    label(c,'(c)');clean(c)
    for obs,col in (('HE',C['high']),('LE',C['low'])):
        d.semilogy(x,np.maximum(np.abs(df[obs+'_fit_error']),1e-11),color=col,
                   lw=1.4,label=obs)
    d.axvline(1,color=C['guide'],ls='--',lw=.75)
    d.set(xlim=(0,1.08),ylim=(1e-11,1e-3),xlabel=r'$U/U_c$',
          ylabel='absolute integral-closure error')
    d.legend(frameon=False,loc='upper left')
    d.text(-.24,1.075,'(d)',transform=d.transAxes,fontweight='bold',fontsize=9.2,ha='left',va='top');clean(d)
    save(fig,'figS6_self_consistent_tangent')

    alt=pd.read_csv(data/'five_bath_alternative_readouts_from_regenerated_ED.csv')
    robust=pd.read_csv(root/'results'/'five_bath_ed_dmft'/'five_bath_readout_robustness.csv')
    fig=plt.figure(figsize=(7.3,3.65))
    gs=fig.add_gridspec(1,2,left=.105,right=.965,bottom=.20,top=.89,wspace=.35)
    a,b=[fig.add_subplot(gs[0,i]) for i in range(2)]
    for lab,col,marker,ls in [('LE Fermi-window',C['low'],'s','-'),('A0 spectral value',C['motion'],'o','--'),('Z Matsubara proxy',C['res'],'D',':')]:
        q=alt[alt.LE_readout==lab].sort_values('fraction')
        a.plot(q.fraction,q.HE_minus_LE,color=col,marker=marker,ls=ls,
               markerfacecolor='white',markeredgewidth=.9,label=lab)
    a.axhline(0,color=C['guide'],lw=.8)
    a.set(xlim=(.20,.80),ylim=(-.5,1.25),xlabel='completion fraction q',
          ylabel='HE minus LE marker (U/t)')
    a.set_xticks([.25,.50,.75]); a.legend(frameon=False,fontsize=6.8,loc='center left',bbox_to_anchor=(.02,.46))
    label(a,'(a)');clean(a)
    # Six original physical windows. Aggregate over three broadenings and three q values.
    summary=robust.groupby(['wmin','wmax']).HE_minus_LE.agg(lambda s:float(np.mean(s<0))).reset_index().sort_values(['wmin','wmax'])
    xx=np.arange(len(summary)); yy=summary.HE_minus_LE.to_numpy(float)
    b.plot(xx,yy,color=C['high'],marker='o',lw=0,ms=5.0)
    b.axhline(1,color=C['guide'],ls=':',lw=.7)
    b.set(xticks=xx,ylim=(-.08,1.10),xlabel='HE window (|ω|/t)',ylabel='fraction HE-first')
    b.set_xticklabels([f'{i:.1f}–{j:.1f}' for i,j in zip(summary.wmin,summary.wmax)],rotation=28,ha='right',fontsize=6.8)
    label(b,'(b)');clean(b)
    save(fig,'figS7_alternative_spectral_readouts')

def build_all_figures() -> None:
    build_main_figures()
    build_supplementary_figures()
    build_revision_supplementary_figures()


if __name__ == "__main__":
    build_all_figures()
