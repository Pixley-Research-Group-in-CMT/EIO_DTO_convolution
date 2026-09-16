"""Plot the raw and historically smoothed interface resistivity."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, FIGURE_DIR, FIELDS


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
})

data = pd.read_csv(DATA_DIR / "transport_summary.csv")
colors = plt.get_cmap("magma")(np.linspace(0.18, 0.82, len(FIELDS)))

# Panel a shows the absolute raw interface resistivity. Panel b isolates the
# angular magnetoresistance after applying the archived smoothing convention.
figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
for field, color in zip(FIELDS, colors):
    raw = data[(data["field"] == field) & (data["mode"] == "raw")]
    smoothed = data[(data["field"] == field) & (data["mode"] == "smoothed")]
    axes[0].plot(
        raw["angle"], raw["rho"], marker="o", ms=2.2, lw=1.0,
        color=color, label=fr"$H={field:g}$",
    )
    axes[1].plot(
        smoothed["angle"], smoothed["delta_rho_over_rho0"],
        marker="o", ms=2.2, lw=1.0, color=color,
    )

axes[0].set_xlabel(r"Field angle $\phi$ (degrees)")
axes[0].set_ylabel(r"Interface resistivity $\rho_{\mathrm{int}}$ (a.u.)")
axes[0].legend(frameon=False, fontsize=7, handlelength=1.5)
axes[0].text(0.02, 0.96, "a", transform=axes[0].transAxes, va="top", fontweight="bold")

axes[1].set_xlabel(r"Field angle $\phi$ (degrees)")
axes[1].set_ylabel(r"$[\rho(\phi)-\rho(0)]/\rho(0)$")
axes[1].text(0.02, 0.96, "b", transform=axes[1].transAxes, va="top", fontweight="bold")

# Use a common angular range so the field-dependent curves can be compared
# directly across the two panels.
for axis in axes:
    axis.set_xlim(0, 60)
    axis.set_xticks([0, 15, 30, 45, 60])
    axis.tick_params(width=0.8, length=3)

FIGURE_DIR.mkdir(exist_ok=True)
output = FIGURE_DIR / "interface_resistivity.pdf"
figure.savefig(
    output,
    bbox_inches="tight",
    metadata={"CreationDate": None, "ModDate": None},
)
plt.close(figure)
print(f"Wrote {output}")
