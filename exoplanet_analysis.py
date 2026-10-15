

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

import os
os.makedirs("figures", exist_ok=True)

nasa = pd.read_csv("oec.csv")


# ---------- helpers ----------

def save(fig, name):
    """Save a figure at 900x600 px and close it."""
    fig.tight_layout()
    fig.savefig(name, dpi=100)
    plt.close(fig)


def new_fig():
    return plt.subplots(figsize=(9, 6))


def lump(series, n):
    """Keep the n most common categories, group the rest as 'Other'."""
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top) | series.isna(), "Other")


def status_group(series):
    """Simplify the 'ListsPlanetIsOn' column into three groups."""
    s = series.fillna("")
    return np.select(
        [s.str.contains("Confirmed"), s.str.contains("Controversial")],
        ["Confirmed", "Controversial"],
        default="Other",
    )


def fit_line(ax, x, y, color, logx=False, logy=False, ci=False):
    """Draw a least-squares line (fit on log scale if the axis is log)."""
    xt = np.log10(x) if logx else np.asarray(x, dtype=float)
    yt = np.log10(y) if logy else np.asarray(y, dtype=float)
    if len(xt) < 3:
        return
    slope, intercept = np.polyfit(xt, yt, 1)
    grid = np.linspace(xt.min(), xt.max(), 100)
    pred = slope * grid + intercept

    if ci:
        resid = yt - (slope * xt + intercept)
        s = np.sqrt(np.sum(resid ** 2) / (len(xt) - 2))
        se = s * np.sqrt(1 / len(xt) + (grid - xt.mean()) ** 2 / np.sum((xt - xt.mean()) ** 2))
        lo, hi = pred - 1.96 * se, pred + 1.96 * se
        if logy:
            lo, hi = 10 ** lo, 10 ** hi
        gx = 10 ** grid if logx else grid
        ax.fill_between(gx, lo, hi, color=color, alpha=0.2)

    gx = 10 ** grid if logx else grid
    gy = 10 ** pred if logy else pred
    ax.plot(gx, gy, color=color, linewidth=1.5)


# ---------- fig 1: mass vs radius ----------
d = nasa.dropna(subset=["PlanetaryMassJpt", "RadiusJpt"])
d = d[(d.PlanetaryMassJpt > 0) & (d.RadiusJpt > 0)]
fig, ax = new_fig()
ax.scatter(d.PlanetaryMassJpt, d.RadiusJpt, s=12, color="black")
ax.set(xscale="log", yscale="log", title="Mass vs Radius",
       xlabel="Planetary Mass", ylabel="Planet Radius")
save(fig, "figures/senay_fig1.png")


# ---------- fig 2: number of exoplanets discovered by method ----------
d = nasa.dropna(subset=["DiscoveryMethod"])
fig, ax = new_fig()
sns.countplot(data=d, x="DiscoveryMethod",
              order=d.DiscoveryMethod.value_counts().index, color="grey", ax=ax)
ax.set(title="Number of Exoplanets Discovered by Method",
       xlabel="Discovery Method", ylabel="Count")
save(fig, "figures/senay_fig2.png")


# ---------- fig 3: distance vs temperature ----------
d = nasa.dropna(subset=["SemiMajorAxisAU", "SurfaceTempK"])
fig, ax = new_fig()
ax.scatter(d.SemiMajorAxisAU, d.SurfaceTempK, s=12, alpha=0.6, color="black")
ax.set(title="Distance vs Temperature",
       xlabel="Distance from Star (AU)", ylabel="Surface Temperature (K)")
save(fig, "figures/senay_fig3.png")


# ---------- fig 4: star temperature vs planet temperature ----------
d = nasa.dropna(subset=["HostStarTempK", "SurfaceTempK"])
fig, ax = new_fig()
ax.scatter(d.HostStarTempK, d.SurfaceTempK, s=12, alpha=0.6, color="black")
ax.set(title="Star Temperature vs Planet Temperature",
       xlabel="Host Star Temperature", ylabel="Planet Surface Temperature")
save(fig, "figures/senay_fig4.png")


# ---------- fig 5: temperature by discovery method ----------
d = nasa.dropna(subset=["DiscoveryMethod", "SurfaceTempK"])
fig, ax = new_fig()
sns.boxplot(data=d, x="DiscoveryMethod", y="SurfaceTempK", color="white", ax=ax)
ax.set(title="Temperature by Discovery Method",
       xlabel="Discovery Method", ylabel="Surface Temperature")
save(fig, "figures/senay_fig5.png")


# ---------- fig 6: discoveries per year ----------
d = nasa.dropna(subset=["DiscoveryYear"]).groupby("DiscoveryYear").size()
fig, ax = new_fig()
ax.plot(d.index, d.values, color="black")
ax.set(title="Discoveries Per Year", xlabel="Year", ylabel="Count")
save(fig, "figures/senay_fig6.png")


# ---------- fig 7: status by discovery method ----------
d = nasa.dropna(subset=["DiscoveryMethod", "ListsPlanetIsOn"])
counts = pd.crosstab(d.DiscoveryMethod, d.ListsPlanetIsOn)
fig, ax = new_fig()
counts.plot(kind="bar", stacked=True, ax=ax, rot=0)
ax.set(title="Status by Discovery Method", xlabel="Discovery Method", ylabel="Count")
ax.legend(title="ListsPlanetIsOn", fontsize=7, title_fontsize=8,
          bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=2)
fig.set_size_inches(9, 9)
save(fig, "figures/senay_fig7.png")


# ---------- fig 8: orbital period distribution ----------
d = nasa.dropna(subset=["PeriodDays"])
fig, ax = new_fig()
ax.hist(d.PeriodDays, bins=30, color="grey", edgecolor="white")
ax.set(title="Orbital Period Distribution", xlabel="Orbital Period", ylabel="Count")
save(fig, "figures/senay_fig8.png")


# ---------- fig 9: distance vs temperature by discovery method ----------
d = nasa.dropna(subset=["SemiMajorAxisAU", "SurfaceTempK", "DiscoveryMethod"])
fig, ax = new_fig()
sns.scatterplot(data=d, x="SemiMajorAxisAU", y="SurfaceTempK",
                hue="DiscoveryMethod", alpha=0.7, ax=ax)
ax.set(title="Distance vs Temperature by Discovery Method",
       xlabel="Distance from Star (AU)", ylabel="Surface Temperature (K)")
save(fig, "figures/senay_fig9.png")


# ---------- fig 10: mass vs radius with orbital period ----------
d = nasa.dropna(subset=["PlanetaryMassJpt", "RadiusJpt", "PeriodDays"])
fig, ax = new_fig()
sns.scatterplot(data=d, x="PlanetaryMassJpt", y="RadiusJpt", size="PeriodDays",
                sizes=(10, 300), alpha=0.5, color="black", ax=ax)
ax.set(title="Mass vs Radius with Orbital Period",
       xlabel="Planetary Mass", ylabel="Planet Radius")
ax.legend(title="Period")
save(fig, "figures/senay_fig10.png")


# ---------- fig 11: distance vs temperature with method, mass, and status ----------
d = nasa.dropna(subset=["SemiMajorAxisAU", "SurfaceTempK", "PlanetaryMassJpt"]).copy()
d = d[(d.SemiMajorAxisAU > 0) & (d.PlanetaryMassJpt > 0)]
d["status_group"] = status_group(d.ListsPlanetIsOn)
d["method_group"] = lump(d.DiscoveryMethod, 4)
d = d.dropna(subset=["method_group"])
g = sns.relplot(data=d, x="SemiMajorAxisAU", y="SurfaceTempK", col="status_group",
                hue="method_group", size="PlanetaryMassJpt", sizes=(10, 300),
                alpha=0.5, height=6, aspect=0.5)
g.set(xscale="log")
g.set_axis_labels("Distance from Star (AU)", "Surface Temperature (K)")
g.set_titles("{col_name}")
g.figure.suptitle("Distance vs Temperature by Method and Status", y=1.02)
g.figure.set_size_inches(9, 6)
g.savefig("figures/senay_fig11.png", dpi=100, bbox_inches="tight")
plt.close(g.figure)


# ---------- fig 12: average temperature over time by method and status ----------
d = nasa.copy()
d["status_group"] = status_group(d.ListsPlanetIsOn)
d["method_group"] = lump(d.DiscoveryMethod, 4)
d = d.dropna(subset=["DiscoveryYear", "method_group", "SurfaceTempK", "PlanetaryMassJpt"])
d = (d.groupby(["DiscoveryYear", "method_group", "status_group"], as_index=False)
       .agg(avg_temp=("SurfaceTempK", "mean"), avg_mass=("PlanetaryMassJpt", "mean")))
statuses = sorted(d.status_group.unique())
methods = sorted(d.method_group.unique())
palette = dict(zip(methods, sns.color_palette(n_colors=len(methods))))
fig, axes = plt.subplots(1, len(statuses), figsize=(9, 6), sharey=True, squeeze=False)
for ax, status in zip(axes[0], statuses):
    sub = d[d.status_group == status]
    for method, grp in sub.groupby("method_group"):
        grp = grp.sort_values("DiscoveryYear")
        ax.plot(grp.DiscoveryYear, grp.avg_temp, color=palette[method], linewidth=1)
        ax.scatter(grp.DiscoveryYear, grp.avg_temp, s=grp.avg_mass * 20 + 10,
                   color=palette[method], alpha=0.7, label=method)
    ax.set(title=status, xlabel="Discovery Year")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
axes[0][0].set_ylabel("Average Surface Temperature")
handles = [plt.Line2D([], [], marker="o", color=palette[m], linestyle="") for m in methods]
fig.legend(handles, methods, title="Method", loc="center right")
fig.suptitle("Average Temperature Over Time by Method and Status")
fig.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig("figures/senay_fig12.png", dpi=100)
plt.close(fig)


# ---------- fig 13: mass vs radius with star temp and orbital period ----------
d = nasa.dropna(subset=["PlanetaryMassJpt", "RadiusJpt", "HostStarTempK", "PeriodDays"]).copy()
d = d[(d.PlanetaryMassJpt > 0) & (d.RadiusJpt > 0)]
d["star_temp_group"] = pd.cut(d.HostStarTempK, [-np.inf, 4500, 6000, np.inf],
                              right=False, labels=["cool", "medium", "hot"])
fig, ax = new_fig()
sns.scatterplot(data=d, x="PlanetaryMassJpt", y="RadiusJpt", hue="star_temp_group",
                size="PeriodDays", sizes=(10, 300), alpha=0.6, ax=ax)
ax.set(xscale="log", yscale="log", title="Mass vs Radius with Star Temp and Period",
       xlabel="Planetary Mass", ylabel="Planet Radius")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "figures/senay_fig13.png")


# ---------- fig 14: orbital period vs planetary mass by star temp group ----------
d = nasa.dropna(subset=["PeriodDays", "PlanetaryMassJpt", "HostStarTempK"]).copy()
d = d[(d.PeriodDays > 0) & (d.PlanetaryMassJpt > 0)]
labels = ["Cool (< 4500K)", "Medium (4500-6000K)", "Hot (> 6000K)"]
d["star_temp_group"] = pd.cut(d.HostStarTempK, [-np.inf, 4500, 6000, np.inf],
                              right=False, labels=labels)
palette = dict(zip(labels, sns.color_palette(n_colors=3)))
fig, ax = new_fig()
for label in labels:
    grp = d[d.star_temp_group == label]
    ax.scatter(grp.PlanetaryMassJpt, grp.PeriodDays, s=12, alpha=0.4,
               color=palette[label], label=label)
    fit_line(ax, grp.PlanetaryMassJpt, grp.PeriodDays, palette[label], logx=True, logy=True)
ax.set(xscale="log", yscale="log",
       title="Orbital Period vs Planetary Mass by Host Star Temperature",
       xlabel="Planetary Mass (Jupiter masses, log scale)",
       ylabel="Orbital Period (days, log scale)")
ax.legend(title="Star Temp Group")
save(fig, "figures/senay_fig14.png")


# ---------- fig 15: proportion of discovery methods across decades ----------
d = nasa.dropna(subset=["DiscoveryYear", "DiscoveryMethod"]).copy()
order = ["Before 2000", "2000s", "2010s", "2020s"]
d["decade"] = pd.cut(d.DiscoveryYear, [-np.inf, 2000, 2010, 2020, np.inf],
                     right=False, labels=order)
d["method_group"] = lump(d.DiscoveryMethod, 5)
props = pd.crosstab(d.decade, d.method_group, normalize="index").reindex(order)
fig, ax = new_fig()
props.plot(kind="bar", stacked=True, ax=ax, rot=0)
ax.set(title="Proportion of Discovery Methods Across Decades",
       xlabel="Decade", ylabel="Proportion")
ax.legend(title="Discovery Method", bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "figures/senay_fig15.png")


# ---------- fig 16: host star mass vs planetary mass by confirmation status ----------
d = nasa.dropna(subset=["HostStarMassSlrMass", "PlanetaryMassJpt"]).copy()
d = d[(d.HostStarMassSlrMass > 0) & (d.PlanetaryMassJpt > 0)]
d["status_group"] = status_group(d.ListsPlanetIsOn)
statuses = sorted(d.status_group.unique())
palette = dict(zip(statuses, sns.color_palette(n_colors=len(statuses))))
fig, axes = plt.subplots(1, len(statuses), figsize=(9, 6), sharey=True, squeeze=False)
for ax, status in zip(axes[0], statuses):
    grp = d[d.status_group == status]
    ax.scatter(grp.HostStarMassSlrMass, grp.PlanetaryMassJpt, s=12, alpha=0.5,
               color=palette[status])
    fit_line(ax, grp.HostStarMassSlrMass, grp.PlanetaryMassJpt, palette[status],
             logy=True, ci=True)
    ax.set(yscale="log", title=status, xlabel="Host Star Mass (solar masses)")
axes[0][0].set_ylabel("Planetary Mass (Jupiter masses, log scale)")
fig.suptitle("Host Star Mass vs Planetary Mass by Confirmation Status")
save(fig, "figures/senay_fig16.png")


# ---------- fig 17: eccentricity distribution by discovery method ----------
d = nasa.dropna(subset=["Eccentricity", "DiscoveryMethod"]).copy()
d = d[(d.Eccentricity >= 0) & (d.Eccentricity <= 1)]
d["method_group"] = lump(d.DiscoveryMethod, 4)
fig, ax = new_fig()
sns.violinplot(data=d, x="method_group", y="Eccentricity", hue="method_group",
               inner="quart", cut=0, legend=False, ax=ax)
for c in ax.collections:
    c.set_alpha(0.6)
sns.stripplot(data=d, x="method_group", y="Eccentricity", color="black",
              jitter=0.1, alpha=0.2, size=2, ax=ax)
ax.set(title="Orbital Eccentricity Distribution by Discovery Method",
       xlabel="Discovery Method",
       ylabel="Eccentricity (0 = circular, 1 = highly elliptical)")
save(fig, "figures/senay_fig17.png")


# ---------- fig 18: cumulative discoveries over time by method ----------
d = nasa.dropna(subset=["DiscoveryYear", "DiscoveryMethod"]).copy()
d["method_group"] = lump(d.DiscoveryMethod, 4)
d = (d.groupby(["DiscoveryYear", "method_group"]).size().reset_index(name="n")
       .sort_values("DiscoveryYear"))
d["cumulative"] = d.groupby("method_group")["n"].cumsum()
fig, ax = new_fig()
sns.lineplot(data=d, x="DiscoveryYear", y="cumulative", hue="method_group",
             linewidth=2, marker="o", markersize=4, ax=ax)
ax.set(title="Cumulative Exoplanet Discoveries Over Time by Method",
       xlabel="Discovery Year", ylabel="Cumulative Discoveries")
ax.legend(title="Discovery Method")
save(fig, "figures/senay_fig18.png")

print("Done: 18 figures saved.")
