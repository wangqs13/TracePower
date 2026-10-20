"""Plot the experiment results."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)


def figure1(rows, output):
    styles = {"full": dict(color="#D68136", linestyle="--", label="Non-truncated"),
              "truncated": dict(color="#208E83", linestyle="-", label="Truncated")}
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), squeeze=False)
    for ri, (family, label) in enumerate((("wishart32", r"Random $GG^\dag$"),
                                         ("uniform32", "Uniform"), ("geometric85", "Geometric"))):
        for qi, q in enumerate((1.25, 1.5, 2.5, 3.5)):
            ax = axes[ri, qi]
            for method in styles:
                subset = sorted((r for r in rows if r["family"] == family and
                                 float(r["q"]) == q and r["method"] == method), key=lambda r: int(r["n"]))
                x = [int(r["N"]) for r in subset]
                ax.plot(x, [float(r["mae"]) for r in subset], **styles[method],
                        linewidth=1.6, marker="o", markersize=2)
                ax.fill_between(x, [float(r["ci_lower"]) for r in subset],
                                [float(r["ci_upper"]) for r in subset],
                                color=styles[method]["color"], alpha=.15, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks([320, 1280, 5120])
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if ri == 0:
                ax.set_title(rf"$q = {q:g}$", fontsize=14)
            if qi == 0:
                ax.set_ylabel(label + "\nMean absolute error", fontsize=13)
            if ri == 2:
                ax.set_xlabel(r"Total copies $N$", fontsize=13)
            ax.tick_params(labelsize=12)
            style(ax)
    handles = [Line2D([], [], **value, linewidth=1.8) for value in styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=13)
    fig.subplots_adjust(top=.92, bottom=.08, left=.08, right=.99, hspace=.3, wspace=.38)
    save(fig, output, "figure1")


def math_integer(value, _):
    return "$" + f"{int(value):,}".replace(",", "{,}") + "$"


def figure2(rows, output):
    fig, axes = plt.subplots(3, 2, figsize=(6.35, 6.65), squeeze=False, sharex="row")
    for ri, q in enumerate((.5, 1.5, 2.5)):
        for ci, (family, label) in enumerate((("uniform", "Uniform spectrum"),
                                            ("head_flat", r"$1/2 +$ uniform tail"))):
            ax = axes[ri, ci]
            subset = sorted((r for r in rows if r["family"] == family and float(r["q"]) == q),
                            key=lambda r: int(r["d"]))
            if not subset:
                continue
            maximum = max(int(r["d"]) for r in subset)
            # Missing thresholds remain gaps, rather than fabricated zero values.
            y = [int(r["N_empirical"]) if r["N_empirical"] != "" else float("nan") for r in subset]
            ax.plot([int(r["d"]) for r in subset], y, color="#218D84", marker="o",
                    markersize=2, linewidth=1.6)
            finite = [v for v in y if v == v]
            ax.set_ylim(0, max(finite) * 1.08 if finite else 1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if q < 1:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                formatter.set_useOffset(False)
            else:
                formatter = FuncFormatter(math_integer)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim(0, maximum * 1.02)
            ax.set_xticks([0, maximum // 2, maximum])
            ax.xaxis.set_major_formatter(FuncFormatter(math_integer))
            ax.tick_params(axis="both", labelsize=12, labelbottom=True)
            if ri == 2:
                ax.set_xlabel(r"Dimension $d$", fontsize=13)
            if ci == 0:
                ax.set_ylabel(rf"$q = {q:g}$" + "\nRequired total copies", fontsize=12)
            if ri == 0:
                ax.set_title(label, fontsize=14)
            if q == 2.5:
                ax.set_ylim(0, 230)
                ax.set_yticks([0, 50, 100, 150, 200])
            style(ax)
    fig.text(.5, .97, r"$\varepsilon=0.1$", ha="center", va="top", fontsize=13)
    fig.subplots_adjust(top=.88, bottom=.085, left=.17, right=.94, hspace=.33, wspace=.42)
    save(fig, output, "figure2")


def draw(results, output):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 12, "axes.labelsize": 13, "mathtext.fontset": "cm",
                         "mathtext.default": "it", "axes.formatter.use_mathtext": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    if "1" in results:
        figure1(results["1"], output)
    if "2" in results:
        figure2(results["2"], output)
