from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import ListedColormap

WIDTH_COL_PT = 241.14749
WIDTH_TEXT_PT = 506.295

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

SUBGROUP_COLORS = [CB_color_cycle[i] for i in [1, 2, 3]]

PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "font.size": 9,
    "pgf.texsystem": 'pdflatex',
    "pgf.preamble": "\\RequirePackage[tt=false, type1=true]{libertine}\\RequirePackage[varqu]{zi4}\\RequirePackage[libertine]{newtxmath}\\RequirePackage{color}",
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1,
    "axes.prop_cycle": cycler('color', CB_color_cycle),
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
}

GOLDEN_RATIO = (5 ** .5 - 1) / 2


def get_color_map():
    return ListedColormap(CB_color_cycle)


def get_page_width():
    return WIDTH_TEXT_PT / 72.27


def get_column_width():
    return WIDTH_COL_PT / 72.27


def get_presentation_size():
    # Figure width in inches
    fig_width_in = WIDTH_COL_PT * 1 / 72.27
    # Figure height in inches
    fig_height_in = fig_width_in * GOLDEN_RATIO
    return fig_width_in, fig_height_in


def get_presentation_size_page():
    # Figure width in inches
    fig_width_in = WIDTH_TEXT_PT * 1 / 72.27
    # Figure height in inches
    fig_height_in = fig_width_in * GOLDEN_RATIO
    return fig_width_in, fig_height_in


def get_subplot_size_col(subplots=2):
    # Calculate divisor for needed space between plots
    scale = 1 / (subplots * 1.063829787)
    # Figure width in inches
    fig_width_in = WIDTH_COL_PT * scale / 72.27
    # Figure height in inches
    fig_height_in = fig_width_in * GOLDEN_RATIO * 1.12
    return fig_width_in, fig_height_in


def get_subplot_size_text(subplots=3):
    # Calculate divisor for needed space between plots
    scale = 1 / (subplots * 1.063829787)
    # Figure width in inches
    fig_width_in = WIDTH_TEXT_PT * scale / 72.27
    # Figure height in inches
    fig_height_in = fig_width_in * GOLDEN_RATIO * 1.12
    return fig_width_in, fig_height_in


def _configure_matplotlib(use_tex: bool) -> bool:
    """Point matplotlib at the right backend and strip the LaTeX-only rcParams.

    ``PARAMS`` asks for ``text.usetex`` and a serif font that only exists in a
    TeX installation. Only the pgf backend can honour that; every other output
    format renders through Agg with LaTeX disabled, so figures (PDF included)
    build without TeX installed.

    Returns ``True`` when LaTeX text rendering is active.
    """
    matplotlib.use("pgf" if use_tex else "agg", force=True)
    params = dict(PARAMS)
    if not use_tex:
        params = {k: v for k, v in params.items()
                  if not k.startswith("pgf.") and k not in ("text.usetex", "font.serif", "font.family")}
        params["text.usetex"] = False
    plt.rcParams.update(params)
    return use_tex


def configure_output_backend(output) -> bool:
    """Select the matplotlib backend from *output*'s file extension."""
    return _configure_matplotlib(Path(output).suffix.lower() == ".pgf")


def configure_output_format(output_format: str) -> bool:
    """Select the matplotlib backend from a format name ('pdf', 'pgf', 'png')."""
    return _configure_matplotlib(str(output_format).lower().lstrip(".") == "pgf")


def tex_text(text: str) -> str:
    """Return *text* unchanged for LaTeX output, else without TeX-only markup.

    Labels carry TeX spacing macros such as ``\\,``; outside ``usetex`` those
    would be drawn literally, so they are replaced by their Unicode equivalent.
    """
    if plt.rcParams.get("text.usetex", False):
        return text
    return text.replace("\\,", " ").replace("\\%", "%")
