#!/usr/bin/env python

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

__all__ = [
    'plot_survival',
    'export_survival',
    'plot_cnv',
]

# fontTools' subsetter sets its own logger to INFO; __init__.py's
# _suppress_logging() can't reach it since matplotlib/fontTools are only
# imported here, lazily, long after that one-time sweep runs.
logging.getLogger('fontTools').setLevel(logging.WARNING)

# fold the arabic ('Grade 2') and roman ('Grade II') notations for the same
# WHO grade onto a single combined label so both map to one colour/strip
_GRADE_LABELS = {
    'Grade 1': 'Grade 1/I',   'Grade I':   'Grade 1/I',
    'Grade 2': 'Grade 2/II',  'Grade II':  'Grade 2/II',
    'Grade 3': 'Grade 3/III', 'Grade III': 'Grade 3/III',
    'Grade 4': 'Grade 4/IV',  'Grade IV':  'Grade 4/IV',
}
# qualitative palette (green/blue/orange/red) chosen so adjacent grades stay
# clearly distinguishable while still rising with WHO grade severity
_GRADE_COLORS = {
    'Grade 1/I':   '#4DAF4A',
    'Grade 2/II':  '#377EB8',
    'Grade 3/III': '#FF7F00',
    'Grade 4/IV':  '#E41A1C',
}
_GRADE_ORDER = ['Grade 1/I', 'Grade 2/II', 'Grade 3/III', 'Grade 4/IV']

# strip plots: every reference dot is drawn at this one size (points**2, so the
# printed diameter is its square root) -- the dots differ in colour and position
# only, never in size. Rows sit 1.0 apart, so a half-band below 0.5 keeps the
# grades from bleeding into each other.
_STRIP_DOT_SIZE  = 4
_STRIP_BAND_HW   = 0.32

# the sample's own row is given more room than a grade row: it is one
# prediction rather than a cloud, and the gap is what reads it as a block of
# its own instead of as a fifth grade
_STRIP_SAMPLE_ROW_H = 1.4   # sample row -> first grade row, in row units
_STRIP_SAMPLE_HW    = 0.45  # half-band the sample's row is given

# gauge panel: a 225 degree arc carrying the longest survival at its start and
# 0 years at its end, so the arc never closes -- the open wedge holds the
# legend. The whole thing is rotated 22.5 degrees anticlockwise from the
# horizontal, which puts both ends at the same angle either side of the
# vertical axis and centres the opening straight down. The panel is drawn twice,
# once per time scale: a compressed one that buys the early years room at the
# expense of the far tail, and a plain linear one.
_GAUGE_MAX_YEARS = 25.0
_GAUGE_THETA_LO  = np.deg2rad(-22.5)   # 0 years, right and just below level
_GAUGE_THETA_HI  = np.deg2rad(202.5)   # _GAUGE_MAX_YEARS, left, mirroring it

# exponent of the compressed scale: arc position is (years/max)**power, so 1.0
# is linear and 0.5 a square root. Lower compresses harder. As a share of the
# arc the first year takes 4% at 1.0, 12% at 0.65, 15% at 0.6, 20% at 0.5 --
# and 21% under the log1p this replaced, which is why a square root barely
# relieves the very short end while it does unpack the middle years.
_GAUGE_POWER = 0.6

# ticks per scale. Kept as two ladders even though they currently coincide: the
# short end (1, 2 years) sits comfortably on the compressed arc but crowds
# against 0 on the linear one, so they are likely to diverge again when either
# scale is retuned
_GAUGE_TICKS_COMPRESSED = [0, 1, 2, 5, 10, 15, 20, 25]
_GAUGE_TICKS_LINEAR     = [0, 1, 2, 5, 10, 15, 20, 25]

# radial layout in axis units (r runs 0..1)
_GAUGE_HUB_R    = 0.05
# the needles reach into the ring stack but stop at the inner edge of the
# Grade 2/II ring, which _plot_gauge works out from the rings actually drawn --
# a fixed radius would land on the wrong ring as soon as a grade is missing.
# Running them all the way out to the rim arc was tried and looked worse:
# eight shafts crossing every cloud. This value only applies when there are no
# grades to measure against
_GAUGE_NEEDLE_R_FALLBACK = 0.55
_GAUGE_NEEDLE_Z = 4
# clearance between an arrowhead's tip and that framing line, in r units --
# roughly 1.5 points as printed
_GAUGE_NEEDLE_GAP = 0.02
# arrowhead size in points (mutation_scale)
_GAUGE_HEAD_SCALE = 6
# any matplotlib ArrowStyle spec, its parameters in units of the scale above.
# 'simple' draws shaft and head as one filled shape, so tail_width is what sets
# the shaft: 0.14 x 6 puts it just under a point. The head is sized against it
# -- at roughly 5:1 it carries the needle's direction while the shaft stays a
# hairline. Linewidth is 0 -- on a filled style that is an outline, which would
# thicken the shaft a second time and blunt the head's swept-back flanks.
_GAUGE_ARROWSTYLE = "simple,head_length=0.8,head_width=0.7,tail_width=0.14"
_GAUGE_ARROW_LW   = 0
# printed radius of the hub logo, in the same r units: small enough that the
# needles read as needles rather than as spokes off a disc. logo-center2.png is
# 210 px wide, so at this size it prints well over 600 ppi, never upscaled
_GAUGE_LOGO_R   = 0.0975
# z-level of the hub: over the needles rather than under them, so the logo caps
# the point where they meet instead of being sliced up by them. The fallback
# dot sits just below it, high enough to cap the needles the same way
_GAUGE_HUB_Z    = 5.5
# the ring stack is pushed outwards and kept narrow: the rings only need to be
# told apart, while the space it frees on the inside is what the needles run in.
# Only the inner end is fixed -- the outer one is solved for in _plot_gauge, so
# that the gap to the rim arc equals the half spacing the rings keep between
# themselves rather than being a third margin of its own
_GAUGE_RING_LO  = 0.70
# a ring's dots spread this fraction of the ring spacing either side of it.
# What is left over is the corridor between two clouds, which is where the
# dial's framing hairline runs: at 0.40 that corridor is only a fifth of a
# spacing wide and the line crowds the neighbouring cloud, so the bands are
# kept well under half.
_GAUGE_RING_BAND_FRAC = 0.28
# capped as well: with only two or three grades present the spacing grows, and
# uncapped the bands would grow with it until the clouds read as filled discs
_GAUGE_RING_HW_MAX = 0.05
_GAUGE_H_IN     = 2.6

# background ramp along the arc: dark green where survival is longest, through
# yellow and orange, to dark red at 0 years. Two layers, because the grade dots
# and the model needles are drawn in these very same hues: over the plotting
# area the ramp is a faint wash that cannot swallow them, while the rim arc
# just outside the rings carries it at full strength, where nothing is drawn on
# top of it. Set _GAUGE_BG_ALPHA to 0 to keep the rim alone.
_GAUGE_BG_COLORS = ["#1B5E20", "#F2D024", "#F07C1E", "#9B1B1B"]  # long -> short
_GAUGE_BG_ALPHA  = 0.13

# the ramp reaches full dark green here and stays there for the rest of the
# arc, rather than running on to _GAUGE_MAX_YEARS. Beyond ~20 years the curves
# rest on the thinnest part of the training data (MAX_FOLLOW_UP_YEARS is 26.5)
# and a survival that long is exceptional in IDH mutant glioma anyway: keeping
# a gradient there would promise a precision the models do not have
_GAUGE_GREEN_FROM_YEARS = 20.0
_GAUGE_RIM_LO, _GAUGE_RIM_HI = 1.00, 1.06

# the panel title is drawn in the dial's own coordinates, just outside the rim
# and straight up, rather than handed to ax.set_title. A polar axes hangs its
# title off the bounding box, and this one's box is the full page width by the
# whole row: the title came out far above the dial and hard against the left
# edge, and how far depended on the row height, so the two gauges disagreed.
# At a fixed radius it sits the same distance over the arc on both.
_GAUGE_TITLE_R  = _GAUGE_RIM_HI
# and this far above it in points. The clearance has to be in points, not in
# radius: what the title has to clear is the row of year labels just outside
# the rim, and those are a fixed printed size whatever the dial is scaled to.
_GAUGE_TITLE_PAD_PT = 12

# report palette, installed as the figure's prop_cycle so anything that does
# draw several series stays inside the report's own hues
_CURVE_COLORS = [
    "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7",
    "#C4AD66", "#77BEDB", "#F7A541", "#A8786E",
]

# the ensemble is the only survival prediction the report draws: the curve, its
# median marker in the strip and the gauge needle are all the same number, so
# they are all the same colour. Its band is the members' own spread, which is
# why that is the same hue washed out rather than a colour of its own.
_ENSEMBLE_COLOR      = "#4878CF"
_ENSEMBLE_BAND_ALPHA = 0.18

# the curves are held per day (~9700 points over MAX_FOLLOW_UP_YEARS) and
# thinned to this many before drawing, which is what keeps the PDF small. Only
# the curve panels are drawn off the thinned frame -- the strip and the gauge
# read their medians off the full daily grid, so this number cannot move them.
_CURVE_DRAW_POINTS = 150

# where the curve panel is ticked, in years. Deliberately the same ladder the
# gauge carries (_GAUGE_TICKS_*) minus its crowded short end, so the flat panel
# and the dial are read against one scale
_CURVE_TICK_YEARS = [0, 5, 10, 15, 20, 25]
# and a small unlabelled mark on every year in between, so a reader can count
# to a year rather than estimate it against a five year gap. Runs to the whole
# year inside MAX_FOLLOW_UP_YEARS (26.5), which is as far as any curve reaches.
# Deliberately minor ticks: the rcParams grid is drawn on major ticks only, so
# these stay marks on the axis instead of ruling the panel into 26 columns.
_CURVE_MINOR_YEARS   = list(range(0, 27))
_CURVE_MINOR_TICK_PT = 1.5


# printed size of the logo, in inches rather than figure fractions so it comes
# out identically on every plot regardless of that figure's own dimensions
_LOGO_HEIGHT_IN = 0.30

# A4 portrait: the page format the report is printed/handed out on (ISO 216,
# 210x297 mm). Everything below is laid out in inches against this page rather
# than in figure fractions, so the printed result is independent of figsize.
_A4_W_IN, _A4_H_IN = 8.27, 11.69

# page margins for the plotting area; the top one also has to clear the whole
# header band (logo + title/description/byline + closing rule)
_MARGIN_L_IN, _MARGIN_R_IN, _MARGIN_T_IN = 0.95, 0.45, 2.35

# printed height of each row and the gap between them; the bottom margin is
# whatever is left over, so the block stays anchored under the header instead
# of being stretched down the full page
_CURVE_H_IN, _STRIP_H_IN, _ROW_GAP_IN = 1.05, 0.95, 0.7

# the optional CNV track on top. Shorter than the 3.2 inch standalone plot it
# shares its drawing code with: on A4 it only has to show which chromosomes
# moved, not carry gene labels
_CNV_H_IN = 1.49

# log2 ratio window the track is drawn in. Tighter than the +/-1.5 it was, and
# deliberately not symmetric: a whole-arm gain sits higher above zero than the
# matching loss sits below it, so an even window spends its lower half on space
# nothing reaches. Bins outside are clipped by the axes rather than rescaling it
# -- the window has to mean the same thing on every report to be comparable
# between samples.
_CNV_YLIM = (-1.2, 1.4)

# sex panel: one probability needs a strip, not a panel, so the bar keeps a
# fixed printed height and is centred in whatever cell it is given rather than
# being stretched to fill it.
_SEX_H_IN = 0.15     # the bar itself; its tick labels hang below it in the row
# share of its column the bar takes up, hung from the left edge so it lines up
# with the panel title: one probability on a 0..1 scale needs a short ruler,
# and a wide one only invites it to be read as a quantity of something. What is
# left over is where the two class initials sit
_SEX_WIDTH_FRAC = 0.85

# the probability row: the subtype bars on the left, the sex strip on the
# right. Both are read on a 0..1 axis, the bars on their y and the strip on its
# x, so the row reads as one block. The height is what the turned-on-their-side
# class names are read against, so it is the one thing this row cannot skimp on
_PROB_H_IN = 1.3

# subtype panel: only the leading classes get their own bar, the tail is summed
# into one. Dropping the tail instead would quietly stop the panel summing to
# 1, and a prediction spread thinly over thirty classes would then read as a
# confident one.
#
# Set above the 21 classes the current classifier carries, so in practice
# nothing is summed away: at that count each bar still has about a quarter inch
# of the panel, which is wider than a value label needs. The cap stays as the
# guard it was meant to be -- a classifier with a hundred classes would print
# them at a hair's width apiece and say nothing.
_SUBTYPE_TOP_N = 24
# a name turned on its side is read against the height of the panel: at 5.5pt
# in a 1.3 inch row that is a little over thirty characters, so this is where
# the clipping has to start
_SUBTYPE_LABEL_MAX = 26
# the called class is set apart from the runners-up, and the summed tail from
# both -- it is an aggregate, not a prediction
_SUBTYPE_COLORS      = ("#4878CF", "#A9BEDD")
_SUBTYPE_OTHER_COLOR = "#CFCFCF"
# bar width in column units. Slim: the panel is three quarters of the page and
# a name only needs a couple of points of it, so the width is better spent on
# more of the ranking than on thicker bars
_SUBTYPE_BAR_W = 0.40
_SUBTYPE_LABEL_Y0 = 0.03   # foot of the rotated names, just off the baseline

# average glyph width as a share of the font size, for a sans-serif at these
# sizes. Only used to decide whether a rotated name clears its own bar, so it
# is allowed to be an estimate -- getting it slightly wrong costs a label its
# white colouring, nothing more
_CHAR_W_FACTOR = 0.5

# headroom above the 0..1 scale, in the same units: the value labels all sit on
# one line just over the top of it, and that line needs somewhere to go. The y
# ladder is pinned to 0/0.5/1 regardless
_SUBTYPE_HEADROOM = 0.14
# where that line sits. Clear of 1.0 so it never touches a full-height bar, and
# far enough under 1 + _SUBTYPE_HEADROOM that the digits fit above it
_SUBTYPE_VALUE_Y  = 1.02

# room kept clear at the foot of every page, so a block can never be laid out
# past the paper edge
_PAGE_BOTTOM_MIN_IN = 0.35

# the page is split into four equal columns so the subtype bars take three
# quarters and the last column is left to the sex strip beside them. The curve
# panel and the strip carry a single prediction and no per-model legend to hang
# beside it, so those span the lot, as the CNV track already did.
_PAGE_COLS = 4
_PANEL_COLS = slice(0, 3)
_SIDE_COL   = slice(3, None)

# blue/orange rather than the conventional pink/blue: the pair stays separable
# under every common form of colour vision deficiency, and the panel is a
# probability readout, not a pictogram
_SEX_COLORS     = ("#4878CF", "#F7A541")
_SEX_RAMP_ALPHA = 0.55

# outside this band the call is effectively binary; inside it the sample is
# worth a second look. Loss of Y (common in older men and in tumours), XXY and
# X loss all land here, which is exactly what a hard label would hide
_SEX_UNCERTAIN_LO, _SEX_UNCERTAIN_HI = 0.10, 0.90

# header logo: the page margin is deliberately tighter than the plotting area's
# own (wider) left margin, which has to hold the y tick labels + axis title, so
# the logo can claim the space above and to the left of the text column
_PAGE_MARGIN_IN   = 0.3
_LOGO_REPORT_H_IN = 1.45

# report header band: a text column to the right of the logo (title, scope
# description, byline), closed off by a double rule that separates the header
# from the plotting area. All offsets are inches from the top of the A4 page,
# so the band keeps its printed proportions whatever the figsize.
# the text column is centred against the (taller) logo rather than hung from
# the page margin, so the two read as one band instead of two stacked blocks
_HEADER_LOGO_GAP_IN = 0.30   # logo -> text column
# the offsets below place the text block symmetrically against the logo, which
# runs from _PAGE_MARGIN_IN to _PAGE_MARGIN_IN + _LOGO_REPORT_H_IN: the block
# spans ~0.42-1.64 in, so both are centred on ~1.03 in. Changing the logo size
# or the page margin means re-centring these.
_HEADER_TITLE_Y_IN  = 0.42
_HEADER_SAMPLE_Y_IN = 0.66   # sample + timestamp, set off from the title above
_HEADER_HAIRLINE_IN = 0.86   # thin rule under the title, text column only
_HEADER_BODY_Y_IN   = 0.98
_HEADER_LINE_H_IN   = 0.16   # baseline-to-baseline within the description
_HEADER_BYLINE_Y_IN = 1.40
_HEADER_RULE_Y_IN   = 1.96   # double rule, full page width, clear of the logo

# near-black rather than pure black: keeps the header from out-shouting the
# curves it introduces, while staying clearly darker than the gray body text
_HEADER_DARK = "#1A1A1A"
_HEADER_GRAY = "#555555"

# hand-wrapped instead of textwrap'd: the break points are chosen so the bold
# run stays on one line and both lines come out roughly equal in printed width
_HEADER_BODY_LINES = [
    r"This reports the $\bf{predicted\ overall\ survival}$ given the DNA methylation profile, using multiple predictors",
    r"trained on >1,200 IDH mutant gliomas. The methodology is experimental and for research purpose only.",
]
_HEADER_BYLINE = "Written and designed by Dr. Y. Hoogstrate and Dr. R. Schoonhoven."

# the year the software was written, not the year this report is generated --
# a copyright notice dates the work, so it must not move with datetime.now().
# Both employers are named: each holds the rights to its own author's work
_COPYRIGHT_YEAR = 2025
_HEADER_COPYRIGHT = f"© {_COPYRIGHT_YEAR} Erasmus MC and Amsterdam UMC"

# rasterisation density for savefig; only the (bitmap) logo is affected -- the
# rest of the PDF is vector. logo.png is 1041x893, so even at the ~1.45 inch
# print height it carries ~615 dpi of its own and is downsampled, never upscaled.
_PDF_DPI = 600


def _add_logo(fig, pad_in=0.04, loc="right", height_in=_LOGO_HEIGHT_IN):
    """Stamp assets/logo.png into the top-*loc* corner of *fig*.

    Returns the printed width of the logo in inches (0.0 when it was skipped),
    so callers can lay text out next to it without knowing its aspect ratio.

    Silently does nothing when the file is absent -- the logo is decoration, so
    a missing asset should never break a prediction run. Call this last: the
    extra axes is placed in figure coordinates and would be ignored (matplotlib
    warns) by a tight_layout() run afterwards.
    """
    # deferred import: utils.py is loaded (via database.py) before
    # __init__.py finishes defining its module-level constants
    from . import ASSETS_PATH

    logo_path = ASSETS_PATH / "logo.png"
    if not logo_path.is_file():
        return 0.0

    img          = plt.imread(logo_path)
    fig_w, fig_h = fig.get_size_inches()

    width_in = height_in * (img.shape[1] / img.shape[0])

    h = height_in / fig_h
    w = width_in / fig_w
    x = pad_in / fig_w if loc == "left" else 1 - w - pad_in / fig_w

    ax = fig.add_axes([x, 1 - h - pad_in / fig_h, w, h], zorder=5)
    ax.imshow(img, interpolation="antialiased")
    ax.axis("off")

    return width_in


def _add_report_header(fig, sentrix_id, logo_w_in):
    """Draw the letterhead-style header band to the right of the logo.

    Holds everything that identifies the report (tool + version, sample,
    generation time), what it does and does not claim, and who signs off on it,
    so none of that has to be repeated as a plot title or page footer.
    """
    import matplotlib.lines as mlines

    # deferred import: utils.py is loaded (via database.py) before
    # __init__.py finishes defining its module-level constants
    from . import __version__

    # a missing logo collapses the gap too, so the text simply moves out to the
    # page margin rather than sitting behind an empty column
    text_x_in = _PAGE_MARGIN_IN + (logo_w_in + _HEADER_LOGO_GAP_IN if logo_w_in else 0.0)
    x         = text_x_in / _A4_W_IN
    x_right   = 1 - _MARGIN_R_IN / _A4_W_IN

    def _y(y_in):
        """Inches from the top of the page -> figure fraction."""
        return 1 - y_in / _A4_H_IN

    def _rule(x0, y_in, lw, color):
        fig.add_artist(mlines.Line2D([x0, x_right], [_y(y_in)] * 2,
                                     transform=fig.transFigure,
                                     lw=lw, color=color))

    # two separate texts rather than one "\n"-joined string: that way the sample
    # line gets its own size and the gap above it is set in inches, not by the
    # font's line spacing
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(x, _y(_HEADER_TITLE_Y_IN),
             f"cognition v{__version__}: autogenerated report on:",
             ha="left", va="top", fontsize=8, fontweight="bold",
             color=_HEADER_DARK)
    fig.text(x, _y(_HEADER_SAMPLE_Y_IN), f"{sentrix_id} ({timestamp})",
             ha="left", va="top", fontsize=7, fontweight="bold",
             color=_HEADER_DARK)

    _rule(x, _HEADER_HAIRLINE_IN, 0.4, "#BBBBBB")

    for i, line in enumerate(_HEADER_BODY_LINES):
        fig.text(x, _y(_HEADER_BODY_Y_IN + i * _HEADER_LINE_H_IN), line,
                 ha="left", va="top", fontsize=6.5, color=_HEADER_DARK)

    fig.text(x, _y(_HEADER_BYLINE_Y_IN), _HEADER_BYLINE,
             ha="left", va="top", fontsize=6, style="italic",
             color=_HEADER_GRAY)

    # upright rather than italic: the notice is a legal statement, not part of
    # the byline it sits under
    fig.text(x, _y(_HEADER_BYLINE_Y_IN + _HEADER_LINE_H_IN), _HEADER_COPYRIGHT,
             ha="left", va="top", fontsize=6, color=_HEADER_GRAY)

    # a single rule, not the classic letterhead double one: the header already
    # carries a hairline under the title, and a second pair would compete with
    # it. Spans the full page width, not just the text column
    _rule(_PAGE_MARGIN_IN / _A4_W_IN, _HEADER_RULE_Y_IN, 0.8, _HEADER_DARK)


def _van_der_corput(n, base=2):
    """First *n* terms of the van der Corput low-discrepancy sequence in [0, 1).

    Successive terms keep bisecting the largest remaining gap, which is what
    makes it fill a band evenly at every prefix length -- unlike random draws,
    which clump.
    """
    seq = np.zeros(n)
    for i in range(1, n + 1):
        value, denom, k = 0.0, 1.0, i
        while k:
            denom *= base
            k, rem = divmod(k, base)
            value += rem / denom
        seq[i - 1] = value
    return seq


def _quasirandom_offsets(values, halfwidth=_STRIP_BAND_HW, base=2):
    """Beeswarm-style y-offsets for a 1-D point cloud, after
    ggbeeswarm::geom_quasirandom (the vipor algorithm).

    *values* is binned along x; within a bin the offsets are handed out from a
    van der Corput sequence, scaled by how full that bin is relative to the
    busiest one. Dense regions therefore spread across the full *halfwidth*
    while sparse ones stay near the row centre, which is what gives the cloud
    its shape.

    Deterministic by construction -- no RNG, so re-runs on the same data are
    identical -- and, unlike a true beeswarm, offsets can never overflow the
    band or force points to be dropped when they don't physically fit.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.empty(0)

    if np.ptp(values) <= 0:  # all identical: one bin, spread over the full band
        nbins, bin_idx = 1, np.zeros(n, dtype=int)
    else:
        # ~sqrt(n) bins: fine enough to follow the distribution's shape without
        # letting single-point bins dictate the density scale
        nbins = max(1, min(64, int(np.ceil(np.sqrt(n)))))
        edges = np.linspace(values.min(), values.max(), nbins + 1)
        bin_idx = np.clip(np.digitize(values, edges[1:-1]), 0, nbins - 1)

    counts  = np.bincount(bin_idx, minlength=nbins)
    busiest = counts.max()
    offsets = np.zeros(n)

    for b in range(nbins):
        members = np.flatnonzero(bin_idx == b)
        if members.size == 0:
            continue
        # sorted within the bin so the woven pattern follows x, not input order
        members = members[np.argsort(values[members], kind="stable")]
        spread = 2 * _van_der_corput(members.size, base) - 1  # [0,1) -> [-1,1)
        offsets[members] = spread * halfwidth * (counts[b] / busiest)

    return offsets


def _gauge_theta(years, power=_GAUGE_POWER):
    """Years -> angle on the gauge arc, compressed by (years/max)**power.

    A power keeps 0 years on the arc's end point where a plain log could not
    (it would run off to minus infinity), and unlike a log it can be dialled:
    see _GAUGE_POWER for what each setting costs the short end. Values past
    _GAUGE_MAX_YEARS are clipped onto the arc's start rather than wrapping
    around past it.
    """
    years = np.clip(np.asarray(years, dtype=float), 0, _GAUGE_MAX_YEARS)
    u = (years / _GAUGE_MAX_YEARS) ** power
    return _GAUGE_THETA_LO + u * (_GAUGE_THETA_HI - _GAUGE_THETA_LO)


def _gauge_background(ax, power=_GAUGE_POWER):
    """Lay the green -> red ramp under the arc: a faint wash across the whole
    wedge, plus a saturated rim arc just outside the grade rings.

    Drawn as one quadmesh per layer rather than a stack of wedges, so the
    transition is genuinely continuous and there are no seams between bands.
    The ramp runs out at _GAUGE_GREEN_FROM_YEARS instead of at the end of the
    arc, so the stretch beyond it is one flat green rather than a gradient.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap  = LinearSegmentedColormap.from_list("gauge", _GAUGE_BG_COLORS)
    theta = np.linspace(_GAUGE_THETA_LO, _GAUGE_THETA_HI, 257)

    def _fraction(angle):
        """Angle -> position along the arc: 0 at its end (0 years), 1 at its
        start (_GAUGE_MAX_YEARS)."""
        return (angle - _GAUGE_THETA_LO) / (_GAUGE_THETA_HI - _GAUGE_THETA_LO)

    # cell centres, flipped because the ramp is written long-survival-first,
    # and rescaled so the green end is hit at _GAUGE_GREEN_FROM_YEARS
    centres = theta[:-1] + np.diff(theta) / 2
    green   = _fraction(_gauge_theta(_GAUGE_GREEN_FROM_YEARS, power))
    shade   = 1 - np.clip(_fraction(centres) / green, 0, 1)

    for r0, r1, alpha in ((0.0, _GAUGE_RIM_LO, _GAUGE_BG_ALPHA),
                          (_GAUGE_RIM_LO, _GAUGE_RIM_HI, 1.0)):
        if alpha <= 0:
            continue
        theta_grid, r_grid = np.meshgrid(theta, [r0, r1])
        ax.pcolormesh(theta_grid, r_grid, shade[None, :], cmap=cmap,
                      vmin=0, vmax=1, alpha=alpha, shading="flat",
                      edgecolors="none", linewidth=0, zorder=0,
                      rasterized=True)


def _gauge_hub_logo(ax, radius=_GAUGE_LOGO_R):
    """Stamp assets/logo-center2.png onto the gauge's hub, sized to *radius* in
    the axes' own r units.

    An AnnotationBbox rather than a nested axes: it is an artist of this axes,
    so it can be slotted into the same z-stack (over the needle bases, see
    _GAUGE_HUB_Z), which a child axes could never be -- those are drawn either
    wholly above or wholly below their parent.

    Silently does nothing when the file is absent, like _add_logo.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    # deferred import: utils.py is loaded (via database.py) before
    # __init__.py finishes defining its module-level constants
    from . import ASSETS_PATH

    logo_path = ASSETS_PATH / "logo-center2.png"
    if not logo_path.is_file():
        return

    img = plt.imread(logo_path)

    # r units -> printed inches, asked of the axes itself: the wedge is scaled
    # to fit its cell, so nothing outside knows how big one r unit ended up
    centre    = ax.transData.transform((0.0, 0.0))
    edge      = ax.transData.transform((0.0, radius))
    radius_in = float(np.hypot(*(edge - centre))) / ax.figure.dpi
    if not np.isfinite(radius_in) or radius_in <= 0:
        return

    # OffsetImage measures in points, and corrects for the renderer's dpi
    # itself, so this stays the same printed size at _PDF_DPI
    zoom = 2 * radius_in * 72 / img.shape[1]

    ax.add_artist(AnnotationBbox(
        OffsetImage(img, zoom=zoom, interpolation="antialiased"),
        (0.0, 0.0), xycoords="data", frameon=False, pad=0.0,
        annotation_clip=False, zorder=_GAUGE_HUB_Z))


def _plot_gauge(ax, grade_df, sample_median, sample_bounds=(np.nan, np.nan),
                power=_GAUGE_POWER, title=None):
    """Circos-style variant of the median strip: time runs along a 225 degree
    arc instead of along x.

    Reference samples are dots at their median survival's angle, spread out
    radially by the same quasirandom scheme the strips use. The predicted
    sample becomes a single clock hand from the hub at the ensemble's median
    (*sample_median*, in days), with the inter-model variance behind it as a
    wedge running between *sample_bounds* -- the two days the band crosses 0.5,
    the same pair the strip draws as a whisker.

    Every WHO grade gets its own concentric ring; *power* only changes how time
    maps onto the arc, not the radial layout.

    *title* is drawn inside the axes, over the top of the arc, rather than being
    handed to ax.set_title -- see _GAUGE_TITLE_R for why.
    """
    import matplotlib.lines as mlines

    g, present = _grade_rows(grade_df)

    ax.set_thetamin(np.rad2deg(_GAUGE_THETA_LO))
    ax.set_thetamax(np.rad2deg(_GAUGE_THETA_HI))
    ax.set_ylim(0, _GAUGE_RIM_HI)  # the rim arc is the outermost thing drawn
    _gauge_background(ax, power)
    _gauge_hub_logo(ax)

    n = len(present)
    if n > 1:
        # the outermost ring keeps half a spacing clear of the rim arc, the
        # same boundary the rings hold between themselves. Solving
        #   _GAUGE_RIM_LO - hi == (hi - _GAUGE_RING_LO) / (2 * (n - 1))
        # for hi is where the expression below comes from
        ring_hi = (2 * (n - 1) * _GAUGE_RIM_LO + _GAUGE_RING_LO) / (2 * n - 1)
        radii   = np.linspace(_GAUGE_RING_LO, ring_hi, n)
        spacing = radii[1] - radii[0]
        # being under half a spacing is also what keeps the outer dots off the rim
        halfwidth = min(spacing * _GAUGE_RING_BAND_FRAC, _GAUGE_RING_HW_MAX)
    else:
        # a lone ring sits midway between the inner end and the rim
        radii     = np.full(max(n, 1), (_GAUGE_RING_LO + _GAUGE_RIM_LO) / 2)
        spacing   = _GAUGE_RIM_LO - _GAUGE_RING_LO
        halfwidth = _GAUGE_RING_HW_MAX

    for i, grade in enumerate(present):
        years = (pd.to_numeric(g.loc[g["grade"] == grade, "median_survival"],
                               errors="coerce").dropna().to_numpy() / 365.25)
        if years.size == 0:
            continue
        theta = _gauge_theta(years, power)
        # spread against theta, not against the years: what has to be pulled
        # apart is what overlaps on the arc, and the scale is far from uniform
        r = radii[i] + _quasirandom_offsets(theta, halfwidth=halfwidth)
        # less transparent than the same dots in the strips: they have to hold
        # their own hue against the tinted background
        ax.scatter(theta, r, s=_STRIP_DOT_SIZE, alpha=0.75, edgecolors="none",
                   color=_GRADE_COLORS[grade], zorder=2)

    # where the ramp runs out: a divider marks the flat green beyond it as a
    # deliberate zone rather than a rendering artefact. Deliberately not a break
    # in the arc -- the angular scale does run on, it is only the colour that
    # stops discriminating
    theta_green = _gauge_theta(_GAUGE_GREEN_FROM_YEARS, power)
    ax.plot([theta_green, theta_green], [0, _GAUGE_RIM_HI], lw=0.5,
            color=_HEADER_DARK, alpha=0.35, zorder=1)

    # the hands run out to the boundary just inside the Grade 2/II ring (with
    # that grade absent, inside the second ring). The rings sit on a linspace,
    # so that boundary is half a spacing in from the ring itself -- the same
    # line that separates 2 from 3 and 3 from 4, rather than the edge of a
    # cloud, whose width is capped and so does not track the spacing
    if present:
        stop = present.index("Grade 2/II") if "Grade 2/II" in present \
            else min(1, len(present) - 1)
        face_r = radii[stop] - spacing / 2
    else:
        face_r = _GAUGE_NEEDLE_R_FALLBACK

    # a hairline closing the dial off at that radius: it turns the needles'
    # stopping point into a bounded face instead of an arbitrary cut-off
    theta_face = np.linspace(_GAUGE_THETA_LO, _GAUGE_THETA_HI, 200)
    ax.plot(theta_face, np.full_like(theta_face, face_r), lw=0.5,
            color=_HEADER_DARK, alpha=0.5, zorder=1.2)

    # the tips stop just short of that line rather than butting up against it
    needle_r = face_r - _GAUGE_NEEDLE_GAP

    # no median when the ensemble curve never crosses 0.5; the dial then shows
    # the reference clouds alone rather than a hand pointing at nothing
    if np.isfinite(sample_median):
        theta = _gauge_theta(sample_median / 365.25, power)

        lo_days, hi_days = sample_bounds
        if np.isfinite(lo_days) and np.isfinite(hi_days):
            # the same inter-model variance the curve panel shades. Swept on
            # the arc rather than drawn as two extra hands: it is one reading
            # with a width, not three predictions
            wedge = np.linspace(_gauge_theta(lo_days / 365.25, power),
                                _gauge_theta(hi_days / 365.25, power), 64)
            ax.fill_between(wedge, _GAUGE_HUB_R, needle_r,
                            color=_ENSEMBLE_COLOR, alpha=_ENSEMBLE_BAND_ALPHA,
                            lw=0, zorder=_GAUGE_NEEDLE_Z - 0.5)

        # one arrow patch rather than a line plus a rotated marker: the head is
        # part of the same object, so it sits exactly on the needle's axis and
        # tapers into it without the angle having to be worked out by hand
        ax.annotate("", xy=(theta, needle_r),
                    xytext=(theta, _GAUGE_HUB_R),
                    xycoords="data", textcoords="data", annotation_clip=False,
                    zorder=_GAUGE_NEEDLE_Z,
                    arrowprops=dict(arrowstyle=_GAUGE_ARROWSTYLE,
                                    color=_ENSEMBLE_COLOR, lw=_GAUGE_ARROW_LW,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=_GAUGE_HEAD_SCALE))
    # the hands' hub, deliberately just under the logo's zorder: it is what
    # carries the centre when logo-center2.png is missing, and stays hidden
    # behind it when it is not
    ax.scatter([0], [0], s=8, color=_HEADER_DARK, zorder=_GAUGE_HUB_Z - 0.1)

    ticks = (_GAUGE_TICKS_LINEAR if power >= 1 else _GAUGE_TICKS_COMPRESSED)
    ax.set_xticks(_gauge_theta(ticks, power))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([])  # radius is a category here, not a quantity
    ax.tick_params(pad=1)
    ax.grid(True, lw=0.4, alpha=0.25)
    ax.spines["polar"].set_linewidth(0.5)

    # no axis title of its own: the tick labels are years and the panel title
    # says so, while a polar xlabel hangs under the bounding box -- which here
    # is the empty wedge straight down, where the grade legend already sits
    if title:
        ax.annotate(title, xy=(np.pi / 2, _GAUGE_TITLE_R),
                    xytext=(0, _GAUGE_TITLE_PAD_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                    annotation_clip=False)

    # the arc deliberately does not close: the open wedge, now centred straight
    # down, is where the legend goes, so it costs no space of its own
    handles = [mlines.Line2D([], [], marker="o", linestyle="none", markersize=3,
                             color=_GRADE_COLORS[gr], label=gr)
               for gr in present]
    if handles:
        ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.16),
                  fontsize=5, handletextpad=0.4, borderpad=0.0,
                  labelspacing=0.25)


def _grade_rows(grade_df):
    """Reference table with the WHO grades folded onto their combined labels,
    plus the grades that actually occur, in fixed severity order (lowest
    first). Shared by every panel that splits the references out by grade."""
    g = grade_df.copy()
    g["grade"] = g["WHO Grade"].map(_GRADE_LABELS)

    present = [gr for gr in _GRADE_ORDER if gr in set(g["grade"].dropna())]
    return g, present


def _sample_median_expected(df, raw_cols):
    """Median and expected survival (in days) of the predicted sample, one
    value per Raw model curve. NaN where a curve never crosses 0.5."""
    from .notebook_functions import _median_expected_survival

    times       = df["days"].to_numpy()
    surv_matrix = df[raw_cols].to_numpy().T  # one row per Raw model curve
    median, expected = _median_expected_survival(surv_matrix, times)
    return {"median_survival": median, "expected_survival": expected}


def _ensemble_curve(df, cols):
    """Equal-weight mean over the model curves in *cols*, plus their standard
    deviation, both per time point.

    Averaging happens on the survival probabilities themselves, never on the
    risk scores: a Cox partial hazard, an AFT -predict_median and DeepHit's
    -rmst live on incompatible scales and their mean carries no meaning. That
    is the same rule notebook_functions.run_ensemble__time_to_event scores in
    cross-validation, so the IBS reported there is a statement about the very
    curve drawn here -- with the caveat that it was measured on out-of-fold
    predictions of the same member set at equal weight.

    The spread is inter-model variance: how far the members disagree about this
    one sample. It is not a confidence interval -- nothing in it accounts for
    sampling error, and members that are wrong together give a narrow band.
    """
    curves = df[cols].to_numpy(dtype=float)
    return curves.mean(axis=1), curves.std(axis=1)


def _sample_ensemble_median(df, cols):
    """Median survival (days) read off the ensemble curve, plus the two days its
    band crosses 0.5 -- ``(median, lower, upper)``, all three NaN where the curve
    they come from never reaches 0.5.

    The bounds are the 0.5 crossings of ``mean - sd`` and ``mean + sd``, which is
    the same band the curve panel shades. So the whisker in the strip and the
    wedge on the gauge are literally the two points where that band cuts the 0.5
    line in the panel above them, and a reader can check one against the other.

    Deliberately not the standard deviation of the members' own medians, which
    was what this drew first: that is a horizontal spread while the band is a
    vertical one, and the two only coincide when the members are pure shifts of
    each other. It also silently dropped every member whose curve never reaches
    0.5 -- those have no median, yet they do sit in the band -- and being
    symmetric it did not even centre on its own skewed distribution.
    """
    from .notebook_functions import _median_expected_survival

    mean, sd = _ensemble_curve(df, cols)
    times    = df["days"].to_numpy()

    # a higher curve crosses 0.5 later, so mean + sd is the upper (more
    # optimistic) bound and mean - sd the lower one
    stack     = np.vstack([mean, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1)])
    median, _ = _median_expected_survival(stack, times)
    return float(median[0]), float(median[1]), float(median[2])


def _median_label(median_days):
    """The ensemble's median survival as a panel title reads it.

    'not reached' rather than a blank or an infinity where the curve never gets
    to 0.5: within the follow-up the report covers, more than half the
    probability mass is still alive, and that is a finding rather than a
    missing number. Shared by the strip and both gauges so the three panels
    quote one figure in one wording.
    """
    if not np.isfinite(median_days):
        return "not reached"
    return f"{median_days / 365.25:.1f} years"


def _plot_grade_strip(fig, gs, row, grade_df, df, raw_cols, share_ax,
                      col_span=0):
    """Draw the thin median-survival strip plot, one dot per reference sample.
    WHO grade is split out as a categorical factor: each grade gets its own
    y-row (with a quasirandom spread within the row, see _quasirandom_offsets)
    so the point clouds no longer overlap. The top y-row holds the predicted
    sample itself: a single marker at the ensemble's median, with the members'
    own spread as a whisker through it. The strip sits in *row* and shares the
    Raw x-axis."""
    g, present = _grade_rows(grade_df)
    sample_row = 0  # the sample gets its own (taller) row above the grades
    positions = {gr: _STRIP_SAMPLE_ROW_H + i for i, gr in enumerate(present)}

    col = "median_survival"

    ax = fig.add_subplot(gs[row, col_span], sharex=share_ax)

    for grade in present:
        vals = pd.to_numeric(g.loc[g["grade"] == grade, col],
                             errors="coerce").dropna() / 365.25
        if vals.empty:
            continue
        y = positions[grade] + _quasirandom_offsets(vals.to_numpy())
        ax.scatter(vals, y, s=_STRIP_DOT_SIZE, alpha=0.55,
                   edgecolors="none", color=_GRADE_COLORS[grade])

    median_days, lo_days, hi_days = _sample_ensemble_median(df, raw_cols)
    sample_years = median_days / 365.25

    # no median at all when the ensemble curve never crosses 0.5; the row then
    # stays empty rather than the panel going missing, since the grade clouds
    # underneath are still worth reading
    if np.isfinite(sample_years):
        # the two years the band cuts the 0.5 line in the panel above. A bound
        # is missing when that edge of the band never gets there -- the upper
        # one especially, on a sample the models put well past the follow-up --
        # and the whisker is then left off rather than invented
        if np.isfinite(lo_days) and np.isfinite(hi_days):
            ax.plot([lo_days / 365.25, hi_days / 365.25],
                    [sample_row, sample_row], lw=1.0, alpha=0.55,
                    color=_ENSEMBLE_COLOR, solid_capstyle="butt", zorder=2)
        ax.scatter(sample_years, sample_row, marker="D", s=18,
                   color=_ENSEMBLE_COLOR, edgecolors="none", zorder=3)
    # keeps the sample row readable as a separate block from the grades
    ax.axhline(sample_row + _STRIP_SAMPLE_ROW_H / 2,
               color="black", lw=0.5, alpha=0.3)

    ax.set_ylim(-_STRIP_SAMPLE_HW - 0.15,
                _STRIP_SAMPLE_ROW_H + max(len(present) - 1, 0) + 0.5)
    ax.invert_yaxis()  # sample on top, then the grades in severity order
    ax.set_yticks([sample_row] + [positions[gr] for gr in present])
    ax.set_yticklabels(["Sample"] + present)
    ax.grid(axis="y", visible=False)
    # sharex hands the locators over from the curve panel above, but how a tick
    # is drawn is per axes, so the minor marks have to be sized here too
    ax.tick_params(axis="x", which="minor", length=_CURVE_MINOR_TICK_PT,
                   width=0.4, labelbottom=False)
    ax.set_title(f"Median predicted Overall Survival: {_median_label(median_days)}",
                 loc="left", fontsize=7)
    ax.set_xlabel("Years [sample taken to predicted event]")

    return ax


def _plot_subtype_bars(ax, subtype_prediction, top_n=_SUBTYPE_TOP_N):
    """Draw the tumor subtype probabilities as sorted vertical bars.

    Returns the label of the called class, so the caller can name it in the
    panel title without repeating the argmax.

    *subtype_prediction* maps class label -> probability. The bars stand up so
    the width goes to the classes and the height to the probability, which is
    also the axis the rotated class names are read against.
    """
    ranked = sorted(subtype_prediction.items(), key=lambda kv: kv[1], reverse=True)
    lead, tail = ranked[:top_n], ranked[top_n:]

    labels = [_shorten_label(name) for name, _ in lead]
    probs  = [float(prob) for _, prob in lead]
    colors = [_SUBTYPE_COLORS[0]] + [_SUBTYPE_COLORS[1]] * (len(lead) - 1)
    if tail:
        labels.append(f"other ({len(tail)})")
        probs.append(float(sum(prob for _, prob in tail)))
        colors.append(_SUBTYPE_OTHER_COLOR)

    xs = np.arange(len(labels))
    ax.bar(xs, probs, width=_SUBTYPE_BAR_W, color=colors, zorder=2)

    for x, label, prob in zip(xs, labels, probs):
        # the names are drawn into the panel rather than hung under it as tick
        # labels: on their side they run half an inch or more, which is more
        # than the gap to the row below and would collide with it. Inside the
        # panel they cost no layout at all -- a short bar's name simply carries
        # on over the background above it
        name_frac = _rotated_label_frac(ax, label)
        inside    = prob > name_frac
        ax.text(x, _SUBTYPE_LABEL_Y0, label, rotation=90, va="bottom",
                ha="center", fontsize=5.5, zorder=3,
                color="#FFFFFF" if inside else _HEADER_DARK)

        # all on one line, just over the top of the scale, rather than each
        # riding its own bar: as a row they can be read straight across and
        # compared, where a staircase of values has to be traced one by one.
        # Nothing can reach up into that line -- names are clipped to
        # _SUBTYPE_LABEL_MAX, which stands well short of 1.0 in this panel --
        # so the row never has to dodge a bar or a name
        ax.text(x, _SUBTYPE_VALUE_Y, f"{prob:.2f}", va="bottom", ha="center",
                fontsize=5.5, color=_HEADER_DARK, zorder=3)

    ax.set_xticks([])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    # the extra room at the top is for the value labels; the ladder stays on
    # 0..1, which is what the bars are actually measured against
    ax.set_ylim(0, 1 + _SUBTYPE_HEADROOM)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return ranked[0][0]


def _rotated_label_frac(ax, label, fontsize=5.5):
    """Share of the y axis a name takes up once it is turned on its side.

    What is a text width lying down is a text height standing up, which is why
    this is measured against the panel's printed height rather than its width.
    """
    text_in   = len(label) * fontsize * _CHAR_W_FACTOR / 72
    axes_h_in = ax.get_position().height * _A4_H_IN
    return text_in / axes_h_in if axes_h_in else 1.0


def _shorten_label(name):
    """Clip an over-long class name, the way the CLI does for its own table."""
    if len(name) <= _SUBTYPE_LABEL_MAX:
        return name
    return name[:_SUBTYPE_LABEL_MAX - 2] + ".."


def _plot_sex_bar(ax, sex_prediction):
    """Draw the predicted sex as one probability running from 0 to 1.

    Returns the label of the called class, so the caller can name it in the
    panel title without repeating the argmax.

    *sex_prediction* maps class label -> probability, straight off the
    LabelEncoder/predict_proba pair the classifier was exported with. In the
    usual two-class case the axis runs from one class to the other, so an
    ambiguous call slides towards the middle instead of being flattened into a
    hard label -- see _SEX_UNCERTAIN_LO/HI. Any other number of classes falls
    back to one bar per class, still on the same 0..1 scale.
    """
    from matplotlib.colors import LinearSegmentedColormap

    labels = list(sex_prediction)
    probs  = [float(sex_prediction[k]) for k in labels]

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    if len(labels) != 2:
        rows = np.arange(len(labels))
        ax.barh(rows, probs, height=0.6, color=_SEX_COLORS[0])
        ax.set_yticks(rows)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        return labels[int(np.argmax(probs))]

    left_label, right_label = labels
    p = probs[1]   # probability of the right-hand class; the left one is 1 - p

    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # the ramp is a reading aid, not data: it whitens through the middle so the
    # marker stays legible exactly where the call is least certain
    ramp = LinearSegmentedColormap.from_list(
        "sex", [_SEX_COLORS[0], "#FFFFFF", _SEX_COLORS[1]])
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(0, 1, 0, 1),
              aspect="auto", cmap=ramp, alpha=_SEX_RAMP_ALPHA, zorder=0)

    ax.axvspan(_SEX_UNCERTAIN_LO, _SEX_UNCERTAIN_HI,
               facecolor="#FFFFFF", alpha=0.45, zorder=1)
    for edge in (_SEX_UNCERTAIN_LO, _SEX_UNCERTAIN_HI):
        ax.axvline(edge, color=_HEADER_GRAY, lw=0.5, ls=(0, (2, 2)), zorder=2)

    ax.plot([p, p], [0, 1], color=_HEADER_DARK, lw=1.0, zorder=3)
    # clip_on=False: the head sits on the top edge of the axes, so half of it
    # would be cut away by the axes box it marks
    ax.plot([p], [1.0], marker="v", ms=4, color=_HEADER_DARK,
            clip_on=False, zorder=4)

    # outside the track rather than in it: the track is only a few points high
    # and the probability readout already claims the room inside it. Initials,
    # because the column is narrow and both the readout and the panel title
    # name the class in full
    ax.text(-0.02, 0.5, left_label[:1].upper(),  ha="right", va="center",
            fontsize=6, color=_HEADER_DARK, zorder=3)
    ax.text(1.02, 0.5, right_label[:1].upper(), ha="left",  va="center",
            fontsize=6, color=_HEADER_DARK, zorder=3)

    # the readout names the class that was actually called, never the other
    # one: 'p(Male) = 0.028' asks the reader to do the subtraction, and reads
    # as a contradiction of the panel title one line above it
    called_label, called_p = (right_label, p) if p >= 0.5 else (left_label, 1 - p)

    # written back towards the middle, away from whichever end the marker sits
    # against, so it never runs off the page or over a class label
    ha, dx = ("left", 0.012) if p < 0.5 else ("right", -0.012)
    ax.text(p + dx, 0.5, f"p({called_label}) = {called_p:.3f}", ha=ha, va="center",
            fontsize=6, color=_HEADER_DARK, zorder=4)

    return called_label


def _page_grid(fig, row_heights, n_cols=_PAGE_COLS):
    """GridSpec over *row_heights* (in inches), hung under the header band.

    Rows keep _ROW_GAP_IN between them wherever the page can afford it, and the
    gap is tightened evenly when it cannot -- a page that quietly lays its last
    row out past the paper edge is worse than one spaced a little tighter.
    Returns the grid and the printed height of the block, which callers need to
    place anything that is not a grid cell.
    """
    from matplotlib.gridspec import GridSpec

    n_gaps     = max(len(row_heights) - 1, 1)
    spare_in   = (_A4_H_IN - _MARGIN_T_IN - _PAGE_BOTTOM_MIN_IN
                  - sum(row_heights))
    gap_in     = max(min(_ROW_GAP_IN, spare_in / n_gaps), 0.0)
    block_h_in = sum(row_heights) + gap_in * (len(row_heights) - 1)

    # rows are sized in inches (see _CURVE_H_IN etc.) and then converted to the
    # figure fractions GridSpec wants; hspace is a fraction of the *mean* row
    # height, which is why the gap can't just be handed over as-is
    gs = GridSpec(len(row_heights), n_cols, height_ratios=row_heights,
                  hspace=gap_in / (sum(row_heights) / len(row_heights)),
                  wspace=0.15,
                  left=_MARGIN_L_IN / _A4_W_IN,
                  right=1 - _MARGIN_R_IN / _A4_W_IN,
                  top=1 - _MARGIN_T_IN / _A4_H_IN,
                  bottom=1 - (_MARGIN_T_IN + block_h_in) / _A4_H_IN,
                  figure=fig)
    return gs, block_h_in


def _new_page(sentrix_id):
    """A blank A4 carrying the header band, which every page repeats: the
    report is printed and handed out, and a loose second sheet has to be able
    to say which sample it belongs to."""
    fig = plt.figure(figsize=(_A4_W_IN, _A4_H_IN))
    logo_w_in = _add_logo(fig, pad_in=_PAGE_MARGIN_IN, loc="left",
                          height_in=_LOGO_REPORT_H_IN)
    _add_report_header(fig, sentrix_id, logo_w_in)
    return fig


def _half_crossing(years, curve):
    """Year at which a non-increasing *curve* passes 0.5, NaN if it never does.

    Interpolated between the two sampled points either side, so the answer does
    not sit on the drawing grid -- see _CURVE_DRAW_POINTS, which is coarse
    enough for that to be visible. Read off the drawn curve rather than off the
    daily frame the strip uses: this marks where the line on *this* panel meets
    0.5, and it has to land on it.
    """
    c = np.asarray(curve, dtype=float)
    if c.min() > 0.5 or c.max() < 0.5:
        return np.nan
    # np.interp wants an increasing x, which a survival curve is once reversed
    return float(np.interp(0.5, c[::-1], np.asarray(years, dtype=float)[::-1]))


def _plot_curves(ax, years, df_sampled, cols, title, legend=True, xlabel=True):
    """One survival-curve panel: the equal-weight ensemble over *cols*, with
    the inter-model variance as a band around it.

    One curve per compiled model is a legend the reader has to decode before
    the panel says anything, and the members are not a dozen competing answers
    -- the ensemble is the prediction and their spread is what it is worth. See
    _ensemble_curve for why the averaging is done on the curves.
    """
    if not cols:
        return

    mean, sd = _ensemble_curve(df_sampled, cols)

    # clipped because a mean near either end plus a standard deviation leaves
    # the probability scale, which would read as the band, not the arithmetic
    ax.fill_between(years, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                    color=_ENSEMBLE_COLOR, alpha=_ENSEMBLE_BAND_ALPHA, lw=0,
                    label=f"Inter-model variance (\u00b11 SD, n={len(cols)})")
    ax.plot(years, mean, lw=1.4, color=_ENSEMBLE_COLOR, label="Ensemble mean")

    # the median marker, drawn only where the band actually crosses 0.5 rather
    # than clear across the panel: its two ends are then the same pair the strip
    # below draws as a whisker and the gauge as a wedge, so the three panels can
    # be checked against each other. Running the full width would say nothing
    # the y ladder does not already say. Left off entirely when an edge of the
    # band never reaches 0.5 -- the strip omits its whisker then too
    left  = _half_crossing(years, np.clip(mean - sd, 0, 1))
    right = _half_crossing(years, np.clip(mean + sd, 0, 1))
    if np.isfinite(left) and np.isfinite(right):
        ax.plot([left, right], [0.5, 0.5], lw=0.4, color=_HEADER_DARK,
                alpha=0.45, solid_capstyle="butt", zorder=1)

    # a fixed five year ladder rather than whatever the locator picks: the
    # panel is read off against the gauge and the strip, and a reader comparing
    # three panels should not have to check three sets of ticks first
    ax.set_xticks(_CURVE_TICK_YEARS)
    ax.set_xticks(_CURVE_MINOR_YEARS, minor=True)
    ax.tick_params(axis="x", which="minor", length=_CURVE_MINOR_TICK_PT,
                   width=0.4, labelbottom=False)

    # the axis title goes only on the bottom panel of a stack, which shares this
    # x axis -- but the ticks are labelled on both, so the curve can be read at
    # a year without tracing down into the panel below it
    if xlabel:
        ax.set_xlabel("Years [sample taken to predicted event]")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(left=0)
    ax.set_title(title, loc="left", fontsize=7)
    if legend:
        ax.legend(loc="upper right")


def _plot_probability_row(fig, gs, row, subtype_prediction, sex_prediction):
    """The subtype bars and the sex strip, side by side on one 0..1 scale."""
    if subtype_prediction:
        ax_subtype = fig.add_subplot(gs[row, _PANEL_COLS])
        called = _plot_subtype_bars(ax_subtype, subtype_prediction)
        ax_subtype.set_title(f"Predicted tumor subtype: {called}",
                             loc="left", fontsize=7)

    if sex_prediction:
        # the strip keeps its printed height instead of filling the cell, so
        # it is placed by hand inside the cell the grid marks out for it
        box    = gs[row, _SIDE_COL].get_position(fig)
        height = _SEX_H_IN / _A4_H_IN
        ax_sex = fig.add_axes([box.x0, box.y0 + (box.height - height) / 2,
                               box.width * _SEX_WIDTH_FRAC, height])
        called = _plot_sex_bar(ax_sex, sex_prediction)
        ax_sex.set_title(f"Predicted sex: {called}", loc="left", fontsize=7)


def plot_survival(df, pdf_out, sentrix_id, grade_only_results=None,
                  cnv_bins_file=None, cnv_detail_file=None,
                  sex_prediction=None, subtype_prediction=None):
    """Render the two-page A4 survival report.

    Page one carries what is read first: the CNV track, the predicted subtype
    and sex probabilities, the raw survival curve and the compressed gauge.
    Page two carries the same material a second way -- the calibrated curve
    and the linear-years gauge -- which is worth a page of its own rather than
    a cramped half column beside the raw one.

    Every survival panel shows one prediction, the equal-weight ensemble over
    the models in *df*, with the members' spread as a band (curve), a whisker
    (strip) and a wedge (gauge). See _ensemble_curve for the averaging rule and
    for what that spread is and is not.

    *cnv_bins_file* is optional: when it is given and the file exists, the
    mepylome CNV track is drawn as the top panel, above the survival curves.
    A missing (or never generated) file just drops the row -- the CNV step of a
    run can fail or be skipped without taking the survival report with it.

    *sex_prediction* and *subtype_prediction* are optional as well: each is a
    class label -> probability mapping, and leaving one out drops its panel the
    same way. With neither, the whole probability row goes.
    """
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        # the header's inline bold run goes through mathtext; point it at the
        # same sans-serif family so it doesn't fall back to DejaVu mid-sentence.
        # 'custom' resolves every mathtext family, not just the ones actually
        # used, so cal/sf/tt have to be answered too -- cal defaults to the
        # generic 'cursive', which most Linux boxes cannot satisfy and which
        # then warns on every render
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic",
        "mathtext.sf": "sans",
        "mathtext.tt": "monospace",
        "axes.prop_cycle": plt.cycler("color", _CURVE_COLORS),
    })
    idx = np.linspace(0, len(df) - 1, _CURVE_DRAW_POINTS, dtype=int)
    df_sampled = df.iloc[idx].reset_index(drop=True)

    years    = df_sampled["days"] / 365.25
    all_cols = [c for c in df_sampled.columns if c != "days"]

    raw_cols = [c for c in all_cols if "__calibrated" not in c]
    cal_cols = [c for c in all_cols if "__calibrated"     in c]

    has_grades = grade_only_results is not None and not grade_only_results.empty
    has_cnv    = cnv_bins_file is not None and Path(cnv_bins_file).is_file()
    has_probs  = bool(sex_prediction) or bool(subtype_prediction)

    median_days, lo_days, hi_days = (_sample_ensemble_median(df, raw_cols)
                                     if has_grades else (np.nan, np.nan, np.nan))

    with PdfPages(pdf_out) as pdf:
        # ---- page one -----------------------------------------------------
        heights = {"cnv":   _CNV_H_IN,  "prob":  _PROB_H_IN,
                   "curve": _CURVE_H_IN, "strip": _STRIP_H_IN,
                   "gauge": _GAUGE_H_IN}
        order = ((["cnv"] if has_cnv else [])
                 + (["prob"] if has_probs else [])
                 + ["curve"]
                 + (["strip", "gauge"] if has_grades else []))
        rows = {name: i for i, name in enumerate(order)}

        fig = _new_page(sentrix_id)
        gs, _ = _page_grid(fig, [heights[name] for name in order])

        if has_cnv:
            # spans both columns: the genome is one continuous axis, and 24
            # chromosomes need every inch of the page width they can get
            ax_cnv = fig.add_subplot(gs[rows["cnv"], :])
            _plot_cnv_track(ax_cnv, cnv_bins_file, cnv_detail_file)
            ax_cnv.set_title(_MEPYLOME_TITLE, loc="left", fontsize=7,
                             url=_MEPYLOME_URL)

        if has_probs:
            _plot_probability_row(fig, gs, rows["prob"],
                                  subtype_prediction, sex_prediction)

        # the curve panel and the strip share both their width and their x
        # axis, so the two read as one stack over the same years
        ax_raw = fig.add_subplot(gs[rows["curve"], :])
        _plot_curves(ax_raw, years, df_sampled, raw_cols,
                     f"Merged / integrated direct output on n={len(raw_cols)} models",
                     xlabel=not has_grades)

        if has_grades:
            ax_strip = _plot_grade_strip(fig, gs, rows["strip"],
                                         grade_only_results, df, raw_cols,
                                         ax_raw, col_span=slice(None))

            # the label carries the exponent itself, so the panel keeps saying
            # what it does when _GAUGE_POWER is retuned
            # a real superscript via mathtext: '**' is a Python-ism and '^'
            # means exclusive-or in about as many languages as it means a power
            ax_gauge = fig.add_subplot(gs[rows["gauge"], :], projection="polar")
            _plot_gauge(
                ax_gauge, grade_only_results, median_days, (lo_days, hi_days),
                power=_GAUGE_POWER,
                title=f"Median predicted Overall Survival: "
                      f"{_median_label(median_days)} "
                      f"(years$^{{{_GAUGE_POWER:g}}}$)")

        pdf.savefig(fig, dpi=_PDF_DPI)
        plt.close(fig)

        # ---- page two -----------------------------------------------------
        # the same two panels on the plain time scale. Nothing to show means no
        # second sheet at all, rather than a page carrying only its header
        order2 = ((["curve"] if cal_cols else [])
                  + (["gauge"] if has_grades else []))
        if not order2:
            return

        # both panels have the page to themselves here, so they are given more
        # of it than their page-one counterparts get: the dial in particular is
        # drawn as a circle the height of its row, so height is what sizes it
        heights2 = {"curve": _CURVE_H_IN * 1.5, "gauge": _GAUGE_H_IN * 1.5}
        rows2 = {name: i for i, name in enumerate(order2)}

        fig = _new_page(sentrix_id)
        gs2, _ = _page_grid(fig, [heights2[name] for name in order2])

        if cal_cols:
            ax_cal = fig.add_subplot(gs2[rows2["curve"], :])
            _plot_curves(ax_cal, years, df_sampled, cal_cols, "Calibrated")

        if has_grades:
            ax_gauge = fig.add_subplot(gs2[rows2["gauge"], :], projection="polar")
            _plot_gauge(
                ax_gauge, grade_only_results, median_days, (lo_days, hi_days),
                power=1.0,
                title=f"Median predicted Overall Survival: "
                      f"{_median_label(median_days)} (linear years)")

        pdf.savefig(fig, dpi=_PDF_DPI)
        plt.close(fig)


def export_survival(prediction_classes, fn=None):
    # deferred import: utils.py is loaded (via database.py) before
    # __init__.py finishes defining its module-level constants
    from . import DAYS_PER_YEAR, MAX_FOLLOW_UP_YEARS

    times = np.arange(1, round(DAYS_PER_YEAR * MAX_FOLLOW_UP_YEARS) + 1)

    #for k in prediction_classes.keys():
        #print(k)
        #print(prediction_classes[k].head(5))

    # clipped to each curve's own domain rather than queried straight: a
    # StepFunction raises outside it, and a model whose horizon stops short of
    # MAX_FOLLOW_UP_YEARS -- or a calibrator that rescaled the time axis -- would
    # otherwise take the whole report down. Held flat past the horizon, which is
    # the same thing export_classifier does for the reference cohort, so the
    # curve and the cloud treat their tails alike.
    from .custom_transformers import curve_on_grid

    df = pd.DataFrame(
        {id_: curve_on_grid(sf, times) for id_, sf in prediction_classes.items()},
    )
    df.insert(0, "days", times)

    if fn is not None:
        df.to_csv(fn, sep="\t", float_format="%.4f", index=False)

    return df


# Fixed karyotype order (hg38 Chromosome column, as written by mepylome_helpers)
# so chromosomes always lay out left-to-right in the standard order, not
# whatever order they happen to appear in the bins/detail CSV.
_CNV_CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# Diverging red-white-blue palette (loss - neutral - gain), matching the
# reference mepylome/CNV browser plots: RdBu(0)=dark red, RdBu(1)=dark blue.
_CNV_CMAP = plt.get_cmap("RdBu")

# credit for the CNV track, which is mepylome's work rather than ours. Spelled
# out in the title as well as attached as a link: the report is meant to be
# printed, and a hyperlink is worth nothing on paper
_MEPYLOME_URL   = "https://github.com/brj0/mepylome"
_MEPYLOME_TITLE = "Mepylome CNV  -  by https://github.com/brj0/mepylome"


def _cnv_chrom_offsets(bins):
    """Cumulative x-offset (bp) per chromosome, in karyotype order, so every
    chromosome can be laid out left-to-right on one shared x-axis."""
    sizes = bins.groupby("Chromosome")["End"].max()
    offsets = {}
    cursor = 0
    for chrom in _CNV_CHROM_ORDER:
        if chrom not in sizes.index:
            continue
        offsets[chrom] = cursor
        cursor += sizes[chrom]
    return offsets, cursor


def _cnv_var_to_alpha(var, min_alpha=0.15, max_alpha=0.9):
    """Alpha per bin/gene: higher Var (noisier estimate) -> more transparent.

    Var has no fixed scale, so it's normalized against its own 95th percentile
    (not an absolute cutoff) -- that way the blending adapts to whatever noise
    level this particular sample/array type happens to have.
    """
    var = pd.to_numeric(pd.Series(var), errors="coerce").fillna(0.0).to_numpy()
    positive = var[var > 0]
    scale = np.percentile(positive, 95) if positive.size else 0.0
    if scale <= 0:
        return np.full(len(var), max_alpha)
    norm = np.clip(var / scale, 0, 1)
    return max_alpha - norm * (max_alpha - min_alpha)


def _cnv_colors(median, var, vlim, min_alpha=0.15, max_alpha=0.9):
    """RGBA per bin/gene: hue+intensity from the (signed) value via the RdBu
    colormap (dark = far from zero, matching the reference plot), alpha from
    Var (noisier estimate -> more transparent) -- two independent channels.
    """
    median = pd.to_numeric(pd.Series(median), errors="coerce").fillna(0.0).to_numpy()
    norm = np.clip(median / vlim, -1, 1)
    rgba = _CNV_CMAP((norm + 1) / 2)
    rgba[:, 3] = _cnv_var_to_alpha(var, min_alpha, max_alpha)
    return rgba


def _plot_cnv_track(ax, bins_file, detail_file, gene_label_threshold=0.3,
                    max_gene_labels=25, plot_detail=True, detail_min_probes=10):
    """Draw the genome-wide CNV track onto *ax*; see plot_cnv for what the
    arguments mean. Split out so the standalone PDF and the panel embedded in
    the survival report are one and the same plot."""
    plot_detail = False

    bins = pd.read_csv(bins_file)
    detail = (pd.read_csv(detail_file) if plot_detail
              else pd.DataFrame(columns=["Chromosome", "Start", "End", "Median", "Var", "N_probes", "Name"]))

    offsets, genome_length = _cnv_chrom_offsets(bins)
    bins = bins[bins["Chromosome"].isin(offsets)].copy()
    bins["x_start"] = bins["Chromosome"].map(offsets) + bins["Start"]
    bins["x_end"] = bins["Chromosome"].map(offsets) + bins["End"]

    detail = detail[detail["Chromosome"].isin(offsets)].dropna(subset=["Median"]).copy()
    detail = detail[detail["N_probes"] > detail_min_probes]
    detail["x"] = detail["Chromosome"].map(offsets) + (detail["Start"] + detail["End"]) / 2

    chrom_order_present = [c for c in _CNV_CHROM_ORDER if c in offsets]
    chrom_sizes = bins.groupby("Chromosome")["End"].max()

    # Zebra background per chromosome so adjacent chromosomes stay visually separated
    for i, chrom in enumerate(chrom_order_present):
        if i % 2 == 1:
            ax.axvspan(offsets[chrom], offsets[chrom] + chrom_sizes[chrom],
                       color="black", alpha=0.04, lw=0)

    # Shared color scale for bins + genes, so both read against the same
    # reference; capped at a sane floor so a near-flat sample doesn't get
    # over-saturated by noise alone.
    vlim = max(bins["Median"].abs().quantile(0.98),
               detail["Median"].abs().quantile(0.98) if len(detail) else 0.0,
               0.3)

    # Each bin as a horizontal segment spanning Start-End (not just its midpoint),
    # faded out (lower alpha) the higher its Var -- noisier bins draw less attention.
    bin_rgba = _cnv_colors(bins["Median"], bins["Var"], vlim)
    bin_segments = [[(x0, y), (x1, y)] for x0, x1, y in
                    zip(bins["x_start"], bins["x_end"], bins["Median"])]
    ax.add_collection(LineCollection(bin_segments, colors=bin_rgba, linewidths=3.5))
    ax.autoscale_view()
    ax.axhline(0, color="black", lw=0.5)

    # Gene-level emphasis on top of the bin track
    if plot_detail:
        gene_rgba = _cnv_colors(detail["Median"], detail["Var"], vlim)
        ax.scatter(detail["x"], detail["Median"], s=5, c=gene_rgba,
                   edgecolors="none", zorder=3)

        labelled = (detail.reindex(detail["Median"].abs().sort_values(ascending=False).index)
                          .loc[lambda d: d["Median"].abs() >= gene_label_threshold]
                          .head(max_gene_labels))
        for _, row in labelled.iterrows():
            ax.annotate(row["Name"], (row["x"], row["Median"]), fontsize=5,
                        xytext=(0, 4), textcoords="offset points", ha="center")

    tick_pos = [offsets[c] + chrom_sizes[c] / 2 for c in chrom_order_present]
    tick_lbl = [c.removeprefix("chr") for c in chrom_order_present]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.set_xlim(0, genome_length)
    ax.set_ylim(*_CNV_YLIM)
    ax.set_ylabel("log2 ratio")


def plot_cnv(bins_file, detail_file, pdf_out, sample_id,
             gene_label_threshold=0.3, max_gene_labels=25, plot_detail=True,
             detail_min_probes=10):
    """Genome-wide CNV plot: per-bin log2 ratio (bins_file) across all
    chromosomes, with gene-level detail (detail_file) highlighted on top.

    Genes are only labelled with their name when |Median| >= gene_label_threshold
    (capped at max_gene_labels, most extreme first) -- detail_file has one row
    per gene (tens of thousands), so labelling all of them would make the plot
    unreadable. Genes with N_probes <= detail_min_probes are dropped entirely:
    their per-gene ratio rests on too few CpGs to be a meaningful estimate.

    `plot_detail=False` also skips reading `detail_file` entirely (not just
    the overlay), so callers that never generated one (e.g. it was skipped
    upstream because the probe manifest's genome build doesn't match
    mepylome's gene annotations -- see mepylome_idat_to_cnv_core) can pass a
    nonexistent path and still get the bin-track-only plot.
    """
    # deferred import: utils.py is loaded (via database.py) before
    # __init__.py finishes defining its module-level constants
    from . import __version__

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(11, 3.2))

    _plot_cnv_track(ax, bins_file, detail_file,
                    gene_label_threshold=gene_label_threshold,
                    max_gene_labels=max_gene_labels, plot_detail=plot_detail,
                    detail_min_probes=detail_min_probes)

    ax.set_title(f"Sentrix ID: {sample_id}", loc="left", fontsize=7)
    fig.suptitle(_MEPYLOME_TITLE, x=0.01, ha="left", fontsize=6, color="gray",
                 url=_MEPYLOME_URL)

    fig.text(0.99, 0.01, f"cognition v{__version__} by Dr. Y. Hoogstrate and Dr. R. Schoonhoven",
             ha="right", va="bottom", fontsize=6, color="gray")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    _add_logo(fig)
    plt.savefig(pdf_out, format="pdf")
    plt.close()
