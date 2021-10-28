import warnings
import math
import argparse
import random
import os
import sqlite3
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy import ndimage

warnings.filterwarnings("ignore")

width, height = 640, 480
linestyles = ["solid", "dashed", "dashdot", "dotted"]
markers = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
]

fontnames = [
    "Andale Mono",
    "Arial",
    "Arial Black",
    "Comic Sans MS",
    "Courier New",
    "Georgia",
    "Impact",
    "Times New Roman",
    "Trebuchet MS",
    "Verdana",
]
fontname = fontnames[0]

# init words list
conn = sqlite3.connect("./wnjpn.db")
jpn_words_list = conn.execute(
    "select * from word where lang = 'jpn' ORDER BY RANDOM();"
)
eng_words_list = conn.execute(
    "select * from word where lang = 'eng' ORDER BY RANDOM();"
)
jpn_words_list = [record[2] for record in jpn_words_list.fetchall()]
eng_words_list = [record[2] for record in eng_words_list.fetchall()]
words_list = [jpn_words_list, eng_words_list]
word_list = words_list[0]

ax = None
fig = None
nlines = None


def random_text(max_length=None):
    word = word_list[np.random.randint(0, len(word_list))]
    if max_length is None:
        return word
    while len(word) > max_length:
        word = word_list[np.random.randint(0, len(word_list))]
    return word


def random_color_table(n):
    scale = 8
    colors = [
        (r, g, b)
        for r in range(256 // scale)
        for g in range(256 // scale)
        for b in range(256 // scale)
    ]
    colors = colors if len(colors) > 0 else [(0, 0, 0)]
    threshold = 256 // n // scale
    table = []

    def f(x):
        for c in table:
            if (
                abs(x[0] - c[0]) <= threshold
                and abs(x[1] - c[1]) <= threshold
                and abs(x[2] - c[2]) <= threshold
            ):
                return False
        return True

    for _ in range(n):
        c = colors[np.random.randint(0, len(colors))] if len(colors) > 0 else (0, 0, 0)
        table.append(c)
        colors = list(filter(f, colors))
    table = [
        (c[0] * scale / 255, c[1] * scale / 255, c[2] * scale / 255) for c in table
    ]
    return table


def force_aspect(aspect=1):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) / aspect)


# Reference: https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
def inline_legend(colors=None):
    N = 32
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((nlines, N, N), dtype=np.float64)

    for l in range(nlines):
        # get xy data and scale it to the NxN squares
        xy = ax.lines[l].get_xydata()
        xy = (xy - [xmin, ymin]) / ([xmax - xmin, ymax - ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot
        mask = (xy[:, 0] >= 0) & (xy[:, 0] < N) & (xy[:, 1] >= 0) & (xy[:, 1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
    # don't use the borders
    ws[:, 0] = 0
    ws[:, N - 1] = 0
    ws[0, :] = 0
    ws[N - 1, :] = 0

    # blur the pop's
    for l in range(nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N / 5)

    label_size = np.random.randint(10, 20)
    inline_labels = []
    colored_inline = np.random.uniform() < 0.5
    for l in range(nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(nlines, dtype=np.float64)
        w[l] = 0.5

        # calculate a field
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)

        pos = np.argmax(p)  # note, argmax flattens the array first
        best_x, best_y = (pos / N, pos % N)
        x = xmin + (xmax - xmin) * best_x / N
        y = ymin + (ymax - ymin) * best_y / N

        label = random_text()
        if colored_inline:
            il = ax.text(
                x,
                y,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=label_size,
                fontname=fontname,
                color=colors[l],
            )
        else:
            il = ax.text(
                x,
                y,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=label_size,
                fontname=fontname,
            )
        inline_labels.append(il)
    return inline_labels


def get_bb_coordinate(e):
    e_bb = e.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
    bb = e_bb.x0, e_bb.y0, e_bb.x1, e_bb.y1
    bb = (int(bb[0]), int(height - bb[3]), int(bb[2]), int(height - bb[1]))
    return bb


def init_ax_fig():
    global ax, fig
    plt.style.use("default")
    plt.tight_layout()
    ax = plt.gca()
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(top=0.8)
    fig.subplots_adjust(right=0.8)
    return ax, fig


def init_axis_scale():
    x_int = np.random.uniform() < 0.5  # whether x values are integers or not
    y_int = np.random.uniform() < 0.5  # whether y values are integers or not
    x_zero_start = np.random.uniform() < 0.5
    y_zero_start = np.random.uniform() < 0.5
    x_start = (
        0 if x_zero_start else np.random.uniform() * (10 ** np.random.randint(0, 4))
    )
    y_start = (
        0 if y_zero_start else np.random.uniform() * (10 ** np.random.randint(0, 4))
    )

    num_format_comma = np.random.uniform() < 0.7
    if x_int:
        x_int = int(x_int)
        if num_format_comma:
            ax.get_xaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
            )
    if y_int:
        y_int = int(y_int)
        if num_format_comma:
            ax.get_yaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
            )

    x_interval = 10 ** (np.random.randint(0, 5) - 1)
    y_interval = 10 ** (np.random.randint(0, 5) - 1)
    x_ticks = np.random.randint(2, 10)
    y_ticks = np.random.randint(2, 8)

    return x_int, y_int, x_start, y_start, x_interval, y_interval, x_ticks, y_ticks


def random_line_charts(i, output_dir):
    global ax, fig, word_list, nlines
    ax, fig = init_ax_fig()

    monotone = np.random.uniform() < 0.5
    grayscale = np.random.uniform() < 0.3 and monotone
    nlines = (
        np.random.randint(2, 5)
        if monotone
        else min(8, math.floor(abs(np.random.normal(0, 3)) + 1))
    )

    (
        _x_int,
        _y_int,
        x_start,
        y_start,
        x_interval,
        y_interval,
        x_ticks,
        y_ticks,
    ) = init_axis_scale()

    align_xdata = np.random.uniform() < 0.6
    flat = np.random.uniform() < 0.05

    # eng_word = int(np.random.uniform()<0.8)
    eng_word = 1
    word_list = words_list[eng_word]

    smooth = np.random.uniform() < 0.6  # smooth lines or not
    add_marker = np.random.uniform() < 0.5 and align_xdata and not smooth
    add_grid = np.random.uniform() < 0.5
    tmp = np.random.uniform()
    grid_axis = "both" if tmp < 0.2 else "y" if tmp < 0.8 else "x"
    tmp = np.random.uniform()
    grid_which = "major" if tmp < 0.7 else "both"
    tmp = np.random.uniform()
    grid_linestyle = "-" if tmp < 0.5 else "--" if tmp < 0.75 else ":"
    grid_linewidth = np.random.uniform(1.0, 3.0)
    grid_alpha = np.random.uniform(0.3, 1.0)
    if add_grid:
        plt.grid(
            b=add_grid,
            which=grid_which,
            axis=grid_axis,
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
        )

    lines = []
    linewidth = np.random.randint(1, 5)
    linewidth = (
        linewidth
        if linewidth <= 2
        else (2 if np.random.uniform() < 0.3 else linewidth)
        if linewidth == 3
        else 2
        if np.random.uniform() < 0.6
        else linewidth
    )
    nxdata = (
        x_ticks if align_xdata else np.random.randint(max(2, x_ticks // 2), x_ticks * 3)
    )
    solid_only = np.random.uniform() < 0.7 and not monotone

    colors = random_color_table(nlines)
    legends = []
    lines_data = []
    shuffled_indices = [0, 1, 2, 3]
    random.shuffle([0, 1, 2, 3])
    linestyle_indices = []
    for j in range(nlines):
        if align_xdata:
            xdata = np.array([x_start + x_interval * i for i in range(nxdata)])
            ydata = np.random.uniform(
                y_start, y_start + y_interval * (y_ticks - 1), nxdata
            )
        else:
            xdata = np.array(
                [
                    x_start + x_interval * (x_ticks - 1) / nxdata * i
                    for i in range(nxdata)
                ]
            )
            ydata = np.random.uniform(
                y_start, y_start + y_interval * (y_ticks - 1), nxdata
            )
        if np.random.uniform() < 0.8:
            ydata = np.array(
                [
                    (
                        ydata[max(i - 1, 0)]
                        + ydata[i]
                        + ydata[min(i + 1, len(ydata) - 1)]
                    )
                    / 3
                    for i in range(len(ydata))
                ]
            )
        if np.random.uniform() < 0.8:
            ydata = np.array(
                [
                    (
                        ydata[max(i - 1, 0)]
                        + ydata[i]
                        + ydata[min(i + 1, len(ydata) - 1)]
                    )
                    / 3
                    for i in range(len(ydata))
                ]
            )
        if flat:
            avg = ydata.mean()
            ydata = np.array([avg for i in range(len(ydata))])
        if smooth:
            xmin = min(xdata)
            xmax = max(xdata)
            xdata_new = np.array(
                [x_start + x_interval * (x_ticks - 1) / 100 * i for i in range(100)]
            )
            xdata_new = xdata_new[xdata_new >= xmin]
            xdata_new = xdata_new[xdata_new <= xmax]
            try:
                f = interp1d(xdata, ydata, "cubic")
                xdata = xdata_new
                ydata = np.array([f(x) for x in xdata_new])
                add_noise = np.random.uniform() < 0.8 and linewidth <= 2
                if add_noise:
                    noise = np.random.normal(0, y_interval / 3, ydata.shape)
                    ydata = ydata + noise
                    ydata = np.array(
                        [
                            (
                                ydata[max(i - 1, 0)]
                                + ydata[i]
                                + ydata[min(i + 1, len(ydata) - 1)]
                            )
                            / 3
                            for i in range(len(ydata))
                        ]
                    )
            except:
                pass
        linestyle = (
            "solid"
            if solid_only
            else linestyles[int(np.random.randint(0, len(linestyles)))]
            if not monotone
            else linestyles[shuffled_indices[j]]
        )
        linestyle_indices.append(
            0
            if linestyle == "solid"
            else 1
            if linestyle == "dashed"
            else 2
            if linestyle == "dashdot"
            else 3
        )
        color = colors[0] if monotone else colors[j]
        color = "black" if grayscale else color
        alpha = np.random.uniform(0.5, 1.0) if grayscale else None
        marker = (
            markers[int(np.random.randint(0, len(markers) - 1))] if add_marker else None
        )
        p = ax.plot(
            xdata,
            ydata,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
        )
        lines.append(p)
        lines_data.append(list(zip(xdata, ydata)))

    add_title = np.random.uniform() < 0.5
    add_axis_label = np.random.uniform() < 0.5
    inline_label = np.random.uniform() < 0.8 and nlines < 5

    aspect = np.random.uniform(0.8, 2.0)
    force_aspect(aspect)

    global fontname
    fontname = fontnames[int(np.random.randint(len(fontnames)))]

    if add_title:
        title_size = np.random.randint(10, 50)
        word = random_text()
        title = ax.set_title(
            word.replace("_", " "), fontsize=title_size, fontname=fontname
        )

    if add_axis_label:
        axis_label_size = np.random.randint(10, 30)
        xlabel_text = random_text()
        ylabel_text = random_text()
        xlabel_text = xlabel_text.replace("_", " ")
        ylabel_text = ylabel_text.replace("_", " ")
        xlabel = plt.xlabel(xlabel_text, fontsize=axis_label_size, fontname=fontname)
        ylabel = plt.ylabel(ylabel_text, fontsize=axis_label_size, fontname=fontname)

    add_legend = np.random.uniform() < 0.5
    horz_legend = False
    if add_legend:
        legend_frame_on = np.random.uniform() < 0.7
        legend_outside = np.random.uniform() < 0.5

        if legend_outside:
            locations = ["center left"]
            if not add_title:
                locations.append("lower center")
            if not add_axis_label:
                locations.append("upper center")
            loc = locations[int(np.random.randint(0, len(locations)))]
            ncol = 1
            if loc == "center left":
                anchor = (np.random.uniform(1.0, 1.05), np.random.uniform(0.1, 0.9))
            elif loc == "lower center":
                anchor = (np.random.uniform(0.1, 0.2), np.random.uniform(1.0, 1.1))
                if np.random.uniform() < 0.5:
                    anchor = (np.random.uniform(0.8, 0.9), np.random.uniform(1.0, 1.1))
                if np.random.uniform() < 0.7:
                    horz_legend = True
                    ncol = nlines
            else:  # upper center
                anchor = (np.random.uniform(0.1, 0.2), np.random.uniform(-0.1, 0.0))
                if np.random.uniform() < 0.5:
                    anchor = (np.random.uniform(0.8, 0.9), np.random.uniform(-0.1, 0.0))
                if np.random.uniform() < 0.7:
                    horz_legend = True
                    ncol = nlines
            l = ax.legend(
                [p[0] for p in lines],
                [random_text(max_length=8).replace("_", " ") for _ in range(nlines)],
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                frameon=legend_frame_on,
                bbox_to_anchor=anchor,
                loc=loc,
                ncol=ncol,
            )
        else:
            l = ax.legend(
                [p[0] for p in lines],
                [random_text(max_length=8).replace("_", " ") for _ in range(nlines)],
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                frameon=legend_frame_on,
                loc="best",
            )
        legends.append(l)
    elif inline_label:
        inline_labels = inline_legend(
            colors if not (monotone or grayscale) else [color] * nlines
        )

    axis_width = np.random.randint(1, 4)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(axis_width)
    add_box = np.random.uniform() < 0.5
    if not add_box:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)

    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"line{i}.png"))

    lines = [x[0].get_data() for x in lines]
    lines = [ax.transData.transform(np.vstack([x, y]).T) for x, y in lines]
    lines = [[(x, height - y) for x, y in line] for line in lines]

    ox, oy = ax.transAxes.transform((0, 0)).T
    oy = height - oy
    tx, ty = ax.transAxes.transform((0, 1)).T
    ty = height - ty
    rx, ry = ax.transAxes.transform((1, 0)).T
    ry = height - ry

    # invalidate generated charts if elements are too much out of frame or too much overlapped, etc.
    valid = True
    all_bbs = []

    # generate annotations
    if add_title:
        title_bb = title.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        title_bb = title_bb.x0, title_bb.y0, title_bb.x1, title_bb.y1
        title_bb = (
            int(title_bb[0]),
            int(height - title_bb[3]),
            int(title_bb[2]),
            int(height - title_bb[1]),
        )
        if (
            title_bb[0] < 0
            or title_bb[1] < 0
            or title_bb[2] >= width
            or title_bb[3] >= height
        ):
            valid = False
        all_bbs.append(title_bb)

    if add_axis_label:
        xlabel_bb = xlabel.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        xlabel_bb = xlabel_bb.x0, xlabel_bb.y0, xlabel_bb.x1, xlabel_bb.y1
        xlabel_bb = (
            int(xlabel_bb[0]),
            int(height - xlabel_bb[3]),
            int(xlabel_bb[2]),
            int(height - xlabel_bb[1]),
        )
        if (
            xlabel_bb[0] < 0
            or xlabel_bb[1] < 0
            or xlabel_bb[2] >= width
            or xlabel_bb[3] >= height
        ):
            valid = False

        ylabel_bb = ylabel.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        ylabel_bb = ylabel_bb.x0, ylabel_bb.y0, ylabel_bb.x1, ylabel_bb.y1
        ylabel_bb = (
            int(ylabel_bb[0]),
            int(height - ylabel_bb[3]),
            int(ylabel_bb[2]),
            int(height - ylabel_bb[1]),
        )
        if (
            ylabel_bb[0] < 0
            or ylabel_bb[1] < 0
            or ylabel_bb[2] >= width
            or ylabel_bb[3] >= height
        ):
            valid = False
        all_bbs.append(xlabel_bb)
        all_bbs.append(ylabel_bb)

    inline_label_bbs = []
    if add_legend:
        legend_texts = ax.get_legend().get_texts()
        legend_text_bbs = []
        for l in legend_texts:
            l_bb = l.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            l_bb = l_bb.x0, l_bb.y0, l_bb.x1, l_bb.y1
            l_bb = (
                int(l_bb[0]),
                int(height - l_bb[3]),
                int(l_bb[2]),
                int(height - l_bb[1]),
            )
            legend_text_bbs.append(l_bb)
            if l_bb[0] < 0 or l_bb[1] < 0 or l_bb[2] >= width or l_bb[3] >= height:
                valid = False
            all_bbs.append(l_bb)
        if horz_legend:
            legend_markers = (
                ax.get_legend().get_children()[0].get_children()[1].get_children()
            )
            legend_markers = [lm.get_children()[0] for lm in legend_markers]
        else:
            legend_markers = (
                ax.get_legend()
                .get_children()[0]
                .get_children()[1]
                .get_children()[0]
                .get_children()
            )
        legend_markers = [lm.get_children()[0] for lm in legend_markers]
        legend_marker_bbs = []
        for lm in legend_markers:
            lm_bb = lm.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            lm_bb = lm_bb.x0, lm_bb.y0, lm_bb.x1, lm_bb.y1
            lm_bb = (
                int(lm_bb[0]),
                int(height - lm_bb[3]),
                int(lm_bb[2]),
                int(height - lm_bb[1]),
            )
            legend_marker_bbs.append(lm_bb)
            if lm_bb[0] < 0 or lm_bb[1] < 0 or lm_bb[2] >= width or lm_bb[3] >= height:
                valid = False
            all_bbs.append(lm_bb)

    elif inline_label:
        for il in inline_labels:
            il_bb = il.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            il_bb = il_bb.x0, il_bb.y0, il_bb.x1, il_bb.y1
            il_bb = (
                int(il_bb[0]),
                int(height - il_bb[3]),
                int(il_bb[2]),
                int(height - il_bb[1]),
            )
            inline_label_bbs.append(il_bb)
            if il_bb[0] < 0 or il_bb[1] < 0 or il_bb[2] >= width or il_bb[3] >= height:
                valid = False
            all_bbs.append(il_bb)

    xtick_labels = ax.get_xticklabels()[1:-1]
    xtick_label_bbs = []
    for xt in xtick_labels:
        xt_bb = xt.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        xt_bb = xt_bb.x0, xt_bb.y0, xt_bb.x1, xt_bb.y1
        xt_bb = (
            int(xt_bb[0]),
            int(height - xt_bb[3]),
            int(xt_bb[2]),
            int(height - xt_bb[1]),
        )
        xtick_label_bbs.append(xt_bb)
        if xt_bb[0] < 0 or xt_bb[1] < 0 or xt_bb[2] >= width or xt_bb[3] >= height:
            valid = False
        all_bbs.append(xt_bb)

    ytick_labels = list(ax.get_yticklabels())[1:-1]
    ytick_label_bbs = []
    for yt in ytick_labels:
        yt_bb = yt.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        yt_bb = yt_bb.x0, yt_bb.y0, yt_bb.x1, yt_bb.y1
        yt_bb = (
            int(yt_bb[0]),
            int(height - yt_bb[3]),
            int(yt_bb[2]),
            int(height - yt_bb[1]),
        )
        ytick_label_bbs.append(yt_bb)
        if yt_bb[0] < 0 or yt_bb[1] < 0 or yt_bb[2] >= width or yt_bb[3] >= height:
            valid = False
        all_bbs.append(yt_bb)

    x_min, x_max = ax.get_xlim()
    xticks = [(tick - x_min) / (x_max - x_min) for tick in ax.get_xticks()][1:-1]
    xticks = [ax.transAxes.transform((tick, 0)).T for tick in xticks]
    xticks = [(x, height - y) for x, y in xticks]

    y_min, y_max = ax.get_ylim()
    yticks = [(tick - y_min) / (y_max - y_min) for tick in ax.get_yticks()][1:-1]
    yticks = [ax.transAxes.transform((0, tick)).T for tick in yticks]
    yticks = [(x, height - y) for x, y in yticks]

    for idx_a, bb_a in enumerate(all_bbs):
        area_a = (bb_a[2] - bb_a[0] + 1) * (bb_a[3] - bb_a[1] + 1)
        for idx_b in range(idx_a + 1, len(all_bbs)):
            bb_b = all_bbs[idx_b]
            area_b = (bb_b[2] - bb_b[0] + 1) * (bb_b[3] - bb_b[1] + 1)
            bb_int = [
                max(bb_a[0], bb_b[0]),
                max(bb_a[1], bb_b[1]),
                min(bb_a[2], bb_b[2]),
                min(bb_a[3], bb_b[3]),
            ]
            if bb_int[2] < bb_int[0] or bb_int[3] < bb_int[1]:
                area_int = 0
            else:
                area_int = (bb_int[2] - bb_int[0] + 1) * (bb_int[3] - bb_int[1] + 1)
            r = area_int / max(1, min(area_a, area_b))
            if r > 0.1:
                valid = False
                break
        if not valid:
            break

    chart = {
        "name": f"{i}",
        "type": "line",
        "data": {
            "title": {
                "value": title.get_text(),
                "x0": int(title_bb[0]),
                "y0": int(title_bb[1]),
                "x1": int(title_bb[2]),
                "y1": int(title_bb[3]),
            }
            if add_title
            else None,
            "xaxis": {
                "start": {"x": int(ox), "y": int(oy)},
                "end": {"x": int(rx), "y": int(ry)},
                "ticks": [
                    {
                        "x": int(xticks[i][0]),
                        "y": int(xticks[i][1]),
                    }
                    for i in range(len(xticks))
                ],
                "tick_labels": [
                    {
                        "value": xtick_labels[i].get_text(),
                        "x0": int(xtick_label_bbs[i][0]),
                        "y0": int(xtick_label_bbs[i][1]),
                        "x1": int(xtick_label_bbs[i][2]),
                        "y1": int(xtick_label_bbs[i][3]),
                    }
                    for i in range(len(xtick_labels))
                ],
                "title": {
                    "value": str(xlabel_text),
                    "x0": int(xlabel_bb[0]),
                    "y0": int(xlabel_bb[1]),
                    "x1": int(xlabel_bb[2]),
                    "y1": int(xlabel_bb[3]),
                }
                if add_axis_label
                else None,
            },
            "yaxis": {
                "start": {"x": int(ox), "y": int(oy)},
                "end": {"x": int(tx), "y": int(ty)},
                "ticks": [
                    {
                        "value": ytick_labels[i].get_text(),
                        "x": int(yticks[i][0]),
                        "y": int(yticks[i][1]),
                    }
                    for i in range(len(yticks))
                ],
                "tick_labels": [
                    {
                        "value": ytick_labels[i].get_text(),
                        "x0": int(ytick_label_bbs[i][0]),
                        "y0": int(ytick_label_bbs[i][1]),
                        "x1": int(ytick_label_bbs[i][2]),
                        "y1": int(ytick_label_bbs[i][3]),
                    }
                    for i in range(len(ytick_labels))
                ],
                "title": {
                    "value": str(ylabel_text),
                    "x0": int(ylabel_bb[0]),
                    "y0": int(ylabel_bb[1]),
                    "x1": int(ylabel_bb[2]),
                    "y1": int(ylabel_bb[3]),
                }
                if add_axis_label
                else None,
            },
            "lines": [
                {
                    "line_width": linewidth,
                    "points": [
                        {
                            "x": int(lines[i][j][0]),
                            "y": int(lines[i][j][1]),
                            "value": {
                                "x": float(lines_data[i][j][0]),
                                "y": float(lines_data[i][j][1]),
                            },
                        }
                        for j in range(len(lines[i]))
                    ],
                    "inline_label": {
                        "value": inline_labels[i].get_text(),
                        "x0": int(inline_label_bbs[i][0]),
                        "y0": int(inline_label_bbs[i][1]),
                        "x1": int(inline_label_bbs[i][2]),
                        "y1": int(inline_label_bbs[i][3]),
                    }
                    if inline_label and not add_legend
                    else None,
                    "legend": {
                        "icon": {
                            "x0": legend_marker_bbs[i][0],
                            "y0": legend_marker_bbs[i][1],
                            "x1": legend_marker_bbs[i][2],
                            "y1": legend_marker_bbs[i][3],
                        },
                        "label": {
                            "x0": legend_text_bbs[i][0],
                            "y0": legend_text_bbs[i][1],
                            "x1": legend_text_bbs[i][2],
                            "y1": legend_text_bbs[i][3],
                            "value": legend_texts[i].get_text(),
                        },
                    }
                    if add_legend
                    else None,
                    "style": linestyle_indices[i],
                }
                for i in range(len(lines))
            ],
        },
    }
    return chart, valid


def main():
    parser = argparse.ArgumentParser(description="Generate Line Charts")
    parser.add_argument("--num", type=int, help="The number of charts to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="lines",
        help="The directory path to output charts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    progress_bar = tqdm(total=args.num)
    i = 0
    while i < args.num:
        chart, valid = random_line_charts(i, args.output)
        if valid:
            with open(
                os.path.join(args.output, f"line{i}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(chart, f)
            i += 1
            progress_bar.update(1)
        plt.clf()
        plt.cla()


if __name__ == "__main__":
    main()
