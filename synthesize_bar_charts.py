import warnings
import math
import argparse
import os
import sqlite3
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

width, height = 640, 480

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

# init words list
conn = sqlite3.connect("./wnjpn.db")
jpn_word_list = conn.execute("select * from word where lang = 'jpn' ORDER BY RANDOM();")
eng_word_list = conn.execute("select * from word where lang = 'eng' ORDER BY RANDOM();")
jpn_word_list = [record[2] for record in jpn_word_list.fetchall()]
eng_word_list = [record[2] for record in eng_word_list.fetchall()]
word_lists = [jpn_word_list, eng_word_list]
ax = None
fig = None
word_list = None
ntypes = None


def random_text(max_length=None):
    word = word_list[np.random.randint(0, len(word_list))]
    if max_length is None:
        return word
    while len(word) > max_length:
        word = word_list[np.random.randint(0, len(word_list))]
    return word


def random_color_table(num):
    scale = 8
    colors = [
        (r, g, b)
        for r in range(256 // scale)
        for g in range(256 // scale)
        for b in range(256 // scale)
    ]
    colors = colors if len(colors) > 0 else [(0, 0, 0)]
    threshold = 256 // num // scale
    table = []

    def validate(color):
        for tcolor in table:
            if (
                abs(color[0] - tcolor[0]) <= threshold
                and abs(color[1] - tcolor[1]) <= threshold
                and abs(color[2] - tcolor[2]) <= threshold
            ):
                return False
        return True

    for _ in range(num):
        color = (
            colors[np.random.randint(0, len(colors))] if len(colors) > 0 else (0, 0, 0)
        )
        table.append(color)
        colors = list(filter(validate, colors))
    table = [
        (c[0] * scale / 255, c[1] * scale / 255, c[2] * scale / 255) for c in table
    ]
    return table


def random_monotone_table(num):
    scale = 8
    colors = list(range(256 // scale))
    threshold = 256 // num // 2 // scale
    table = []
    rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    rgb = rgb[int(np.random.randint(0, len(rgb)))]

    def validate(color):
        for tcolor in table:
            if abs(color - tcolor) < threshold:
                return False
        return True

    for _ in range(num):
        color = colors[int(np.random.randint(0, len(colors)))]
        table.append(color)
        colors = list(filter(validate, colors))
    table = [
        (c * scale * rgb[0] / 255, c * scale * rgb[1] / 255, c * scale * rgb[2] / 255)
        for c in table
    ]
    return table


def force_aspect(aspect=1):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) / aspect)


def get_bb_coordinate(elem):
    bbox = elem.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
    bbox = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    bbox = (int(bbox[0]), int(height - bbox[3]), int(bbox[2]), int(height - bbox[1]))
    return bbox


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


def random_bar_charts(i, output_dir, horz=False):
    global ax, fig, word_list, ntypes
    ax, fig = init_ax_fig()

    monotone = np.random.uniform() < 0.3  # or True
    ntypes = min(4, math.floor(abs(np.random.normal(0, 2)) + 1))

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

    fontname = fontnames[int(np.random.randint(len(fontnames)))]
    align_xdata = np.random.uniform() < 0.5 or True

    stacked = np.random.uniform() < 0.5

    # eng_word = int(np.random.uniform()<0.8)
    eng_word = 1
    word_list = word_lists[eng_word]

    add_grid = np.random.uniform() < 0.3
    plt.grid(add_grid)
    tmp = np.random.uniform()
    grid_axis = "both" if tmp < 0.1 else "y" if tmp < 0.9 else "x"
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
    axis_below = np.random.uniform() < 0.8
    if axis_below:
        ax.set_axisbelow(True)

    nxdata = (
        x_ticks if align_xdata else np.random.randint(max(2, x_ticks // 2), x_ticks * 2)
    )

    colors = random_monotone_table(ntypes) if monotone else random_color_table(ntypes)
    legends = []
    bars = []
    bars_data = []
    bar_width = np.random.uniform(x_interval / ntypes / 2, x_interval / ntypes * 2 / 3)
    last_bottoms = [0] * nxdata
    use_hatch = np.random.uniform() < 0.3
    hatch_patterns = ["++", "xx", "..", "**", "//", "\\\\"] + (
        ["||"] if horz else ["--"]
    )
    use_tick_label = np.random.uniform() < 0.3
    for j in range(ntypes):
        thinner = np.random.uniform() < 0.2
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
        if not stacked:
            xdata = xdata + bar_width * j
        ydata = np.array(
            [
                (ydata[max(i - 1, 0)] + ydata[i] + ydata[min(i + 1, len(ydata) - 1)])
                / 3
                for i in range(len(ydata))
            ]
        )
        if thinner:
            for k in range(ydata.shape[0]):
                data_size = np.random.randint(3, 10)
                ydata[k] /= data_size
        color = colors[j]
        hatch_pattern = hatch_patterns[int(np.random.randint(0, len(hatch_patterns)))]
        tick_labels = []
        if not use_tick_label:
            tick_labels = [None] * (xdata.shape[0])
        else:
            for k in range(xdata.shape[0]):
                word = random_text(max_length=6)
                tick_labels.append(word)
        if horz:
            if use_hatch:
                ax_bar = ax.barh(
                    xdata,
                    ydata,
                    tick_label=tick_labels if use_tick_label else None,
                    height=bar_width,
                    color=color,
                    left=last_bottoms if stacked else 0,
                    fill=None,
                    hatch=hatch_pattern,
                    edgecolor=color,
                )
            else:
                ax_bar = ax.barh(
                    xdata,
                    ydata,
                    tick_label=tick_labels if use_tick_label else None,
                    height=bar_width,
                    color=color,
                    left=last_bottoms if stacked else 0,
                )
        else:
            if use_hatch:
                ax_bar = ax.bar(
                    xdata,
                    ydata,
                    tick_label=tick_labels if use_tick_label else None,
                    width=bar_width,
                    color=color,
                    bottom=last_bottoms if stacked else 0,
                    fill=None,
                    hatch=hatch_pattern,
                    edgecolor=color,
                )
            else:
                ax_bar = ax.bar(
                    xdata,
                    ydata,
                    tick_label=tick_labels if use_tick_label else None,
                    width=bar_width,
                    color=color,
                    bottom=last_bottoms if stacked else 0,
                )
        bars_data.append(list(zip(xdata, ydata, tick_labels)))
        bars.append(ax_bar)
        last_bottoms += ydata

    add_title = np.random.uniform() < 0.5
    add_axis_label = np.random.uniform() < 0.5

    aspect = np.random.uniform(0.8, 2.0)
    force_aspect(aspect)

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
        ylabel = ylabel_text.replace("_", " ")
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
                    ncol = ntypes
            else:  # upper center
                anchor = (np.random.uniform(0.1, 0.2), np.random.uniform(-0.1, 0.0))
                if np.random.uniform() < 0.5:
                    anchor = (np.random.uniform(0.8, 0.9), np.random.uniform(-0.1, 0.0))
                if np.random.uniform() < 0.7:
                    horz_legend = True
                    ncol = ntypes
            ax_legend = ax.legend(
                [b[0] for b in bars],
                [random_text(max_length=8).replace("_", " ") for _ in range(ntypes)],
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                frameon=legend_frame_on,
                bbox_to_anchor=anchor,
                loc=loc,
                ncol=ncol,
            )
        else:
            ax_legend = ax.legend(
                [b[0] for b in bars],
                [random_text(max_length=8).replace("_", " ") for _ in range(ntypes)],
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                frameon=legend_frame_on,
                loc="best",
            )
        legends.append(ax_legend)

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

    plt.savefig(os.path.join(output_dir, f'{"hbar" if horz else "vbar"}{i}.png'))

    org_x, org_y = ax.transAxes.transform((0, 0)).T
    org_y = height - org_y
    trns_x, trns_y = ax.transAxes.transform((0, 1)).T
    trns_y = height - trns_y
    rev_x, rev_y = ax.transAxes.transform((1, 0)).T
    rev_y = height - rev_y

    valid = True
    all_bbs = []

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
            or title_bb[2] >= 640
            or title_bb[3] >= 480
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
            or xlabel_bb[2] >= 640
            or xlabel_bb[3] >= 480
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
            or ylabel_bb[2] >= 640
            or ylabel_bb[3] >= 480
        ):
            valid = False
        all_bbs.append(xlabel_bb)
        all_bbs.append(ylabel_bb)

    if add_legend:
        legend_texts = ax.get_legend().get_texts()
        legend_text_bbs = []
        for legend_text in legend_texts:
            l_bb = legend_text.get_window_extent(
                renderer=plt.gcf().canvas.get_renderer()
            )
            l_bb = l_bb.x0, l_bb.y0, l_bb.x1, l_bb.y1
            l_bb = (
                int(l_bb[0]),
                int(height - l_bb[3]),
                int(l_bb[2]),
                int(height - l_bb[1]),
            )
            legend_text_bbs.append(l_bb)
            if l_bb[0] < 0 or l_bb[1] < 0 or l_bb[2] >= 640 or l_bb[3] >= 480:
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
        for legend_marker in legend_markers:
            lm_bb = legend_marker.get_window_extent(
                renderer=plt.gcf().canvas.get_renderer()
            )
            lm_bb = lm_bb.x0, lm_bb.y0, lm_bb.x1, lm_bb.y1
            lm_bb = (
                int(lm_bb[0]),
                int(height - lm_bb[3]),
                int(lm_bb[2]),
                int(height - lm_bb[1]),
            )
            legend_marker_bbs.append(lm_bb)
            if lm_bb[0] < 0 or lm_bb[1] < 0 or lm_bb[2] >= 640 or lm_bb[3] >= 480:
                valid = False
            all_bbs.append(lm_bb)

    if use_tick_label:
        xtick_labels = ax.get_xticklabels()
    else:
        xtick_labels = ax.get_xticklabels()[1:-1]
    xtick_label_bbs = []
    for xt_label in xtick_labels:
        xt_bb = xt_label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        xt_bb = xt_bb.x0, xt_bb.y0, xt_bb.x1, xt_bb.y1
        xt_bb = (
            int(xt_bb[0]),
            int(height - xt_bb[3]),
            int(xt_bb[2]),
            int(height - xt_bb[1]),
        )
        xtick_label_bbs.append(xt_bb)
        if xt_bb[0] < 0 or xt_bb[1] < 0 or xt_bb[2] >= 640 or xt_bb[3] >= 480:
            valid = False
        all_bbs.append(xt_bb)

    ytick_labels = list(ax.get_yticklabels())[:-1]
    ytick_label_bbs = []
    for yt_label in ytick_labels:
        yt_bb = yt_label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        yt_bb = yt_bb.x0, yt_bb.y0, yt_bb.x1, yt_bb.y1
        yt_bb = (
            int(yt_bb[0]),
            int(height - yt_bb[3]),
            int(yt_bb[2]),
            int(height - yt_bb[1]),
        )
        ytick_label_bbs.append(yt_bb)
        if yt_bb[0] < 0 or yt_bb[1] < 0 or yt_bb[2] >= 640 or yt_bb[3] >= 480:
            valid = False
        all_bbs.append(yt_bb)

    x_min, x_max = ax.get_xlim()
    if use_tick_label:
        xticks = [(tick - x_min) / (x_max - x_min) for tick in ax.get_xticks()]
    else:
        xticks = [(tick - x_min) / (x_max - x_min) for tick in ax.get_xticks()][1:-1]
    xticks = [ax.transAxes.transform((tick, 0)).T for tick in xticks]
    xticks = [(x, height - y) for x, y in xticks]

    y_min, y_max = ax.get_ylim()
    yticks = [(tick - y_min) / (y_max - y_min) for tick in ax.get_yticks()][1:-1]
    yticks = [ax.transAxes.transform((0, tick)).T for tick in yticks]
    yticks = [(x, height - y) for x, y in yticks]

    bar_bbs = [[get_bb_coordinate(b) for b in bbs.get_children()] for bbs in bars]
    for bboxes in bar_bbs:
        for bbox in bboxes:
            all_bbs.append(bbox)

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
            ratio = area_int / max(1, min(area_a, area_b))
            if ratio > 0.1:
                valid = False
                break
        if not valid:
            break

    chart = {
        "name": f"{i}",
        "type": "hbar" if horz else "vbar",
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
                "start": {"x": int(org_x), "y": int(org_y)},
                "end": {"x": int(rev_x), "y": int(rev_y)},
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
                "start": {"x": int(org_x), "y": int(org_y)},
                "end": {"x": int(trns_x), "y": int(trns_y)},
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
            "stacked": stacked,
            "bars": [
                {
                    "bar": [
                        {
                            "x0": int(bar_bbs[i][j][0]),
                            "y0": int(bar_bbs[i][j][1]),
                            "x1": int(bar_bbs[i][j][2]),
                            "y1": int(bar_bbs[i][j][3]),
                            "value": {
                                "x": float(bars_data[i][j][0]),
                                "y": float(bars_data[i][j][1]),
                                "label": bars_data[i][j][2],
                            },
                        }
                        for j in range(len(bar_bbs[i]))
                    ],
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
                }
                for i in range(len(bars))
            ],
        },
    }
    return chart, valid


def main():
    parser = argparse.ArgumentParser(description="Generate Bar Charts")
    parser.add_argument("--horz", action="store_true", help="Horizontal Bar")
    parser.add_argument("--num", type=int, help="The number of charts to generate")
    parser.add_argument(
        "--output", type=str, default="bars", help="The directory path to output charts"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    chart_type = "hbar" if args.horz else "vbar"

    progress_bar = tqdm(total=args.num)
    i = 0
    while i < args.num:
        chart, valid = random_bar_charts(i, args.output, args.horz)
        if valid:
            with open(
                os.path.join(args.output, f"{chart_type}{i}.json"),
                "w",
                encoding="utf-8",
            ) as f_out:
                json.dump(chart, f_out)
            i += 1
            progress_bar.update(1)
        plt.clf()
        plt.cla()


if __name__ == "__main__":
    main()
