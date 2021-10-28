import warnings
import math
import argparse
import os
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.font_manager import FontProperties

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
fontname = fontnames[0]

conn = sqlite3.connect("./wnjpn.db")
jpn_word_list = conn.execute("select * from word where lang = 'jpn' ORDER BY RANDOM();")
eng_word_list = conn.execute("select * from word where lang = 'eng' ORDER BY RANDOM();")
jpn_word_list = [record[2] for record in jpn_word_list.fetchall()]
eng_word_list = [record[2] for record in eng_word_list.fetchall()]
words_list = [jpn_word_list, eng_word_list]
word_list = words_list[0]
ax = None


def random_text(max_length=None):
    word = word_list[np.random.randint(0, len(word_list))]
    if max_length is None:
        return word
    while len(word) > max_length:
        word = word_list[np.random.randint(0, len(word_list))]
    return word


def rand_color_table(n):
    scale = 8
    colors = [
        (r, g, b)
        for r in range(256 // scale)
        for g in range(256 // scale)
        for b in range(256 // scale)
    ]
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


def get_bb_coordinate(e):
    e_bb = e.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
    bb = e_bb.x0, e_bb.y0, e_bb.x1, e_bb.y1
    bb = (int(bb[0]), int(height - bb[3]), int(bb[2]), int(height - bb[1]))
    return bb


def random_intervals(n, start=0.0, end=1.0):
    if n == 1:
        return []
    mid_value = np.random.uniform(start + 0.001, end - 0.001)
    if n == 2:
        return [mid_value]
    mid_index = (n - 1) // 2
    return (
        random_intervals(mid_index + 1, start, mid_value)
        + [mid_value]
        + random_intervals(n - mid_index - 1, mid_value, end)
    )


def init_ax():
    global ax
    plt.style.use("default")
    ax = plt.gca()
    ax.set_aspect("equal")
    return ax


def random_pie_charts(i, output_dir):
    global ax, word_list, fontname
    ax = init_ax()

    fontname = fontnames[int(np.random.randint(len(fontnames)))]
    thinner = np.random.uniform() < 0.3
    npies = max(min(10, math.floor(abs(np.random.normal(0, 4)) + 1)), 2)
    intervals = [0.0] + random_intervals(npies) + [1.0]
    slice_ratios = [intervals[j + 1] - intervals[j] for j in range(npies)]
    sort_slices = np.random.uniform() < 0.5
    if sort_slices:
        slice_ratios.sort()

    slice_amounts = None
    total_amount = None

    # eng_word = int(np.random.uniform()<0.8)
    eng_word = 1
    word_list = words_list[eng_word]

    colors = rand_color_table(npies)

    donut = np.random.uniform() < 0.2
    explode = np.random.uniform() < 0.3 and not donut
    explodes = (
        [
            0
            if slice_ratios[j] >= 0.5
            else np.random.uniform(0.1, 0.5)
            if np.random.uniform() < 0.2
            else 0
            for j in range(npies)
        ]
        if explode
        else [0] * npies
    )

    use_hatch = np.random.uniform() < 0.3
    hatch_patterns = ["++", "xx", "..", "**", "//", "\\\\", "||", "--"]

    add_pie_label = np.random.uniform() < 0.4
    add_amount = np.random.uniform() < 0.5 and not add_pie_label
    if add_amount:
        ndigits = np.random.randint(2, 5)
        total_amount = np.random.randint(10 ** ndigits, 10 ** (ndigits + 1) - 1)
        slice_amounts = [total_amount * slice_ratios[j] for j in range(npies)]
        acc = 0
        for j in range(npies - 1):
            if thinner:
                d = np.random.randint(2, 5)
                slice_amounts[j] /= d
            slice_amounts[j] = int(slice_amounts[j])
            acc += slice_amounts[j]
            intervals[j + 1] = acc / total_amount
        slice_amounts[-1] = total_amount - acc
        slice_ratios = [sa / total_amount for sa in slice_amounts]
    valid = True
    for sr in slice_ratios:
        if sr < 0:
            valid = False

    if not valid:
        return None, valid

    pie_labels = None
    if add_pie_label or add_amount:
        pie_labels = (
            [word_list[np.random.randint(0, len(word_list) - 1)] for _ in range(npies)]
            if add_pie_label
            else [str(sa) for sa in slice_amounts]
        )

    radius = np.random.uniform(0.8, 1.2)

    if use_hatch:
        size = np.random.uniform(10, 15)
        pie = ax.pie(
            slice_ratios,
            labels=pie_labels
            if add_pie_label
            else slice_amounts
            if add_amount
            else None,
            textprops={
                "fontname": fontname,
                "fontproperties": FontProperties(size=size),
            },
            labeldistance=np.random.uniform(0.5, 1.2),
            colors=[(1, 1, 1)] * len(colors),
            explode=explodes,
            radius=radius,
        )
    else:
        size = np.random.uniform(10, 15)
        pie = ax.pie(
            slice_ratios,
            labels=pie_labels
            if add_pie_label
            else slice_amounts
            if add_amount
            else None,
            textprops={
                "fontname": fontname,
                "fontproperties": FontProperties(size=size),
            },
            labeldistance=np.random.uniform(0.5, 1.2),
            colors=colors,
            explode=explodes,
            radius=radius,
        )

    if donut:
        donut_radius = np.random.uniform(0.4, 0.6) * radius
        _center_circle = ax.pie([1], colors=[(1, 1, 1)], radius=donut_radius)

    for j in range(len(pie[0])):
        hatch_pattern = hatch_patterns[int(np.random.randint(0, len(hatch_patterns)))]
        pie[0][j].set_edgecolor(colors[j])
        pie[0][j].set_hatch(hatch_pattern)

    add_title = np.random.uniform() < 0.5

    if add_title:
        title_size = np.random.randint(10, 50)
        title = ax.set_title(
            word_list[np.random.randint(0, len(word_list) - 1)].replace("_", " "),
            fontsize=title_size,
            fontname=fontname,
        )

    legends = []
    add_legend = np.random.uniform() < 0.7 and not add_pie_label
    horz_legend = False
    if add_legend:
        legend_frame_on = False
        legend_outside = np.random.uniform() < 0.6
        horz_legend = False
        if legend_outside:
            legend_frame_on = np.random.uniform() < 0.7
            locations = ["center left", "center right", "upper center"]
            if not add_title:
                locations.append("lower center")
            loc = locations[int(np.random.randint(0, len(locations)))]
            ncol = 1
            if loc == "center left":
                anchor = (np.random.uniform(1.0, 1.05), np.random.uniform(0.1, 0.9))
            elif loc == "center right":
                anchor = (np.random.uniform(-0.05, 0.0), np.random.uniform(0.1, 0.9))
            elif loc == "lower center":
                anchor = (np.random.uniform(0.1, 0.9), np.random.uniform(1.0, 1.1))
                if np.random.uniform() < 0.7:
                    horz_legend = True
                    ncol = npies
            else:  # upper center
                anchor = (np.random.uniform(0.1, 0.9), np.random.uniform(-0.1, 0.0))
                if np.random.uniform() < 0.7:
                    horz_legend = True
                    ncol = npies
            l = ax.legend(
                [random_text() for _ in range(npies)]
                if pie_labels is None
                else pie_labels,
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                frameon=legend_frame_on,
                bbox_to_anchor=anchor,
                loc=loc,
                ncol=ncol,
            )
        else:
            l = ax.legend(
                [random_text().replace("_", " ") for _ in range(npies)]
                if pie_labels is None
                else pie_labels,
                prop={"family": fontname, "size": np.random.randint(10, 15)},
                loc="best",
            )

        legends.append(l)

    plt.savefig(os.path.join(output_dir, f"pie{i}.png"))

    valid = True
    all_bbs = []

    slices = list(pie[0])
    pie_labels = list(pie[1])
    pie_label_bbs = []
    if add_pie_label or add_amount:
        for l in pie_labels:
            pie_label_bb = l.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            pie_label_bb = (
                pie_label_bb.x0,
                pie_label_bb.y0,
                pie_label_bb.x1,
                pie_label_bb.y1,
            )
            pie_label_bb = (
                int(pie_label_bb[0]),
                int(height - pie_label_bb[3]),
                int(pie_label_bb[2]),
                int(height - pie_label_bb[1]),
            )
            pie_label_bbs.append(pie_label_bb)
            if (
                pie_label_bb[0] < 0
                or pie_label_bb[1] < 0
                or pie_label_bb[2] >= 640
                or pie_label_bb[3] >= 480
            ):
                valid = False
            all_bbs.append(pie_label_bb)

    slice_bbs = []
    slice_centers = []
    pie_bb = (10000, 10000, -10000, -10000)
    front_bbs = []
    for s in slices:
        slice_bb = s.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        slice_bb = slice_bb.x0, slice_bb.y0, slice_bb.x1, slice_bb.y1
        slice_bb = (
            int(slice_bb[0]),
            int(height - slice_bb[3]),
            int(slice_bb[2]),
            int(height - slice_bb[1]),
        )
        slice_bbs.append(pie_bb)
        slice_center = s.get_transform().transform(s.center)
        slice_center = (int(slice_center[0]), int(slice_center[1]))
        slice_centers.append(slice_center)
        pie_bb = (
            min(pie_bb[0], slice_bb[0]),
            min(pie_bb[1], slice_bb[1]),
            max(pie_bb[2], slice_bb[2]),
            max(pie_bb[3], slice_bb[3]),
        )
    front_bbs.append(pie_bb)
    pie_center = ax.transAxes.transform((0.5, 0.5))
    pie_center = (int(pie_center[0]), int(pie_center[1]) + 2)
    pie_radius = slices[0].get_transform().transform([(slices[0].r, 0), (0, 0)])
    pie_radius = int(pie_radius[0][0] - pie_radius[1][0] + 0.5)
    slice_offsets = [
        (s[0] - pie_center[0], s[1] - pie_center[1]) for s in slice_centers
    ]

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
            if l_bb[0] < 0 or l_bb[1] < 0 or l_bb[2] >= 640 or l_bb[3] >= 480:
                valid = False
            all_bbs.append(l_bb)
            front_bbs.append(l_bb)
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
            if lm_bb[0] < 0 or lm_bb[1] < 0 or lm_bb[2] >= 640 or lm_bb[3] >= 480:
                valid = False
            all_bbs.append(lm_bb)

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

    overlaps = 0
    for idx_a, bb_a in enumerate(front_bbs):
        area_a = (bb_a[2] - bb_a[0] + 1) * (bb_a[3] - bb_a[1] + 1)
        for idx_b in range(idx_a + 1, len(front_bbs)):
            bb_b = front_bbs[idx_b]
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
            if r > 0.9:
                overlaps += 1
                if overlaps >= 3:
                    valid = False
                    break
        if not valid:
            break

    chart = {
        "name": f"{i}",
        "type": "pie",
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
            "pie": {
                "center": {"x": int(pie_center[0]), "y": int(pie_center[1])},
                "radius": int(pie_radius),
                "total_amount": total_amount,
                "slices": [
                    {
                        "angle0": slices[j].theta1,
                        "angle1": slices[j].theta2,
                        "ratio": slice_ratios[j],
                        "amount": slice_amounts[j] if slice_amounts else None,
                        "explode": explodes[j],
                        "donut": {"radius": donut_radius} if donut else None,
                        "offset": {
                            "x": slice_offsets[j][0],
                            "y": slice_offsets[j][1],
                        },
                        "inline_label": {
                            "value": pie_labels[j].get_text(),
                            "x0": int(pie_label_bbs[j][0]),
                            "y0": int(pie_label_bbs[j][1]),
                            "x1": int(pie_label_bbs[j][2]),
                            "y1": int(pie_label_bbs[j][3]),
                        }
                        if len(pie_label_bbs) > 0 and add_pie_label
                        else None,
                        "amount_label": {
                            "value": pie_labels[j].get_text(),
                            "x0": int(pie_label_bbs[j][0]),
                            "y0": int(pie_label_bbs[j][1]),
                            "x1": int(pie_label_bbs[j][2]),
                            "y1": int(pie_label_bbs[j][3]),
                        }
                        if len(pie_label_bbs) > 0 and add_amount
                        else None,
                        "legend": {
                            "icon": {
                                "x0": legend_marker_bbs[j][0],
                                "y0": legend_marker_bbs[j][1],
                                "x1": legend_marker_bbs[j][2],
                                "y1": legend_marker_bbs[j][3],
                            },
                            "label": {
                                "x0": legend_text_bbs[j][0],
                                "y0": legend_text_bbs[j][1],
                                "x1": legend_text_bbs[j][2],
                                "y1": legend_text_bbs[j][3],
                                "value": legend_texts[j].get_text(),
                            },
                        }
                        if add_legend
                        else None,
                    }
                    for j in range(npies)
                ],
            },
        },
    }
    return chart, valid


def main():
    parser = argparse.ArgumentParser(description="Generate Pie Charts")
    parser.add_argument("--num", type=int, help="The number of charts to generate")
    parser.add_argument(
        "--output", type=str, default="pies", help="The directory path to output charts"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    progress_bar = tqdm(total=args.num)
    i = 0
    while i < args.num:
        chart, valid = random_pie_charts(i, args.output)
        if valid:
            with open(
                os.path.join(args.output, f"pie{i}.json"), "w", encoding="utf-8"
            ) as f_out:
                json.dump(chart, f_out)
            i += 1
            progress_bar.update(1)
        plt.clf()
        plt.cla()


if __name__ == "__main__":
    main()
