import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
# import matplotlib.cm as cm

# Muted palette (architecture-figure blues/greens, no pastel callout boxes).
THESIS_EIG_COLORS = {
    'marker': '#3d4f5f',
    'marker_edge': '#ffffff',
    'grid': '#e8ebef',
    'axis': '#5a6573',
    'zero_line': '#a8b0ba',
    'text': '#2d3748',
    'baseline': '#5f7349',
    'coupled': '#3f7f74',
}


def _eig_label(lam):
    freq_hz = lam.imag / (2 * np.pi)
    mag = np.abs(lam)
    damp_pct = -100 * lam.real / mag if mag > 0 else 0.0
    return f'{freq_hz:.2f} Hz\n{damp_pct:.1f}%'


def _is_conjugate_pair(a, b, tol_re, tol_im):
    return (
        abs(a.real - b.real) < tol_re
        and abs(a.imag + b.imag) < tol_im
        and a.imag * b.imag < 0
    )


def _eigs_for_annotation(eigs, x_span, y_lim):
    """Annotate distinct modes; skip duplicate positions and minor conjugates only."""
    tol_re = max(0.4, 0.003 * x_span)
    tol_im = max(0.05, 0.02 * y_lim)
    pos_tol_re = max(0.15, 0.0015 * x_span)
    pos_tol_im = max(0.02, 0.005 * y_lim)
    label_both_conj_im = 0.25 * y_lim
    selected = []
    for i, lam in enumerate(eigs):
        if lam.imag < 0 and abs(lam.imag) < label_both_conj_im and any(
            _is_conjugate_pair(lam, eigs[j], tol_re, tol_im) for j in range(len(eigs)) if j != i
        ):
            continue
        pos = (lam.real, lam.imag)
        if any(
            abs(pos[0] - p[0]) < pos_tol_re and abs(pos[1] - p[1]) < pos_tol_im
            for _, p in selected
        ):
            continue
        selected.append((lam, pos))
    return selected


def _estimate_label_size_px(label_text, fs_annot, dpi):
    lines = label_text.split('\n')
    max_chars = max(len(line) for line in lines)
    px_per_pt = dpi / 72.0
    fs_px = fs_annot * px_per_pt
    width = max_chars * fs_px * 0.58 + 10
    height = len(lines) * fs_px * 1.25 + 8
    return width, height


def _label_bbox_px(cx, cy, width, height):
    return (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height)


def _bboxes_overlap(a, b, pad=6):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _bbox_inside(bbox, inner):
    return bbox[0] >= inner[0] and bbox[1] >= inner[1] and bbox[2] <= inner[2] and bbox[3] <= inner[3]


def _axes_inner_bbox_px(ax, renderer, pad=12):
    bb = ax.get_window_extent(renderer)
    return (bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def _screen_label_groups(ax, to_label, min_px=78):
    """Group labels whose anchor markers sit close together on screen."""
    fig = ax.figure
    fig.canvas.draw()
    disp = [ax.transData.transform(pos) for _, pos in to_label]
    n = len(to_label)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(disp[i][0] - disp[j][0], disp[i][1] - disp[j][1]) < min_px:
                union(i, j)

    grouped = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)
    return list(grouped.values()), disp


def _seed_angles_for_groups(to_label, groups, disp, y_lim, ax):
    """Outward angles per label; crowded screen groups get evenly spaced fan-outs."""
    plot_cx = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
    plot_cy = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
    plot_disp = ax.transData.transform((plot_cx, plot_cy))
    xlim = ax.get_xlim()
    ylim_ax = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim_ax[1] - ylim_ax[0]
    seeds = {}

    def _edge_biased_seed(i):
        pos = to_label[i][1]
        x_frac = (pos[0] - xlim[0]) / x_span if x_span > 0 else 0.5
        y_frac = (pos[1] - ylim_ax[0]) / y_span if y_span > 0 else 0.5
        if x_frac < 0.12:
            return 0.0
        if x_frac > 0.88:
            return np.pi
        if y_frac > 0.88:
            return -np.pi / 2
        if y_frac < 0.12:
            return np.pi / 2
        dx = disp[i][0] - plot_disp[0]
        dy = disp[i][1] - plot_disp[1]
        return float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 1.0 else np.pi / 2

    for group in groups:
        if len(group) == 1:
            i = group[0]
            im = to_label[i][1][1]
            if abs(im) > 0.35 * y_lim:
                seeds[i] = np.pi / 2 if im > 0 else -np.pi / 2
            else:
                seeds[i] = _edge_biased_seed(i)
            continue

        cx = float(np.mean([disp[i][0] for i in group]))
        cy = float(np.mean([disp[i][1] for i in group]))
        ranked = sorted(group, key=lambda i: np.arctan2(disp[i][1] - cy, disp[i][0] - cx))
        x_frac = float(np.mean([(to_label[i][1][0] - xlim[0]) / x_span for i in group]))
        for rank, i in enumerate(ranked):
            im = to_label[i][1][1]
            if abs(im) > 0.35 * y_lim:
                seeds[i] = np.pi / 2 if im > 0 else -np.pi / 2
            elif x_frac > 0.85:
                seeds[i] = np.pi + np.pi * (rank + 0.5) / len(group) - np.pi / 2
            elif x_frac < 0.15:
                seeds[i] = np.pi * (rank + 0.5) / len(group) - np.pi / 2
            else:
                seeds[i] = 2 * np.pi * rank / len(group) - np.pi / 2
    return seeds


def _pick_label_offset(ax, pos, label_text, fs_annot, seed_angle, placed_bboxes, inner_bbox):
    """Find a non-overlapping in-axes offset using estimated text boxes."""
    dpi = ax.figure.dpi
    width, height = _estimate_label_size_px(label_text, fs_annot, dpi)
    px_per_pt = dpi / 72.0
    anchor = ax.transData.transform(pos)

    best = None
    for radius in range(30, 125, 5):
        for k in range(20):
            angle = seed_angle + k * (np.pi / 10)
            ox = radius * np.cos(angle)
            oy = radius * np.sin(angle)
            cx = anchor[0] + ox * px_per_pt
            cy = anchor[1] + oy * px_per_pt
            bbox = _label_bbox_px(cx, cy, width, height)
            if not _bbox_inside(bbox, inner_bbox):
                continue
            if any(_bboxes_overlap(bbox, other) for other in placed_bboxes):
                continue
            return ox, oy, bbox

    for radius in range(30, 125, 5):
        for k in range(20):
            angle = seed_angle + k * (np.pi / 10)
            ox = radius * np.cos(angle)
            oy = radius * np.sin(angle)
            cx = anchor[0] + ox * px_per_pt
            cy = anchor[1] + oy * px_per_pt
            bbox = _label_bbox_px(cx, cy, width, height)
            if _bbox_inside(bbox, inner_bbox):
                return ox, oy, bbox

    ox = 45 * np.cos(seed_angle)
    oy = 45 * np.sin(seed_angle)
    cx = anchor[0] + ox * px_per_pt
    cy = anchor[1] + oy * px_per_pt
    return ox, oy, _label_bbox_px(cx, cy, width, height)


def _place_eig_annotations(ax, to_label, theme, fs_annot, y_lim):
    """Place labels using one draw + screen-space overlap checks."""
    arrowprops = dict(
        arrowstyle='-',
        color=theme['zero_line'],
        lw=0.55,
        shrinkA=4,
        shrinkB=4,
    )
    groups, disp = _screen_label_groups(ax, to_label)
    seeds = _seed_angles_for_groups(to_label, groups, disp, y_lim, ax)
    renderer = ax.figure.canvas.get_renderer()
    inner_bbox = _axes_inner_bbox_px(ax, renderer)

    # Place isolated / distant modes first; crowded groups last.
    order = sorted(
        range(len(to_label)),
        key=lambda i: (
            max(
                np.hypot(disp[i][0] - disp[j][0], disp[i][1] - disp[j][1])
                for j in range(len(to_label))
                if j != i
            ),
            i,
        ),
        reverse=True,
    )

    placed_bboxes = []
    for i in order:
        lam, pos = to_label[i]
        label_text = _eig_label(lam)
        ox, oy, bbox = _pick_label_offset(
            ax, pos, label_text, fs_annot, seeds[i], placed_bboxes, inner_bbox
        )
        placed_bboxes.append(bbox)
        ax.annotate(
            label_text,
            pos,
            xytext=(ox, oy),
            textcoords='offset points',
            fontsize=fs_annot,
            color=theme['text'],
            linespacing=1.25,
            ha='center',
            va='center',
            arrowprops=arrowprops,
            zorder=4,
            annotation_clip=False,
        )


def plot_eigs_thesis(
    eigs,
    ax=None,
    annotate=True,
    figsize=None,
    colors=None,
    print_ready=True,
    save_path=None,
):
    """S-plane eigenvalue plot styled for thesis figures (standard axes for cross-plot comparison)."""
    theme = colors or THESIS_EIG_COLORS
    eigs = np.asarray(eigs)
    if print_ready:
        figsize = figsize or (7.5, 5.4)
        marker_s = 200
        fs_label, fs_tick, fs_annot = 13, 12, 11
    else:
        figsize = figsize or (6.5, 5.0)
        marker_s = 120
        fs_label, fs_tick, fs_annot = 11, 10, 9

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    else:
        fig = ax.figure

    ax.set_facecolor('white')
    ax.scatter(
        eigs.real,
        eigs.imag,
        s=marker_s,
        c=theme['marker'],
        edgecolors=theme['marker_edge'],
        linewidths=1.4,
        zorder=3,
    )
    ax.axvline(0, color=theme['zero_line'], linewidth=0.75, linestyle='--', zorder=1)
    ax.axhline(0, color=theme['zero_line'], linewidth=0.75, linestyle='--', zorder=1)
    ax.grid(True, color=theme['grid'], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlabel('Real part (1/s)', fontsize=fs_label, color=theme['text'])
    ax.set_ylabel('Imaginary part (rad/s)', fontsize=fs_label, color=theme['text'])
    ax.tick_params(labelsize=fs_tick, colors=theme['axis'], width=0.6, length=4)
    for spine in ax.spines.values():
        spine.set_color(theme['axis'])
        spine.set_linewidth(0.65)

    re_min, re_max = eigs.real.min(), eigs.real.max()
    x_span = max(re_max - re_min, 1.0)
    x_pad = 0.04 * x_span
    ax.set_xlim(re_min - x_pad, max(re_max + x_pad, x_pad))

    y_peak = np.max(np.abs(eigs.imag))
    y_lim = max(1.5 * y_peak, 0.02 * x_span)
    ax.set_ylim(-y_lim, y_lim)

    for fmt in (ax.xaxis, ax.yaxis):
        fmt.set_major_formatter(ScalarFormatter(useOffset=False))
        fmt.get_major_formatter().set_scientific(False)

    fig.tight_layout()

    if annotate:
        to_label = _eigs_for_annotation(eigs, x_span, y_lim)
        _place_eig_annotations(ax, to_label, theme, fs_annot, y_lim)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if str(save_path).lower().endswith('.png') else None)
    return fig, ax


def plot_eigs(eigs):
    fig, ax = plt.subplots(1)
    sc = ax.scatter(eigs.real, eigs.imag)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = '{:.2f} Hz\n{:.2f}%'.format(pos[1] / (2 * np.pi), -100 * pos[0] / np.sqrt(sum(pos ** 2)))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('C0')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def phasor(vec, start=0j, ax=None, **kwargs):

    if not ax:
        fig, ax = plt.subplots(1, subplot_kw=dict(aspect=1))
    return ax.annotate('',
                       xy=(vec.real + start.real, vec.imag + start.imag),
                       xytext=(start.real, start.imag),
                       arrowprops=dict(arrowstyle='->', **kwargs),
                       annotation_clip=False)


def plot_mode_shape(mode_shape, ax=None, normalize=False, xy0=np.empty(0), linewidth=2, auto_lim=False, colors=plt.colormaps['Set1']):

    if not ax:
        ax = plt.subplot(111, projection='polar')
    if auto_lim:
        ax.set_rlim(0, max(abs(mode_shape)))

    if xy0.shape == (0,):
        xy0 = np.zeros_like(mode_shape)
    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
    ax.axes.get_yaxis().set_major_formatter(NullFormatter())
    ax.grid(color=[0.85, 0.85, 0.85])
    # f_txt = ax.set_xlabel('f={0:.2f}'.format(f), color=cluster_color_list(), weight='bold', family='Times New Roman', )

    if normalize:
        mode_shape_max = mode_shape[np.argmax(np.abs(mode_shape))]
        if abs(mode_shape_max) > 0:
            mode_shape = mode_shape * np.exp(-1j * np.angle(mode_shape_max)) / np.abs(mode_shape_max)

    pl = []
    for i, (vec, xy0_) in enumerate(zip(mode_shape, xy0)):
        pl.append(ax.annotate("",
                              xy=(np.angle(vec), np.abs(vec)),
                              xytext=(np.angle(xy0_), np.abs(xy0_)),
                              arrowprops=dict(arrowstyle="->",
                                              #linewidth=linewidth,
                                              #linestyle=style_,
                                              color=colors(i),
                                              )))  # , headwidth=1, headlength = 1))

    return pl
