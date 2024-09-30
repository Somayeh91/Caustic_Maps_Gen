import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from more_info import best_AD_models_info
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from maps_util import read_map_meta
import sys
# import corner
import matplotlib as mpl


def plot_example_conv(map1, map2, direc):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    min_, max_ = np.min(map1), np.max(map1)
    ax[0].imshow(map1, cmap='gray',
                 norm=LogNorm(vmin=min_, vmax=max_))
    ax[0].set_title('Original map', size=20)
    ax[0].axis('off')

    ax[1].imshow(map2, cmap='gray',
                 norm=LogNorm(vmin=min_, vmax=max_))
    ax[1].set_title('Convolved map', size=20)
    ax[1].axis('off')
    fig.savefig(direc)
    plt.close()


def plot_example_lc(ID, maps, lc_set, origin, rsrc, direc):
    t1 = lc_set[0][0]
    lc1 = lc_set[0][1]
    t2 = lc_set[1][0]
    lc2 = lc_set[1][1]
    origin1 = origin[0]
    origin2 = origin[1]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    min_, max_ = np.min(maps[0]), np.max(maps[0])
    _, k, g, s, _, _ = read_map_meta(ID)

    ax[0, 0].imshow(maps[0], cmap='gray',
                    norm=LogNorm(vmin=min_, vmax=max_))
    ax[0, 0].plot([origin1[0, 0], origin1[1, 1]], [origin1[0, 1], origin1[1, 0]], 'r')
    ax[0, 0].axis('off')
    circ = Circle((origin1[0, 0], origin1[0, 1]), rsrc / 0.05, edgecolor='red', facecolor='None')
    ax[0, 0].add_patch(circ)
    ax[0, 0].text(0.05, 0.05, "$\kappa = %.1f, \it \gamma = %.1f, \it {s} = %.1f $" % (k, g, s),
                  color='yellow', size=20,
                  transform=ax[0, 0].transAxes)
    ax[0, 0].invert_yaxis()

    ax[1, 0].imshow(maps[1], cmap='gray',
                    norm=LogNorm(vmin=min_, vmax=max_))
    ax[1, 0].plot([origin2[0, 0], origin2[1, 1]], [origin2[0, 1], origin2[1, 0]], 'r')
    ax[1, 0].axis('off')
    circ = Circle((origin2[0, 0], origin2[0, 1]), rsrc / 0.05, edgecolor='red', facecolor='None')
    ax[1, 0].add_patch(circ)
    # ax[1,0].text(0.05,0.05,"$\kappa = %.1f, \it \gamma = %.1f, \it {s} = %.1f $"%(k, g, s),
    #                      color='yellow', size = 15,
    #                     transform = ax[1,0].transAxes)
    ax[1, 0].text(0.05, 0.05,
                  "Convolved with rsrc=%0.1f" % rsrc,
                  color='yellow',
                  size=20,
                  transform=ax[1, 0].transAxes)
    ax[1, 0].invert_yaxis()

    ax[0, 1].plot(t1 / 365, lc1, '.', color='#502db3')
    ax[1, 1].plot(t2 / 365, lc2, '.', color='#502db3')
    ax[0, 1].grid(which='major', axis='x', linestyle='--')
    ax[1, 1].grid(which='major', axis='x', linestyle='--')
    ax[1, 1].set_xlabel('Time (years)', size=15)

    fig.tight_layout()
    fig.savefig(direc)
    plt.close()


def metric_comparison_plot(metric, metric_name, imgs, plots_dir, mode='no_unc'):
    plt.figure()
    data = best_AD_models_info['data']
    job_names = best_AD_models_info['job_names']
    columns = best_AD_models_info['columns']
    rows = best_AD_models_info['rows']

    if len(metric) != len(job_names):
        print('Check the model IDs.')
        sys.exit()

    n_rows = len(data)

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        cell_text.append([x for x in data[row]])

    # Reverse colors and text labels to display the last value at the top.
    if mode == 'no_unc':
        plt.plot(columns, metric[:, 0], marker='*', markersize=10, color='#502db3')
    elif mode == 'unc':
        plt.errorbar(columns,
                     metric[:, 0],
                     yerr=metric[:, 1],
                     fmt='o',
                     color='#502db3')
    # cell_text = np.array(cell_text).T.tolist()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='bottom',
                          cellLoc='center')

    for r in range(-1, len(columns)):
        if r > -1:
            the_table[0, r].set_height(0.1)
        for e in range(1, len(rows) + 1):
            the_table[e, r].set_height(0.08)
    the_table.set_fontsize(10)
    # the_table.set_height(0.08)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    lcs = metric[:, 0]

    unit = (np.max(lcs) - np.min(lcs)) / 50

    img_pos = np.array([[0.5, lcs[0]],
                        [1.6, lcs[1] - 5 * unit],
                        [2, lcs[2] + 7 * unit],
                        [3, lcs[3] + 7 * unit],
                        [4, lcs[4] + 7 * unit],
                        [4.8, lcs[5] + 10 * unit]])
    for i in range(1, 7):
        imagebox = OffsetImage(imgs[i], zoom=0.2, cmap='gray',
                               norm=LogNorm(vmin=np.min(imgs[0]), vmax=np.max(imgs[0])))
        # Annotation box for solar pv logo
        # Container for the imagebox referring to a specific position *xy*.
        ab = AnnotationBbox(imagebox, (img_pos[i - 1][0], img_pos[i - 1][1]), frameon=False)
        plt.gca().add_artist(ab)

    imagebox = OffsetImage(imgs[0], zoom=0.2, cmap='gray', norm=LogNorm())
    ab = AnnotationBbox(imagebox, (4.8, np.max(lcs) - 3 * unit), frameon=False)
    plt.gca().add_artist(ab)
    plt.text(4.4, np.max(lcs) - 11 * unit, 'True Map',
             color='black', size=10)
    # plt.ylabel(f"Loss in ${value_increment}'s")
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.ylabel('%s score' % metric_name)
    plt.title('%s Score per trial run' % metric_name)
    # fig.tight_layout()
    plt.savefig(plots_dir + '%s_scores_per_job_l2.pdf' % metric_name,
                bbox_inches='tight')


def fit_lc_metric_exmple(num_rows, with_maps=False):
    fig = plt.figure(figsize=(6, 12))
    if with_maps:
        gs = fig.add_gridspec(num_rows, 1)

        axs0 = []
        axs1 = []
        axs2 = []
        axs3 = []
        for row in range(num_rows):
            axs0.append(fig.add_subplot(gs[row, 0]))
            axs1.append(fig.add_subplot(gs[row, 1]))
            axs2.append(fig.add_subplot(gs[row, 2]))
            axs3.append(fig.add_subplot(gs[row, 3]))

        return fig, [axs0, axs1, axs2, axs3]
    else:
        gs = fig.add_gridspec(num_rows, 4, height_ratios=(1, 1, 1, 2))

        axs0 = []
        axs1 = []
        axs2 = []
        axs3 = []
        for row in range(num_rows):
            axs0.append(fig.add_subplot(gs[row, 0]))
            axs1.append(fig.add_subplot(gs[row, 1]))
            axs2.append(fig.add_subplot(gs[row, 2]))
            axs3.append(fig.add_subplot(gs[row, 3]))

        return fig, [axs0, axs1, axs2, axs3]


def metric_comparison_per_rsrc_plot(rsrcs, metric, metric_name, plots_dir, mode='no_unc'):
    data = best_AD_models_info['data']
    job_names = best_AD_models_info['job_names']
    columns = best_AD_models_info['columns']
    rows = best_AD_models_info['rows']

    source_sizes = rsrcs
    colors = ['#e51f00', '#ffaa00', '#59b359', '#502db3']

    n_rows = len(data)

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        cell_text.append([x for x in data[row]])

    # Reverse colors and text labels to display the last value at the top.
    for j, rsrc in enumerate(source_sizes):
        if mode == 'no_unc':
            plt.plot(columns, metric[:, j], marker='*',
                     markersize=10, color=colors[j],
                     label='$r_{source}= %.1f$' % rsrc)
        elif mode == 'unc':
            plt.errorbar(columns,
                         metric[:, 2 * j],
                         yerr=metric[:, 2 * j + 1],
                         fmt='o',
                         color=colors[j],
                         label='$r_{source}= %.1f$' % rsrc)

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='bottom',
                          cellLoc='center')

    for r in range(-1, len(columns)):
        if r > -1:
            the_table[0, r].set_height(0.1)
        for e in range(1, len(rows) + 1):
            the_table[e, r].set_height(0.08)
    the_table.set_fontsize(10)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.xticks([])
    plt.ylabel('%s score' % metric_name)
    plt.title('%s Score per trial run' % metric_name)
    plt.legend()
    plt.savefig(plots_dir + '%s_scores_per_job_per_rsrc.pdf' % metric_name,
                bbox_inches='tight')


def kgs_lc_distance_metric(kgs,
                           lc_metric,
                           kgs_ref,
                           kgs_closest,
                           output_direc,
                           plot_name,
                           axis='kg',
                           type_='scatter'
                           ):
    print('Plotting %s %s plot... ' % (type_, axis))
    plt.figure()
    axis_labels = ["$\kappa$", "$\gamma$", "$s$"]
    title_labels = {'k': 0, 'g': 1, 's': 2}
    plot_x = title_labels[axis[0]]
    plot_y = title_labels[axis[1]]
    w = 0.05

    plt.figure()
    colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
    # norm = mpl.colors.LogNorm(vmin=np.min(lc_metric),
    # 					      vmax=np.max(lc_metric))
    # colors = colormap(lc_metric)
    if type_ == 'scatter':
        sc = plt.scatter(kgs[:, plot_x],
                         kgs[:, plot_y],
                         c=lc_metric,
                         cmap=colormap,
                         s=25)
    elif type_ == 'hist2d':
        sc = plt.hist2d(kgs[:, plot_x],
                        kgs[:, plot_y],
                        bins=[np.arange(np.min(kgs[:, plot_x]), np.max(kgs[:, plot_x]) + w, w),
                              np.arange(np.min(kgs[:, plot_y]), np.max(kgs[:, plot_y]) + w, w)],
                        weights=lc_metric)

    print(kgs[:, 0])
    print(kgs[:, 1])
    print(kgs[:, 2])
    # print(lc_metric)
    if hasattr(kgs_ref, "__len__"):
        plt.scatter([kgs_ref[plot_x]], [kgs_ref[plot_y]], c='SteelBlue', marker='x', s=40, label='True %s' % type_)
        plt.scatter([kgs_closest[plot_x]], [kgs_closest[plot_y]], c='IndianRed', marker='x', s=40,
                    label='Predicted %s' % type_)
        plt.legend()
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=np.min(lc_metric),
                vmax=np.max(lc_metric))
    cbar = plt.colorbar(sm)
    cbar.set_label('Log lc distance Metric', size=18)

    plt.xlabel(axis_labels[plot_x])
    plt.ylabel(axis_labels[plot_y])
    plt.savefig(output_direc + type_ + '_' + plot_name)
    plt.close()


def plot_kgs_lc_distance_metric(kgs, ID_ref,
                                fit_lc_metrics,
                                fit_lc_per_map_metrics,
                                output_direc,
                                plot_axis='kg',
                                conv_mode=0):
    kgs_ref = kgs[ID_ref]
    print('Referenc kgs is %.1f, %.1f, %.1f' % (kgs_ref[0], kgs_ref[1], kgs_ref[2]))
    metrics_tmp = np.log10((fit_lc_metrics / fit_lc_per_map_metrics))
    metrics_per_map_tmp = (fit_lc_per_map_metrics / fit_lc_per_map_metrics)
    k_closest = kgs[:, 0][np.argmin(metrics_tmp)]
    g_closest = kgs[:, 1][np.argmin(metrics_tmp)]
    s_closest = kgs[:, 2][np.argmin(metrics_tmp)]
    kgs_closest = np.array([k_closest,
                            g_closest,
                            s_closest])
    title = plot_axis + '_fitted_true_lc_rsrc%.1f.png' % conv_mode
    title_per_map = plot_axis + '_fitted_true_lc_per_map_rsrc%.1f.png' % conv_mode

    print('Title is %s' % title)
    kgs_lc_distance_metric(kgs=kgs,
                           lc_metric=metrics_tmp,
                           kgs_ref=kgs_ref,
                           kgs_closest=kgs_closest,
                           output_direc=output_direc,
                           plot_name=title,
                           axis=plot_axis,
                           type_='scatter'
                           )
    kgs_lc_distance_metric(kgs=kgs,
                           lc_metric=metrics_per_map_tmp,
                           kgs_ref=np.nan,
                           kgs_closest=kgs_closest,
                           output_direc=output_direc,
                           plot_name=title_per_map,
                           axis=plot_axis,
                           type_='scatter'
                           )


def fig_loss(model_history):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(np.array(model_history['val_loss']), label='Validation loss')
    plt.plot(np.array(model_history['loss']), label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    return fig


def compareinout(output, testimg, dim, output_dir, model_design_key, ID):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    ax.imshow(testimg.reshape((dim, dim)), cmap='gray', norm=LogNorm())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = fig.add_subplot(132)
    ax.imshow(output.reshape((dim, dim)), cmap='gray', norm=LogNorm())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.savefig(output_dir + 'model_%s_test_%i.png' % (model_design_key, ID), bbox_inches='tight')


def compareinout_bt_to_kgs(output_kgs, input_kgs, output_dir):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(input_kgs[:, 0], output_kgs[:, 0], 'o', label=f'$\kappa$')
    plt.plot(input_kgs[:, 0], input_kgs[:, 0], '-')
    plt.plot(input_kgs[:, 1], output_kgs[:, 1], 'o', label=f'$\gamma$')
    plt.plot(input_kgs[:, 2], output_kgs[:, 2], 'o', label='s')

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.legend()
    fig.savefig(output_dir + 'kgs_pred_vs_true.png', bbox_inches='tight')


def analyze_bt_to_kgs_results(output_kgs, input_kgs, output_dir):
    k_err = (input_kgs[:, 0] - output_kgs[:, 0])
    g_err = (input_kgs[:, 1] - output_kgs[:, 1])
    s_err = (input_kgs[:, 2] - output_kgs[:, 2])
    errors = np.stack((k_err, g_err, s_err), axis=1)
    figure = corner.corner(errors)
    figure.savefig(output_dir + 'corner_bt_to_kgs.png', bbox_inches='tight')