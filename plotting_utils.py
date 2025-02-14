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
    """
    Plots and saves a side-by-side comparison of an original and a convolved map.

    Parameters:
    -----------
    map1 : numpy.ndarray
        The original map before convolution.
    map2 : numpy.ndarray
        The convolved map after processing.
    direc : str
        Path to save the generated figure.

    Returns:
    --------
    None
        The function saves the figure to `direc` and does not return any values.

    Notes:
    ------
    - Displays `map1` and `map2` side by side with a logarithmic color scale (`LogNorm`).
    - Uses grayscale (`cmap='gray'`) for visualization.
    - The title 'Original map' is assigned to `map1`, and 'Convolved map' to `map2`.
    - The figure is saved at the specified location and then closed to free memory.
    """

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
    """
    Plots and saves a visualization of microlensing maps and their corresponding light curves.

    Parameters:
    -----------
    ID : int or str
        Identifier of the microlensing map being plotted.
    maps : list of numpy.ndarray
        A list containing two maps:
        - maps[0]: The original map.
        - maps[1]: The convolved map.
    lc_set : list of tuples
        A list containing two tuples for light curves:
        - lc_set[0]: (time, light curve) for the original map.
        - lc_set[1]: (time, light curve) for the convolved map.
    origin : list of numpy.ndarray
        List of arrays indicating the coordinates of the light curve extraction origins:
        - origin[0]: Origin for the original map.
        - origin[1]: Origin for the convolved map.
    rsrc : float
        Source radius used for convolution.
    direc : str
        Path to save the generated figure.

    Returns:
    --------
    None
        The function saves the figure to `direc` and does not return any values.

    Notes:
    ------
    - Reads map metadata (`k`, `g`, `s`) using `read_map_meta(ID)`.
    - Displays the original and convolved maps side by side with a logarithmic color scale (`LogNorm`).
    - Adds a red circle indicating the source radius `rsrc` on both maps.
    - Plots the light curves extracted from both maps.
    - The x-axis of light curves is converted to years (`time / 365`).
    - The figure is saved at `direc` and closed to free memory.
    """
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
    """
       Plots and saves a comparison of model evaluation metrics, optionally with uncertainty bars, and overlays
       example images for visual reference.

       Parameters:
       -----------
       metric : numpy.ndarray
           A 2D array containing metric values for different models.
           - `metric[:, 0]`: Primary metric values.
           - `metric[:, 1]` (optional): Uncertainty values (used if `mode='unc'`).
       metric_name : str
           Name of the metric being plotted (e.g., "FID", "SSM", "LC_Sim").
       imgs : list of numpy.ndarray
           List of images corresponding to different models.
           - `imgs[0]`: The true/reference map.
           - `imgs[1:]`: Generated or reconstructed maps for different models.
       plots_dir : str
           Directory where the plot should be saved.
       mode : str, default='no_unc'
           Determines whether to include uncertainty bars:
           - 'no_unc': Plots only the metric values.
           - 'unc': Includes error bars using `metric[:, 1]`.

       Returns:
       --------
       None
           The function saves the figure as a PDF and does not return any values.

       Notes:
       ------
       - Extracts metadata (`data`, `job_names`, `columns`, `rows`) from `best_AD_models_info`.
       - Plots metric scores for different models, using either markers (`'*'`) or error bars (`'o'`).
       - Displays a table at the bottom summarizing model results.
       - Overlays example images at specific data points for visual comparison.
       - Uses logarithmic normalization (`LogNorm`) for image display.
       - Saves the final plot as a PDF file in `plots_dir`.
       """
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
    """
        Creates a figure with a grid layout for visualizing light curve fitting and metric evaluation examples.

        Parameters:
        -----------
        num_rows : int
            The number of rows to include in the figure.
        with_maps : bool, default=False
            Determines the layout structure:
            - If `True`, creates a single-column grid layout (one subplot per row).
            - If `False`, creates a four-column grid layout with height ratios (1,1,1,2).

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object.
        axs : list of lists
            A list containing four lists of subplot axes:
            - axs[0]: List of subplots for the first column.
            - axs[1]: List of subplots for the second column.
            - axs[2]: List of subplots for the third column.
            - axs[3]: List of subplots for the fourth column.

        Notes:
        ------
        - Uses `gridspec` to define a flexible grid layout.
        - If `with_maps` is `True`, only a single-column grid is created.
        - If `with_maps` is `False`, a four-column grid is created with a custom height ratio.
        - Subplot axes are stored in separate lists for easy access.
        """
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
    """
    Plots and saves a comparison of model evaluation metrics across different source sizes (rsrc).

    Parameters:
    -----------
    rsrcs : list or numpy.ndarray
        List of source sizes (rsrc) used for comparison.
    metric : numpy.ndarray
        A 2D array containing metric values for different models and source sizes.
        - If `mode='no_unc'`, `metric[:, j]` contains the metric values for `rsrcs[j]`.
        - If `mode='unc'`, `metric[:, 2*j]` contains the metric values, and `metric[:, 2*j+1]` contains uncertainties.
    metric_name : str
        Name of the metric being plotted (e.g., "FID", "SSM", "LC_Sim").
    plots_dir : str
        Directory where the plot should be saved.
    mode : str, default='no_unc'
        Determines whether to include uncertainty bars:
        - 'no_unc': Plots only the metric values using markers.
        - 'unc': Includes error bars for uncertainty representation.

    Returns:
    --------
    None
        The function saves the figure as a PDF and does not return any values.

    Notes:
    ------
    - Extracts metadata (`data`, `job_names`, `columns`, `rows`) from `best_AD_models_info`.
    - Plots metric scores for different models as a function of source size (`rsrc`).
    - Uses different colors to distinguish between source sizes.
    - Displays a table at the bottom summarizing model results.
    - Adjusts subplot layout to fit the table and avoids overlapping elements.
    - Saves the final plot as a PDF file in `plots_dir`.
    """
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
    """
        Plots and saves a visualization of kappa-gamma-shear (kgs) parameters against light curve distance metrics.

        Parameters:
        -----------
        kgs : numpy.ndarray
            A 2D array containing kappa (κ), gamma (γ), and shear (s) values for different data points.
        lc_metric : numpy.ndarray
            An array containing the light curve distance metric corresponding to each kgs data point.
        kgs_ref : numpy.ndarray or None
            Reference (true) kgs values, used for marking on the plot.
        kgs_closest : numpy.ndarray or None
            Predicted kgs values, used for marking on the plot.
        output_direc : str
            Directory path where the plot should be saved.
        plot_name : str
            Name of the plot file to be saved.
        axis : str, default='kg'
            Specifies which kgs parameters to plot on the x and y axes:
            - 'kg': κ (kappa) vs. γ (gamma).
            - 'ks': κ (kappa) vs. s (shear).
            - 'gs': γ (gamma) vs. s (shear).
        type_ : str, default='scatter'
            The type of plot to generate:
            - 'scatter': Creates a scatter plot where points are colored by `lc_metric`.
            - 'hist2d': Creates a 2D histogram weighted by `lc_metric`.

        Returns:
        --------
        None
            The function saves the figure to `output_direc` and does not return any values.

        Notes:
        ------
        - The color scale is determined using the 'viridis' colormap.
        - If `kgs_ref` and `kgs_closest` are provided, they are plotted as markers:
            - `kgs_ref`: SteelBlue 'x' marker (True value).
            - `kgs_closest`: IndianRed 'x' marker (Predicted value).
        - A colorbar is added to indicate the light curve distance metric.
        - The function automatically determines which kgs parameters to plot based on `axis`.
        - The figure is saved with the provided `plot_name` in `output_direc` and then closed to free memory.
        """
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
    """
    Computes the reference and closest kappa-gamma-shear (kgs) parameters based on light curve distance metrics
    and generates scatter plots for visualization.

    Parameters:
    -----------
    kgs : numpy.ndarray
        A 2D array of kappa (κ), gamma (γ), and shear (s) values for different data points.
    fit_lc_metrics : numpy.ndarray
        Array of fitted light curve metrics.
    fit_lc_per_map_metrics : numpy.ndarray
        Array of light curve metrics computed per map.
    ID_ref : int
        Index of the reference kgs value.
    output_direc : str
        Directory where the plots should be saved.
    plot_axis : str
        Specifies the kgs parameters to plot (e.g., 'kg', 'ks', 'gs').
    conv_mode : float
        The convolution mode value, used to name the saved plot files.

    Returns:
    --------
    None
        The function generates and saves two scatter plots but does not return any values.

    Notes:
    ------
    - Extracts the reference kgs parameters using `ID_ref` and prints them.
    - Computes the log-transformed light curve distance metric (`metrics_tmp`).
    - Determines the kgs parameters closest to the minimum metric value (`kgs_closest`).
    - Constructs plot titles using `plot_axis` and `conv_mode`.
    - Calls `kgs_lc_distance_metric` to generate:
      1. A scatter plot comparing kgs to `metrics_tmp` with `kgs_ref` and `kgs_closest` marked.
      2. A scatter plot comparing kgs to `metrics_per_map_tmp`, without marking `kgs_ref`.
    """
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
    """
    Plots and returns a figure displaying the training and validation loss over iterations.

    Parameters:
    -----------
    model_history : dict
        Dictionary containing the model's training history with keys such as:
        - 'loss': Training loss values over iterations.
        - 'val_loss' (optional): Validation loss values over iterations.

    Returns:
    --------
    fig : matplotlib.figure.Figure or None
        The generated loss plot figure if 'loss' is present in `model_history`.
        Returns `None` if 'loss' is not found in `model_history`.

    Notes:
    ------
    - If 'val_loss' is present, it is plotted alongside 'loss' for comparison.
    - The function labels the axes and includes a legend for clarity.
    - If 'loss' is missing from `model_history`, the function returns without plotting.
    """
    fig = plt.figure(figsize=(8, 8))
    if 'val_loss' in model_history.keys():
        plt.plot(np.array(model_history['val_loss']), label='Validation loss')
    if 'loss' in model_history.keys():
        plt.plot(np.array(model_history['loss']), label='loss')
    else:
        return
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    return fig


def compareinout(output, testimg, dim, output_dir, model_design_key, ID):
    """
        Compares the input test image with the model-generated output and saves the visualization.

        Parameters:
        -----------
        output : numpy.ndarray
            The reconstructed or generated output image from the model.
        testimg : numpy.ndarray
            The original test image.
        dim : int
            The dimension of the square images (both `testimg` and `output` should be `dim x dim`).
        output_dir : str
            Directory where the comparison figure should be saved.
        model_design_key : str
            Identifier for the model used in generating `output`, used in the saved filename.
        ID : int
            ID of the test sample, used in the saved filename.

        Returns:
        --------
        None
            The function saves the figure to `output_dir` and does not return any values.

        Notes:
        ------
        - Creates a side-by-side grayscale (`cmap='gray'`) comparison of the original (`testimg`) and the output (`output`).
        - Uses a logarithmic color scale (`LogNorm()`) for better visualization of image intensity variations.
        - Hides axis labels for clarity.
        - Saves the figure with a filename format: `'model_{model_design_key}_test_{ID}.png'` in `output_dir`.
        """
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
    """
        Compares the predicted kappa-gamma-shear (kgs) values with the true input values and saves the visualization.

        Parameters:
        -----------
        output_kgs : numpy.ndarray
            The predicted kgs values from the model, with shape (N, 3), where:
            - Column 0: Predicted kappa (κ).
            - Column 1: Predicted gamma (γ).
            - Column 2: Predicted shear (s).
        input_kgs : numpy.ndarray
            The true kgs values, with shape (N, 3), in the same order as `output_kgs`.
        output_dir : str
            Directory where the comparison plot should be saved.

        Returns:
        --------
        None
            The function saves the figure to `output_dir` and does not return any values.

        Notes:
        ------
        - Plots true vs. predicted values for κ, γ, and s.
        - Adds a reference line (`input_kgs[:, 0]` vs. itself) to indicate perfect predictions.
        - Includes axis labels and a legend for clarity.
        - Saves the figure as `'kgs_pred_vs_true.png'` in `output_dir`.
        """
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
    """
        Analyzes and visualizes the errors in predicting kappa-gamma-shear (kgs) parameters using a corner plot.

        Parameters:
        -----------
        output_kgs : numpy.ndarray
            The predicted kgs values from the model, with shape (N, 3), where:
            - Column 0: Predicted kappa (κ).
            - Column 1: Predicted gamma (γ).
            - Column 2: Predicted shear (s).
        input_kgs : numpy.ndarray
            The true kgs values, with shape (N, 3), in the same order as `output_kgs`.
        output_dir : str
            Directory where the error analysis plot should be saved.

        Returns:
        --------
        None
            The function saves the corner plot to `output_dir` and does not return any values.

        Notes:
        ------
        - Computes the prediction errors for κ, γ, and s:
            - `k_err = input_kgs[:, 0] - output_kgs[:, 0]`
            - `g_err = input_kgs[:, 1] - output_kgs[:, 1]`
            - `s_err = input_kgs[:, 2] - output_kgs[:, 2]`
        - Stacks these errors into a (N, 3) array and visualizes them using `corner.corner()`.
        - Saves the generated corner plot as `'corner_bt_to_kgs.png'` in `output_dir`.
        - The corner plot shows error distributions and correlations between κ, γ, and s errors.
        """
    k_err = (input_kgs[:, 0] - output_kgs[:, 0])
    g_err = (input_kgs[:, 1] - output_kgs[:, 1])
    s_err = (input_kgs[:, 2] - output_kgs[:, 2])
    errors = np.stack((k_err, g_err, s_err), axis=1)
    figure = corner.corner(errors)
    figure.savefig(output_dir + 'corner_bt_to_kgs.png', bbox_inches='tight')