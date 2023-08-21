import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from typing import Literal

from shap_plots.utils.helper_funcs import format_value, safe_isinstance
from shap_plots.utils import colors
from shap_plots.utils.labels import labels
from shap_plots.utils.logit_transformation import shap_transform_scale


# TODO: If we make a JS version of this plot then we could let users click on a bar and then see the dependence
# plot that is associated with that feature get overlaid on the plot...it would quickly allow users to answer
# why a feature is pushing down or up. Perhaps the best way to do this would be with an ICE plot hanging off
# of the bar...
def waterfall(
        shap_values,
        predicted_probability: float,
        max_display=10,
        show=True,
        link: Literal['identity', 'logit'] = 'logit',
        global_renderer: Literal['matplotlib', 'plotly'] = 'plotly'
):
    """Plots an explanation of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features.

    Features are sorted by the magnitude of their SHAP values with the smallest
    magnitude features grouped together at the bottom of the plot when the number of
    features in the models exceeds the ``max_display`` parameter.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional :class:`.Explanation` object that contains the feature values and SHAP values to plot.

    max_display : str
        The maximum number of features to plot (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it
        has been created.

    Examples
    --------

    See `waterfall plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html>`_.

    """

    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed! Perhaps try "
            "`shap.plots.waterfall(shap_values[0])` or for multi-output models, "
            "try `shap.plots.waterfall(shap_values[0, 0])`."
        )
        raise ValueError(emsg)

    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    if link == 'identity':
        values = shap_values.values
        base_values = float(shap_values.base_values)
    elif link == 'logit':
        shap_transformed = shap_transform_scale(shap_values, predicted_probability)
        values = shap_transformed.values
        base_values = float(shap_transformed.base_values)

    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = format_value(float(features[order[i]]), "%0.03f") + " = " + feature_names[order[i]]
            else:
                yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
             left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
             left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    if global_renderer == 'matplotlib':
        # draw the positive arrows
        for i in range(len(pos_inds)):
            dist = pos_widths[i]
            arrow_obj = plt.arrow(
                pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
                head_length=min(dist, hl_scaled),
                color=colors.red_rgb, width=bar_width,
                head_width=bar_width,
            )

            if pos_low is not None and i < len(pos_low):
                plt.errorbar(
                    pos_lefts[i] + pos_widths[i], pos_inds[i],
                    xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                    ecolor=colors.light_red_rgb,
                )

            txt_obj = plt.text(
                pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='center', verticalalignment='center', color="white",
                fontsize=12,
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            # if the text overflows the arrow then draw it after the arrow
            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

                txt_obj = plt.text(
                    pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                    horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                    fontsize=12,
                )

        # draw the negative arrows
        for i in range(len(neg_inds)):
            dist = neg_widths[i]

            arrow_obj = plt.arrow(
                neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
                head_length=min(-dist, hl_scaled),
                color=colors.blue_rgb, width=bar_width,
                head_width=bar_width,
            )

            if neg_low is not None and i < len(neg_low):
                plt.errorbar(
                    neg_lefts[i] + neg_widths[i], neg_inds[i],
                    xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                    ecolor=colors.light_blue_rgb,
                )

            txt_obj = plt.text(
                neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='center', verticalalignment='center', color="white",
                fontsize=12,
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            # if the text overflows the arrow then draw it after the arrow
            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

                txt_obj = plt.text(
                    neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                    horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                    fontsize=12,
                )

        # draw the y-ticks twice, once in gray and then again with just the feature names in black
        # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
        ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8)
        plt.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)

        # put horizontal lines for each feature row
        # for i in range(num_features):
        #     plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        # mark the prior expected value and the model prediction
        # plt.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        fx = base_values + values.sum()
        given_prediction_line = plt.axvline(fx, 0, 1, color=colors.light_red_rgb, linestyle="--", linewidth=0.7, zorder=-1)
        plt.legend(labels=['Given Prediction Confidence'], handles=[given_prediction_line])

        # clean up the main axis
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        ax.tick_params(labelsize=13)
        #plt.xlabel("\nModel output", fontsize=12)

        # draw the E[f(X)] tick mark
        # xmin, xmax = ax.get_xlim()
        # ax2 = ax.twiny()
        # ax2.set_xlim(xmin, xmax)
        # ax2.set_xticks([base_values, base_values+1e-8])  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
        # ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['left'].set_visible(False)

        # # draw the f(x) tick mark
        # ax3 = ax2.twiny()
        # ax3.set_xlim(xmin, xmax)
        # # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
        # ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
        # # ax3.set_xticklabels(["$f(x)$", "$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
        # tick_labels = ax3.xaxis.get_majorticklabels()
        # tick_labels[0].set_transform(tick_labels[0].get_transform(
        # ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
        # tick_labels[1].set_transform(tick_labels[1].get_transform(
        # ) + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
        # tick_labels[1].set_color("#999999")
        # ax3.spines['right'].set_visible(False)
        # ax3.spines['top'].set_visible(False)
        # ax3.spines['left'].set_visible(False)

        # adjust the position of the E[f(X)] = x.xx label
        # tick_labels = ax2.xaxis.get_majorticklabels()
        # tick_labels[0].set_transform(tick_labels[0].get_transform(
        # ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
        # tick_labels[1].set_transform(tick_labels[1].get_transform(
        # ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))

        # tick_labels[1].set_color("#999999")

        # color the y tick labels that have the feature values as gray
        # (these fall behind the black ones with just the feature name)
        tick_labels = ax.yaxis.get_majorticklabels()
        for i in range(num_features):
            tick_labels[i].set_color("#999999")

    elif global_renderer == 'plotly':
        df_pos = pd.DataFrame()
        df_pos['widths'] = pos_widths
        df_pos['inds'] = pos_inds
        df_pos['dir'] = 'pos'

        df_neg = pd.DataFrame()
        df_neg['widths'] = neg_widths
        df_neg['inds'] = neg_inds
        df_neg['dir'] = 'neg'

        df = pd.concat([df_pos, df_neg], axis=0)
        df['widths'] = df.apply(lambda x: -x['widths'] if x['inds']=='neg' else x['widths'], axis=1)
        df['inds'] = df['inds']+1
        df = pd.concat([df, pd.DataFrame({'widths': [0.3310746], 'inds': [0], 'dir': ['pos']})], axis=0)
        df.sort_values('inds', ascending=True, inplace=True)
        yticklabels.pop(-1)
        yticklabels.insert(0, '')
        df['ylabels'] = yticklabels
        df['widths'] = np.round(df['widths'], 2)

        red_rgb = f'rgb({colors.red_rgb[0]}, {colors.red_rgb[1]}, {colors.red_rgb[2]})'
        blue_rgb = f'rgb({colors.blue_rgb[0]}, {colors.blue_rgb[1]}, {colors.blue_rgb[2]})'
        plotly_fig = go.Figure()
        plotly_fig.add_trace(
            go.Waterfall(
                orientation='h',
                name='Features Contribution',
                measure=['absolute']+['relative' for _ in range(df.shape[0]-1)],
                y=df['ylabels'],
                x=df['widths'],
                connector = {
                    'mode': 'between',
                    'line': {
                        'width': 4,
                        'color': 'rgb(0, 0, 0)',
                        'dash': 'solid'
                    }
                },
                totals = {
                    'marker': {
                        'color': 'rgba(0, 0, 0, 0)'
                    }
                },
                decreasing={
                    'marker': {
                        'color': blue_rgb,
                        'line': {
                            'color': blue_rgb
                        }
                    }
                },
                increasing={
                    'marker': {
                        'color': red_rgb,
                        'line': {
                            'color': red_rgb
                        }
                    }
                }
            )
        )
        plotly_fig.add_trace(
            go.Scatter(
                x=[np.round(df['widths'].sum(), 2), np.round(df['widths'].sum(), 2)],
                name='Predicted Probability',
                mode='lines',
                y=[
                    df[df['inds']==df['inds'].min()]['ylabels'].values[0],
                    df[df['inds']==df['inds'].max()]['ylabels'].values[0]
                ],
                line={'dash': 'dot'}
            )
        )
        baseline_label = ['']
        features_labels = df.query('inds > 0')['widths'].tolist()
        features_labels = [f'+{label}' if label > 0 else label for label in features_labels]
        data_labels = baseline_label+features_labels
        plotly_fig.data[0].text = data_labels
        plotly_fig.update_layout(
            title = 'Prediction Flow Chart',
            height=800,
            width=1200,
            **{
                'yaxis': {'showgrid': False},
                'xaxis': {'showgrid': False, 'range': [plt.xlim()[0], plt.xlim()[1]]},
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        )

    if show:
        if global_renderer == 'matplotlib':
            plt.show()
        elif global_renderer == 'plotly':
            plt.close()
            return plotly_fig.show()
    else:
        if global_renderer == 'matplotlib':
            return plt.gcf()
        elif global_renderer == 'plotly':
            plt.close()
            return plotly_fig