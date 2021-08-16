import plotly.io as pio


def set_plot_appearances(graph_line_color='#323232', background_color='#0F0F0F', text_color='#DAD9D8',
                         default_plotly_template='plotly_dark'):

    pio.templates["enDAQ"] = pio.templates[default_plotly_template]

    # Line Colors
    colorway = ['#EE7F27', '#6914F0', '#2DB473', '#D72D2D', '#3764FF', '#FAC85F']
    colorbar = [[0.0, '#6914F0'],
                [0.2, '#3764FF'],
                [0.4, '#2DB473'],
                [0.6, '#FAC85F'],
                [0.8, '#EE7F27'],
                [1.0, '#D72D2D']]
    pio.templates["enDAQ"]['layout']['colorway'] = colorway
    pio.templates["enDAQ"]['layout']['colorscale']['sequential'] = colorbar
    pio.templates["enDAQ"]['layout']['colorscale']['sequentialminus'] = colorbar
    pio.templates["enDAQ"]['layout']['colorscale']['diverging'] = [[0.0, '#6914F0'],
                                                                   [0.5, '#f7f7f7'],
                                                                   [1.0, '#EE7F27']]
    plot_types = ['contour', 'heatmap', 'heatmapgl', 'histogram2d', 'histogram2dcontour', 'surface']
    for p in plot_types:
        pio.templates["enDAQ"]['data'][p][0].colorscale = colorbar

    # Text
    # dictionary = dict(font=dict(family="Open Sans", size=24, color=text_color))
    # pio.templates["enDAQ"]['layout']['annotations'] = [(k, v) for k, v in dictionary.items()]
    pio.templates["enDAQ"]['layout']['font'] = dict(family="Open Sans", size=16, color=text_color)
    pio.templates["enDAQ"]['layout']['title_font'] = dict(family="Open Sans", size=24, color=text_color)

    # Background Color
    pio.templates["enDAQ"]['layout']['paper_bgcolor'] = background_color
    pio.templates["enDAQ"]['layout']['plot_bgcolor'] = background_color
    pio.templates["enDAQ"]['layout']['geo']['bgcolor'] = background_color
    pio.templates["enDAQ"]['layout']['polar']['bgcolor'] = background_color
    pio.templates["enDAQ"]['layout']['ternary']['bgcolor'] = background_color

    # Graph Lines
    pio.templates["enDAQ"]['data']['scatter'][0].marker.line.color = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['xaxis']['gridcolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['xaxis']['linecolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['yaxis']['gridcolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['yaxis']['linecolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['zaxis']['gridcolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['scene']['zaxis']['linecolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['xaxis']['gridcolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['xaxis']['linecolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['yaxis']['gridcolor'] = graph_line_color
    pio.templates["enDAQ"]['layout']['yaxis']['linecolor'] = graph_line_color

    # Set Default
    pio.templates.default = "enDAQ"


def set_theme(theme='dark'):
    """
    Sets the plot appearances based on a known 'theme'.

    :param theme: A string denoting which plot appearance color scheme to use.
     Current options are 'dark' and 'light'.
    """
    assert isinstance(theme, str), "'theme' must be given as a string"
    if theme == 'dark':
        set_plot_appearances()
    elif theme == 'light':
        set_plot_appearances(graph_line_color='#DAD9D8', background_color='#FFFFFF', text_color='#404041',
                             default_plotly_template='plotly_white')
    else:
        raise Exception(
            "Theme %s not known.  Try customizing the appearences with the 'set_plot_appearances' function." % theme)
