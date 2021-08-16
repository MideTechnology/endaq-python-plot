import plotly.io as pio
import numpy as np


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



def get_center_of_coordinates(lats, lons, as_list=False, as_degrees=True):
    """
    Inputs and outputs are measured in degrees.
    
    :param lats: An ndarray of latitude points
    :param lons: An ndarray of longitude points
    :param as_list: If True, return a length 2 list of the latitude and longitude coordinates.   If not return a
     dictionary of format {"lon": lon_center, "lat": lat_center}
    :param as_degrees: A boolean value representing if the 'lats' and 'lons' parameters are given in degrees (as opposed
     to radians).  These units will be used for the returned values as well.  
    :return:
    """
    # Convert coordinates to radians if given in degrees
    if as_degrees:
        lats *= np.pi / 180
        lons *= np.pi / 180

    # Convert coordinates to 3D coordinates
    x_coords = np.cos(lats) * np.cos(lons)
    y_coords = np.sin(lats) * np.cos(lons)
    z_coords = np.sin(lons)

    # Caluculate the means of the coordinates in 3D
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    z_mean = np.mean(z_coords)

    # Convert back to lat/lon from 3D coordinates
    lat_center = np.arctan2(y_mean, x_mean)
    lon_center = np.arctan2(z_mean, np.sqrt(x_mean ** 2 + y_mean ** 2))

    # Convert back to degrees from radians
    if as_degrees:
        lat_center *= 180 / np.pi
        lon_center *= 180 / np.pi

    if as_list:
        return [lat_center, lon_center]

    return {
        "lat": lat_center,
        "lon": lon_center,
    }



def determine_plotly_map_zoom(
        lons: tuple = None,
        lats: tuple = None,
        lonlats: tuple = None,
        projection: str = "mercator",
        width_to_height: float = 2.0,
        margin: float = 1.2,
) -> float:
    """
    
    Originally based on the following post:
    https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
    Finds optimal zoom for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434
    
    :param lons: tuple, optional, longitude component of each location
    :param lats: tuple, optional, latitude component of each location
    :param lonlats: tuple, optional, gps locations
    :param projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    :param width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.
    :param margin: The desired margin around the plotted points (where 1 would be no-margin)
    :return: 
    
    NOTES:
     - This could be potentially problematic.  By simply averaging min/max coorindates
      you end up with situations such as the longitude lines -179.99 and 179.99 being
      almost right next to each other, but their center is calculated at 0, the other side of the earth. 
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError("Must pass lons & lats or lonlats")
            
    # longitudinal range by zoom level (20 to 1) in degrees, log scaled, with 360 as min zoom
    lon_zoom_range = np.array([360 / 2 ** (19 - j) for j in range(20)], dtype=np.float32)

    if projection == "mercator":
        maxlon, minlon = max(lons), min(lons)
        maxlat, minlat = max(lats), min(lats)
        
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(f"{projection} projection is not implemented")

    return zoom
    
    