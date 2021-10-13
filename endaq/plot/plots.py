import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal

from endaq.calc.psd import to_octave, welch

from .utilities import determine_plotly_map_zoom, get_center_of_coordinates



DEFAULT_ATTRIBUTES_TO_PLOT_INDIVIDUALLY = np.array([
    'accelerationPeakFull', 'accelerationRMSFull', 'velocityRMSFull', 'psuedoVelocityPeakFull',
    'displacementRMSFull', 'gpsSpeedFull', 'gyroscopeRMSFull', 'microphonoeRMSFull',
    'temperatureMeanFull', 'pressureMeanFull'])


def multi_file_plot_row(multi_file_db, rows_to_plot=DEFAULT_ATTRIBUTES_TO_PLOT_INDIVIDUALLY, recording_colors=None,
                        width_per_subplot=400):
    """
    Creates a Plotly figure plotting all the desired attributes from the given DataFrame.

    :param multi_file_db: The Pandas DataFrame of data to plot attributes from
    :param rows_to_plot: A numpy ndarray of strings with the names of the attributes to plot
    :param recording_colors: The colors to make each of the points (All will be the same color if None is given)
    :param width_per_subplot: The width to make every subplot
    :return: A Plotly figure of all the subplots desired to be plotted
    """

    assert isinstance(rows_to_plot, np.ndarray), \
        "Instead of an ndarray given for 'rows_to_plot', a variable of type %s was given" % str(type(rows_to_plot))
    assert len(rows_to_plot) > 0, "At least one value must be given for 'rows_to_plot'"
    assert all(isinstance(row, str) for row in rows_to_plot), "All rows to plot must be given as a string!"

    should_plot_row = np.array([not multi_file_db[r].isnull().all() for r in rows_to_plot])

    if recording_colors is None:
        recording_colors = np.full(len(multi_file_db), 0)

    rows_to_plot = rows_to_plot[should_plot_row]

    fig = make_subplots(
        rows=1,
        cols=len(rows_to_plot),
        subplot_titles=rows_to_plot,
    )

    for j, row_name in enumerate(rows_to_plot):
        fig.add_trace(
            go.Scatter(
                x=multi_file_db['recording_ts'],
                y=multi_file_db[row_name],
                name=row_name,
                mode='markers',
                text=multi_file_db['serial_number_id'].values,
                marker_color=recording_colors,
                # marker_color=multi_file_db['serial_number_id'].map(SERIALS_TO_INDEX.get).values,
            ),
        row=1,
        col=j + 1,
        )
    return fig.update_layout(width=len(rows_to_plot)*width_per_subplot, showlegend=False)


def general_get_correlation_figure(merged_df, recording_colors=None, hover_names=None,
                                   characteristics_to_show_on_hover=[], starting_cols=None):
    if recording_colors is None:
        recording_colors = np.full(len(merged_df), 0)

    cols = [col for col, t in zip(merged_df.columns, merged_df.dtypes) if t != np.object]

    point_naming_characteristic = merged_df.index if hover_names is None else hover_names

    # This is not necessary, but usually produces easily discernible correlations or groupings of files/devices.
    # The hope is that when the initial plot has these characteristics, it will encourage
    # the exploration of this interactive plot.
    start_dropdown_indices = [0, 1]
    first_x_var = cols[0]
    first_y_var = cols[1]
    if starting_cols is not None and starting_cols[0] in cols and starting_cols[1] in cols:
        for j, col_name in enumerate(cols):
            if col_name == starting_cols[0]:
                first_x_var, start_dropdown_indices[0] = col_name, j
            if col_name == starting_cols[1]:
                first_y_var, start_dropdown_indices[1] = col_name, j

    # Create the scatter plot of the initially selected variables
    fig = px.scatter(
        merged_df,
        x=first_x_var,
        y=first_y_var,
        color=recording_colors,
        hover_name=point_naming_characteristic,
        hover_data=characteristics_to_show_on_hover,
        # width=800,
        # height=500,
    )

    # Create the drop-down menus which will be used to choose the desired file characteristics for comparison
    drop_downs = []
    for axis in ['x', 'y']:
        drop_downs.append([
            dict(
                method="update",
                args=[{axis: [merged_df[cols[k]]]},
                      {'%saxis.title.text' % axis: cols[k]},
                      # {'color': recording_colors},
                      ],
                label=cols[k]) for k in range(len(cols))
        ])

    # Sets up various apsects of the Plotly figure that is currently being produced.  This ranges from
    # aethetic things, to setting the dropdown menues as part of the figure
    fig.update_layout(
        title_x=0.4,
        # width=800,
        # height=500,
        showlegend=False,
        updatemenus=[{
            'active': start_j,
            'buttons': drop_down,
            'x': 1.125,
            'y': y_height,
            'xanchor': 'left',
            'yanchor': 'top',
        } for drop_down, start_j, y_height in zip(drop_downs, start_dropdown_indices, [1, .85])])

    # Adds text labels for the two drop-down menus
    for axis, height in zip(['X', 'Y'], [1.05, .9]):
        fig.add_annotation(
            x=1.1,  # 1.185,
            y=height,
            xref='paper',
            yref='paper',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            text="%s-Axis Measurement" % axis,
        )
    return fig


def get_pure_numpy_2d_pca(df, recording_colors=None):
    """
    Get a Plotly figure of the 2d PCA for the given DataFrame.   This will have dropdown menus to select
    which components are being used for the X and Y axis.

    :param df: The dataframe of points to compute the PCA with
    :param recording_colors: See the same parameter in the general_get_correlation_figure function
    :return: A plotly figure as described in the main function description

    TODO:
     - Add assert statements to ensure the given dataframe contains enough values of the desired type
     - Add assert statement to ensure the recording_colors given (if not None) are the proper length
    """

    # Drop all non-float64 type columns, and drop all columns with standard deviation of 0 because this will result
    # in division by 0
    X = df.loc[:, (df.std() != 0) & (np.float64 == df.dtypes)].dropna(axis='columns')

    # Standardize the values (so that mean of each variable is 0 and standard deviation is 1)
    X = (X - X.mean()) / X.std()

    # Get the shape of the data to compute PCA for
    n, m = X.shape

    # Compute covariance matrix
    covariance = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(covariance)

    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)

    # Create a new DataFrame with column names describing the fact that the values are the principal components
    pca_df = pd.DataFrame(X_pca)
    pca_df.columns = (pca_df.columns + 1).map(lambda x: "Component %d" % x)

    # Produce the Plotly figure for the points after being transfored with the PCA computed,
    # with dropdown menus to allow selection of what PCA components are being analyzed
    fig = general_get_correlation_figure(
        pca_df,
        recording_colors,
        characteristics_to_show_on_hover=[df.index],
        hover_names=df.index
    )

    return fig


def gen_map(df_map, mapbox_access_token, filter_points_by_positive_groud_speed=True, color_by_column="GNSS Speed: Ground Speed"):
    """
    Plots GPS data on a map from a single recording, shading the points based some characteristic
    (defaults to ground speed).
    
    :param df_map: The pandas dataframe containing the recording data.
    :param mapbox_access_token: The access token (or API key) needed to be able to plot against
     a map.
    :param filter_points_by_positive_groud_speed: A boolean variable, which will filter
     which points are plotted by if they have corresponding positive ground speeds.  This helps
     remove points which didn't actually have a GPS location found (was created by a bug in the hardware I believe).
    :param color_by_column: The dataframe column title to color the plotted points by.
    """
    if filter_points_by_positive_groud_speed:
        df_map = df_map[df_map["GNSS Speed: Ground Speed"] > 0]
    
    zoom = determine_plotly_map_zoom(lats=df_map["Location: Latitude"], lons=df_map["Location: Longitude"])
    center = get_center_of_coordinates(lats=df_map["Location: Latitude"], lons=df_map["Location: Longitude"])
    
    px.set_mapbox_access_token(mapbox_access_token)
    
    fig = px.scatter_mapbox(
        df_map,
        lat="Location: Latitude",
        lon="Location: Longitude",
        color=color_by_column,
        size_max=15,
        zoom=zoom - 1,
        center=center,
    )

    return fig
    

def octave_spectrogram(df, win, bins_per_octave=3, freq_start=20, max_freq=float('inf'), db_scale=True, log_scale_y_axis=True):
    """
    Produces an octave spectrogram of the given data.

    :param df: The dataframe of sensor data
    :param win: The time window for each of the columns in the spectrogram
    :param bins_per_octave: The number of frequency bins per octave
    :param freq_start: The center of the first frequency bin
    :param max_freq: The maximum frequency to plot
    :param db_scale: If the spectrogram should be log scaled for visibility (with 10*log10(x))
    :param log_scale_y_axis: If the y-axis of the plot should be log scaled
    :return: a tuple containing:
     - the frequency bins
     - the time bins
     - the spectrogram data
     - the corresponding plotly figure
    """
    ary = df.values.squeeze()

    fs = (len(df) - 1) / (df.index[-1] - df.index[0])
    N = int(fs * win) #Number of points in the fft
    w = signal.blackman(N, False)
    
    freqs, bins, Pxx = signal.spectrogram(ary, fs, window=w, nperseg=N, noverlap=0)

    time_dfs = [pd.DataFrame({bins[j]: Pxx[:, j]}, index=freqs) for j in range(len(bins))]

    octave_dfs = list(map(lambda x: to_octave(x, freq_start, octave_bins=bins_per_octave, agg='sum'), time_dfs))
    
    combined_df = pd.concat(octave_dfs, axis=1)
    
    freqs = combined_df.index.values
    Pxx = combined_df.values
    
    include_freqs_mask = freqs <= max_freq
    Pxx = Pxx[include_freqs_mask]
    freqs = freqs[include_freqs_mask]

    if db_scale:
        Pxx = 10 * np.log10(Pxx)

    trace = [go.Heatmap(x=bins, y=freqs, z=Pxx, colorscale='Jet')]
    layout = go.Layout(
        yaxis={'title': 'Frequency (Hz)'},
        xaxis={'title': 'Time (s)'},
    )

    fig = go.Figure(data=trace, layout=layout)
    
    if log_scale_y_axis:
        fig.update_yaxes(type="log")

    fig.update_traces(showscale=False)

    return freqs, bins, Pxx, fig
    
	
def octave_psd_bar_plot(df, bins_per_octave=3, f_start=20, yaxis_title='', log_scale_y_axis=True):
    """
    Produces a bar plot of an octave psd.

    :param df: The dataframe of sensor data
    :param bins_per_octave: The number of frequency bins per octave
    :param f_start: The center of the first frequency bin
    :param yaxis_title: The text to label the y-axis
    :param log_scale_y_axis: If the y-axis should be log scaled
    """
    psd_df = welch(df, 1, scaling='spectrum')

    octave_psd_df = to_octave(
        psd_df,
        f_start, 
        bins_per_octave,
        agg='sum',
    )

    frequency_durations = np.diff(octave_psd_df.index)

    bar = go.Bar(
        x=octave_psd_df.index.values,
        y=octave_psd_df.values.squeeze(),
        width=frequency_durations,
    )
    layout = go.Layout(
        yaxis={'title': yaxis_title},
        xaxis={'title': 'Frequency (Hz)'},
    )

    fig = go.Figure(data=bar, layout=layout)

    if log_scale_y_axis:
        fig.update_xaxes(type="log")

    return fig


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def audio_dB_plot(df, density_of_air=1.204, speed_of_sound_medium=343, reference_intensity=1e-12,
                  rolling_window_len=101, max_points_to_plot=1000):
    """
    Produces a Plotly Figure of decibels vs time from audio recorded by an enDAQ device.

    :param df: The Dataframe of audio data
    :param density_of_air: The density of the air in the location of the recording in kg/m^3.  Defaults
     to the density of air at STP
    :param speed_of_sound_medium: The speed of sound in the air at the location of the recording, in m/s.
    :param reference_intensity: The relative sound intensity level.  More information on this and how to estimate it
     can be found here: https://courses.lumenlearning.com/physics/chapter/17-3-sound-intensity-and-sound-level/
    :param rolling_window_len: The window size of the rolling maximum pressure difference window
    :param max_points_to_plot: The maximum number of points to plot.  If the time series exceeds this number,
     the data will be resampled to this number of points
    """
    rolling_ary = rolling_window(df.values.squeeze(), rolling_window_len)

    delta_p = (np.max(rolling_ary, axis=1) - np.min(rolling_ary, axis=1)) / 2

    intensity = (delta_p) ** 2 / (2 * density_of_air * speed_of_sound_medium)

    db = 10 * np.log10(intensity / reference_intensity)

    times = df.index.values

    # This is done since the rolling windowed operations reduces the number of samples
    time_stamps_to_skip = (rolling_window_len - 1) // 2
    times = times[time_stamps_to_skip: - time_stamps_to_skip]

    if len(db) > max_points_to_plot:
        resampled_db, times = signal.resample(db, max_points_to_plot, t=times)
    else:
        resampled_db = db

    fig = px.line(
        x=times,
        y=resampled_db,
    )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Decibels (dB)",
    )

    return fig