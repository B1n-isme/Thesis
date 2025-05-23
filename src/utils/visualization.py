import os
from typing import List, Optional
import pandas as pd
from utilsforecast.plotting import plot_series


class ForecastVisualizer:
    """
    A utility class for plotting and saving forecast visualizations using utilsforecast.plotting.plot_series.

    Attributes
    ----------
    Y_df : pd.DataFrame
        DataFrame of actual time series with columns ['unique_id', 'ds', 'y']
    Y_hat_df : pd.DataFrame
        DataFrame of forecasted values with columns ['unique_id', 'ds', 'y']
    """
    def __init__(
        self,
        Y_df: pd.DataFrame,
        Y_hat_df: pd.DataFrame,
    ):
        self.Y_df = Y_df.copy()
        self.Y_hat_df = Y_hat_df.copy()

    def plot(
        self,
        ids: Optional[List[int]] = None,
        levels: Optional[List[int]] = None,
        max_insample_length: Optional[int] = None,
        plot_anomalies: bool = False,
        engine: str = 'matplotlib',
        plot_random: bool = False,
        **kwargs
    ):
        """
        Generate a forecast vs. actual plot.

        Parameters
        ----------
        ids : List[int], optional
            List of series IDs to plot (default None = all)
        levels : List[int], optional
            Prediction intervals to display, e.g., [80, 95]
        max_insample_length : int, optional
            Number of past periods to display
        plot_anomalies : bool
            Whether to highlight anomalies outside intervals
        engine : str
            Backend engine ('matplotlib' or 'plotly')
        plot_random : bool
            If True, select random series to plot
        **kwargs
            Additional kwargs passed to plot_series

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
            The generated figure object
        """
        fig = plot_series(
            self.Y_df,
            forecasts_df=self.Y_hat_df,
            ids=ids,
            level=levels,
            max_insample_length=max_insample_length,
            plot_anomalies=plot_anomalies,
            engine=engine,
            plot_random=plot_random,
            **kwargs
        )
        return fig

    def save(
        self,
        fig,
        save_dir: str = '.',
        filename: str = 'forecast_plot.png',
        dpi: int = 300
    ):
        """
        Save the figure to disk.

        Parameters
        ----------
        fig : Figure
            The figure object returned by plot()
        save_dir : str
            Directory to save the figure
        filename : str
            File name for the saved image
        dpi : int
            Resolution in dots per inch

        Returns
        -------
        str
            Full path to the saved file
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return path


# Example
# --------
# from forecast_visualizer import ForecastVisualizer
# vis = ForecastVisualizer(Y_df, Y_hat_df)
# fig = vis.plot(ids=[0,1], levels=[80,95], max_insample_length=36, plot_anomalies=True)
# out_path = vis.save(fig, save_dir='outputs', filename='forecast.png')
# print(f"Plot saved to {out_path}")
