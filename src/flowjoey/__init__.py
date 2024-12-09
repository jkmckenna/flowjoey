"""flowjoey"""

import logging
import warnings

from .load_plate import load_plate, load_plates
from .load_sample_sheet import load_sample_sheet
from .logistic_functions import logistic_curve, logistic_derivative, logistic_second_derivative
from .predefined_gates import predefined_gates
from .well_functions import count_events, calculate_median_gfp
from .load_adata import load_adata
from .rolling_average import rolling_average
from .animated_flow_plots import *
from .modeling_plots import rolling_average_plot

from importlib.metadata import version

package_name = "flowjoey"
__version__ = version(package_name)

__all__ = [
    "calculate_median_gfp",
    "count_events",
    "load_adata",
    "load_plate",
    "load_plates",
    "load_sample_sheet",
    "logistic_curve",
    "logistic_derivative",
    "logistic_second_derivative",
    "predefined_gates",
    "rolling_average",
    "kde_2d_evolution",
    "anim_flow",
    "plot_interactive_3d_kde",
    "plotly_animate_kde",
    "plotly_animate_kde_with_interpolation",
    "rolling_average_plot"
]