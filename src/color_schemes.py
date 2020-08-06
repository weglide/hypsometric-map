from typing import NamedTuple

import numpy as np


class ColorSchema(NamedTuple):
    # Colors and stops have equal length
    colors: np.ndarray
    stops: np.ndarray


traditional = ColorSchema(
    colors = np.array([
        [0, 0, 81],
        [0, 174, 162],
        [59, 107, 65],
        [58, 128, 55],
        [107, 161, 76],
        [234, 228, 150],
        [205, 159, 67],
        [183, 108, 93],
        [150, 110, 150],
        [175, 175, 175],
        [245, 245, 245],
        [144, 224, 255],
    ]),
    stops = np.array([ 
        # Stops blend from current to next value
        -8000,  # Dark Blue
        -40,    # Light Blue
        0,      # Dark Green
        150,    # Medium Green
        400,    # Light Green
        800,    # Light Yellow
        1350,   # Gold	
        1800,   # Red
        2300,   # Violet
        3000,   # Grey
        3000,   # Gra
        4100,   # White
        6600    # Ice Blue
     ])
)

neon = ColorSchema(
    colors = np.array([
        [0, 0, 81],
        [0, 174, 162],
        [107, 185, 100], # dark green
        [158, 200, 62], # green
        [250, 219, 84], # mustard
        [255, 172, 64], # orange
        [225, 142, 97], # brown
        [255, 108, 79],  # red
        [255, 132, 169], # pink
        [170, 128, 203], # purple
        [187, 180, 203], # gray
        [239, 143, 244], # white
        [123, 218, 255], # blue
    ]),
    stops = np.array([ # Stops blend from current to next value
        -8000,  # Dark Blue
        -40,    # Light Blue
        0,      # Emerald
        180,    # Neon Green 
        380,    # Lemon Yellow
        700,    # Orange
        1000,   # Brown
        1350,   # red
        1800,   # Pink
        2300,   # Purple
        3000,   # Gray
        4100,   # White 
        6600    # Ice Blue
    ])
)
