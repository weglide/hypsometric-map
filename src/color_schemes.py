from typing import NamedTuple

import numpy as np


class ColorSchema(NamedTuple):
    # assert len(stops) == len(colors) + 1
    colors: np.ndarray
    stops: np.ndarray


moritz_mix = ColorSchema(
    colors = np.array([
        [0, 0, 81],         # Dark Blue
        [0, 174, 162],      # Light Blue
        [211, 210, 185],    # Grey
        [199, 202, 170],    # Darker Green
        [215, 227, 166],    # Light Green
        [238, 237, 176],    # Pale Spring Bud (Yellow)
        [238, 210, 142],    # Gold
        [208, 178, 140],    # Tan
        [218, 179, 168],    # Silver Pink
        [207, 215, 209],    # Grey
        [245, 245, 245],    # White   
        [198, 231, 247],    # Ice Blue
    ]),
    stops = np.array([ 
        # Stop indicates when this color is reached completely. 
        # Color next top stop indicates which color is reached at this height
        # For example, white transforms to blue between 4100 and 6600

        -8000,  # Dark Blue
        -40,    # Light Blue
        0,      # Grey
        50,    # Darker Green
        350,    # Light Green
        500,    # Pale Spring Bud (Yellow)
        1050,   # Gold
        1800,   # Tan
        2300,   # Silver Pink
        3100,   # Grey
        3800,   # White
        6000    # Ice Blue
     ])
)


traditional = ColorSchema(
    colors = np.array([
        [0, 0, 81],         # Dark Blue
        [0, 174, 162],      # Light Blue
        [59, 107, 65],      # Dark Green
        [58, 128, 55],      # Medium Green
        [107, 161, 76],     # Light Green
        [234, 228, 150],    # Light Yellow
        [205, 159, 67],     # Gold
        [183, 108, 93],     # Red
        [150, 110, 150],    # Violet
        [175, 175, 175],    # Gray
        [245, 245, 245],    # White   
        [144, 224, 255],    # Ice Blue
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
        6600,   # Ice Blue
     ])
)

neon = ColorSchema(
    colors = np.array([
        [0, 0, 81],         # Dark Blue
        [0, 174, 162],      # Light Blue
        [107, 185, 100],    # Emerald
        [158, 200, 62],     # Neon Green
        [250, 219, 84],     # Lemon Yellow
        [255, 172, 64],     # Orange
        [225, 142, 97],     # Brown
        [255, 108, 79],     # Red
        [255, 132, 169],    # Pink
        [170, 128, 203],    # Purple
        [187, 180, 203],    # Gray
        [239, 143, 244],    # White
        [123, 218, 255],    # Blue
    ]),
    stops = np.array([
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
