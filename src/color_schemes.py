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
        [222, 218, 200],    # Grey
        [199, 202, 170],    # Darker Green
        [210, 214, 170],    # Light Green
        [222, 221, 176],    # Light Yellow
        [225, 216, 165],    # Medium Champagne
        [220, 199, 149],    # Wheat
        [208, 178, 140],    # Tan
        # [218, 179, 168],    # Silver Pink
        [190, 199, 192],    # Grey
        [245, 245, 245],    # White   
        [212, 235, 235],    # Ice Blue
    ]),
    stops = np.array([ 
        # Stops blend from current to next value
        # Color next top stop indicates which color is reached at this height
        # For example, white transforms to blue between 4100 and 6600

        -8000,  # Dark Blue
        -40,    # Light Blue
        0,      # Grey
        150,    # Darker Green
        400,    # Light Green
        800,    # Light Yellow
        1350,   # Medium Champagne
        1800,   # Wheat
        2300,   # Tan
        3000,   # Grey
        3100,   # White
        4100,   # Ice Blue
        6600    
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
