from typing import Tuple

import numpy as np


class WebMercator:
    """Implementation of WebMercator projection.

    Tile size is assumed to be 512.
    """

    @staticmethod
    def lat_factor(y: np.ndarray, zl: int) -> np.ndarray:
        """Return cosine of latitude for pixel of zoomlevel

        Args:
            y (np.ndarray): pixel range as ndarray
            zl (int): zoomlevel

        Returns:
            np.ndarray: cosine of lat for each pixel
        """
        if isinstance(y, int):
            assert y <= 512 * (2 ** zl)
        else:
            assert y[-1] <= 512 * (2 ** zl)
        latitude = (
            np.arctan(np.exp(-(y * (2 * np.pi) / (512 * (2 ** zl)) - np.pi)))
            - np.pi / 4
        ) * 2
        return np.cos(latitude)

    @staticmethod
    def pixel_to_coords(x: int, y: int, zl: int) -> Tuple[float, float]:
        """Convert (x, y) to coordinates for particular zoom level.

        Args:
            x (int): x indice
            y (int): y indice
            zl (int): Zoomlevel

        Returns:
            Tuple[float, float]: (lon, lat) in degrees
        """
        lon = x * (2 * np.pi) / (512 * (2 ** zl)) - np.pi
        lat = (
            np.arctan(np.exp(-(y * (2 * np.pi) / (512 * (2 ** zl)) - np.pi)))
            - np.pi / 4
        ) * 2

        return np.degrees(lon), np.degrees(lat)

    @staticmethod
    def coords_to_pixel(lon: float, lat: float, zl: int) -> Tuple[int, int]:
        """Convert (lat, lon) to pixel indices for particular zoom level.

        Args:
            lat (float): Latitude, degrees
            lon (float): Longitude, degrees
            zl (int): Zoomlevel

        Returns:
            Tuple[int, int]: (x, y) pixel count
        """
        assert -180 <= lon <= 180, "Longitude must be be in range [-180, 180]"
        assert (
            np.abs(lat) <= 85.0511
        ), "Latitude must be be in range [-85.0511, 85.0511]"

        lon = np.radians(lon)
        lat = np.radians(lat)
        x = 512 * (2 ** zl) * (lon + np.pi) / (2 * np.pi)
        y = (
            512
            * (2 ** zl)
            * (np.pi - np.log(np.tan(np.pi / 4 + lat / 2)))
            / (2 * np.pi)
        )
        return x, y

    @staticmethod
    def tile_to_coords(x: int, y: int, z: int):
        """Get coordinates of specific tile in Web Mercator projection.

        Args:
            x (int): x indice of tile
            y (int): y indice of tile
            z (int): zoom level of tile

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]: (upper_left, lower_right)
        """
        upper_left = WebMercator.pixel_to_coords(x * 512, y * 512, z)
        lower_right = WebMercator.pixel_to_coords((x + 1) * 512, (y + 1) * 512, z)
        return upper_left, lower_right
