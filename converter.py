import time
import os
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as Img
from tqdm import tqdm


# WGS 84, 1979
pol_radius: float = 6356752.314
equator_radius: float = 6378137.0


class Color:
    dirname = Path('planet/terrarium') # path to source files
    hd_dirname = Path('planet/hd_terrarium') # path to hd source files

    foldername: str = 'hypsometric' # foldername to replace "terrarium" in output
    hd_foldername: str = 'hd_terrarium' # foldername to replace hd data

    zl_position: int = 2
    x_position: int = 3
    y_position: int = 4

    meters_per_degree_x = equator_radius * np.pi / 180
    meters_per_degree_y = pol_radius * np.pi / 180

    land_colors_old = np.array([
        [0, 0, 81],
        [0, 174, 162],
        [59, 123, 65],
        [117, 169, 97],
        [231, 226, 162],
        [205, 159, 67],
        [183, 108, 93],
        [165, 120, 165],
        [200, 200, 200],
        [255, 255, 255],
        [171, 231, 255],
    ])
    land_colors = np.array([
        [0, 0, 81],
        [0, 174, 162],
        [59, 123, 65],
        [102, 170, 75],
        [238, 233, 158],
        [205, 159, 67],
        [183, 108, 93],
        [149, 108, 149],
        [204, 204, 204],
        [255, 255, 255],
        [171, 231, 255],
    ])

    land_stops = np.array([-8000, -40, 0, 220, 700, 1300, 2100, 2500, 3000, 3800, 6800])

    def __init__(self, interpolate: bool=True, hd: bool=False, hillshade: bool=False):
        self.interpolate = interpolate
        self.hd = hd
        self.hillshade = hillshade

    def set_meters_per_pixel(self, zl: int, lat: float):
        """Determine how many meters correspond to one pixel."""
        pixel_per_tile = 512
        degree_per_tile = 360 / 2**zl
        degree_per_pixel = degree_per_tile / pixel_per_tile
        self.meters_per_pixel_x = self.meters_per_degree_x * degree_per_pixel * np.cos(lat)

    def get_latitude(self, zl: int, y: int) -> float:
        """Determine average latitude of tile.
        
        Mercator goes from 85.0511 to -85.0511, which is the result of 
        arctan(sinh(π))
        """
        lat_range = 2 * 85.0511
        n_tiles = zl + 1
        slope = - lat_range / n_tiles
        return slope * (y + 0.5) + 85.0511

    def get_elevation(self, array: np.ndarray) -> np.ndarray:
        """Convert terrarium encoded data to actual elevation in meters."""
        return (array[:,:,0] * 256 + array[:,:,1] + array[:,:,2] / 256) - 32768

    def normalize(self, lower_bound: np.ndarray, upper_bound: np.ndarray, val: np.ndarray):
        return (val - lower_bound) / (upper_bound - lower_bound)

    def blend_color_val(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        blended = np.sqrt((1 - t) * a*a + t * b*b)
        return np.rint(blended).astype(np.uint8)

    def get_hypsometric_color(self, elevation: np.ndarray) -> np.ndarray:
        """Apply custom hypsometric color scheme for elevation values.

        Args:
            elevation (np.ndarray): ndarray containing elevation data

        Returns:
            np.ndarray: ndarray containing rgb values for each elevation value.
        """
        hyp = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
        bins = np.digitize(elevation, self.land_stops) - 1
        pos = self.normalize(self.land_stops[bins], self.land_stops[bins + 1], elevation)
        color1 = self.land_colors[bins]
        color2 = self.land_colors[bins + 1]

        # blend rgb values
        for i in range(3):
            if self.interpolate:
                hyp[:,:,i] = self.blend_color_val(color1[:,:,i], color2[:,:,i], pos)
            else:
                hyp[:,:,i] = color1[:,:,i]

        return hyp

    def linear_intensity(self, x, y_0=0.0, m=0.015):
        return y_0 + x*m
        
    def terrarium_to_hypsometric(self, image: Img.Image, zl: int) -> Img.Image:
        # convert image to 3D numpy array (size * size * 3)
        data = np.array(image)
        assert data.shape[0] == data.shape[1]

        elevation = self.get_elevation(data)

        # enforce lower and upper bound
        elevation[elevation <= self.land_stops[0]] = self.land_stops[0] + 1
        elevation[elevation >= self.land_stops[-1]] = self.land_stops[-1] - 1

        data = self.get_hypsometric_color(elevation)
        if self.hillshade:
            hillshade = self.get_hillshade(elevation)
            # intensity = self.linear_intensity(zl)
            intensity = 0.3
            for i in range(3):
                # data[:,:,i] = data[:,:,i] * (1-intensity + hillshade*intensity)
                data[:,:,i] = np.minimum(np.maximum(data[:,:,i]+hillshade*255*intensity, 0), 255)
        img = Img.fromarray(data, 'RGB')
        return img

    def get_hillshade(self, array: np.ndarray, azimuth: float=315, altitude: float=45) -> np.ndarray:
        """Get hillshade per pixel as float number from -1 to 1

        Args:
            array (np.ndarray): Elevation data per pixel
            azimuth (float, optional): Azimuth angle of the sun, degrees. Defaults to 315.
            altitude (float, optional): Altitude of the sun, degrees. Defaults to 45.

        Returns:
            np.ndarray: Hillshading value from -1 (maximum hillshade) to +1 (minimum hillshade)
        """
        x, y = np.gradient(array) / self.meters_per_pixel_x
        slope = np.arctan(np.sqrt(x*x + y*y))

        # aspect range: [-π, π]
        aspect = np.arctan2(y, -x)
        azimuth_math = (360. - azimuth + 90.) % 360.
        azimuth_rad = np.radians(azimuth_math)
        zenith = np.radians(90 - altitude)
        shaded = ((np.cos(zenith) * np.cos(slope)) +
                 (np.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect)))

        # normalize by zenith value to have zero impact for zero float
        return shaded - np.cos(zenith)

    def merge_tiles(self):
        """Iterate all tiles and merge 4 tiles from zl n to one tile in zl n-1."""
        n: int = 0
        start = time.time()
        zoom_dirs = [f for f in self.dirname.iterdir() if f.is_dir()]
        zoom_dirs.sort()
        # we can not merge for highest zoom level
        for zoom_dir in zoom_dirs[:-1]:
            x_dirs_lower = [f for f in zoom_dir.iterdir() if f.is_dir()]
            for x_dir in x_dirs_lower:
                hd_x_dir = self.change_folder_in_path(x_dir, 1, self.hd_foldername)
                hd_x_dir.mkdir(parents=True, exist_ok=True)

                y_files = [f for f in x_dir.iterdir() if f.is_file()]
                for y_file in y_files:
                    print(f'Start with {y_file}')
                    hd_y_file = self.change_folder_in_path(y_file, 1, self.hd_foldername)
                    childs = self.get_childs(y_file)
                    new_image = self.merge_single_tile_numpy(childs)
                    new_image.save(hd_y_file, 'png')
                    n += 1

        print('----------------------------------------------')
        print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')

    def merge_single_tile(self, childs):
        """Merge 4 tiles from zl n to one tile in zl n-1."""
        im1, im2, im3, im4 = [Img.open(c) for c in childs]
        dst = Img.new('RGB', (im1.width * 2, im1.height * 2))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        dst.paste(im3, (im1.width, 0))
        dst.paste(im4, (im1.width, im1.height))
        return dst

    def merge_single_tile_numpy(self, childs):
        """Merge 4 tiles from zl n to one tile in zl n-1."""
        im1, im2, im3, im4 = [np.asarray(Img.open(c)) for c in childs]
        hd_image = np.concatenate(
            (np.concatenate((im1, im2)), np.concatenate((im3, im4))),
            axis=1
        )
        return Img.fromarray(hd_image)

    def change_folder_in_path(self, path: Path, position: int, foldername: str) -> Path:
        """Change folder in path."""
        # split in parts
        parts = list(path.parts)

        # change foldername on position
        parts[position] = foldername
        return Path(*parts)

    def get_childs(self, path: Path, change: int=1) -> List[Path]:
        """Get the four tiles from zl n+1 which correspond to one tile from zl n."""
        childs: List[Path] = []
        parts = list(path.parts)
        zl = int(parts[self.zl_position])
        x = int(parts[self.x_position])
        y, file_ending = os.path.splitext(parts[self.y_position])
        y = int(y)
        parts[self.zl_position] = str(zl+change)

        for i in range(2):
            new_x = x*2 + i
            parts[self.x_position] = str(new_x)
            for j in range(2):
                new_y = y*2 + j
                parts[self.y_position] = str(new_y) + file_ending
                childs.append(Path(*parts))
        return childs

    def run(self):
        """Convert from terrarium to hypsometric and apply hillshade."""
        n: int = 0
        start = time.time()
        dirname = self.dirname if not self.hd else self.hd_dirname
        zoom_dirs = [f for f in dirname.iterdir() if f.is_dir()]
        zoom_dirs.sort()
        
        for i, zoom_dir in enumerate(zoom_dirs):
            x_dirs = [f for f in zoom_dir.iterdir() if f.is_dir()]

            # determine zoom level
            parts = list(zoom_dir.parts)
            zl = int(parts[self.zl_position])
            print(f'Zoomlevel: {zl}')

            for x_dir in tqdm(x_dirs):
                y_files = [f for f in x_dir.iterdir() if f.is_file()]
                hyp_x_dir = self.change_folder_in_path(x_dir, 1, self.foldername)
                hyp_x_dir.mkdir(parents=True, exist_ok=True)

                for y_file in y_files:
                    parts = list(y_file.parts)
                    y, _ = os.path.splitext(parts[self.y_position])
                    avg_latitude = self.get_latitude(zl, int(y))
                    self.set_meters_per_pixel(zl, avg_latitude)

                    hyp_y_file = self.change_folder_in_path(y_file, 1, self.foldername)
                    if hyp_y_file.is_file(): continue
                    # Load Image (JPEG/JPG needs libjpeg to load)
                    # print(f'Start with {y_file}')
                    original_image = Img.open(y_file)

                    # convert to hypsometric
                    new_image = self.terrarium_to_hypsometric(original_image, zl)

                    # save last zoomlevel with better quality
                    if i == len(zoom_dirs)-1:
                        new_image.save(hyp_y_file.with_suffix('.jpeg'), 'jpeg', quality=85, subsampling=0, optimize=True, progressive=False)
                    else:
                        new_image.save(hyp_y_file.with_suffix('.jpeg'), 'jpeg', quality=46, subsampling=0, optimize=True, progressive=False)
                    n += 1
            
        print('----------------------------------------------')
        print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')


if __name__ == '__main__':
    color = Color(hd=True, hillshade=True)

    # uncomment if hd tiles are required
    # color.merge_tiles()

    # uncomment if hypsometric and hillshade are required
    color.run()
