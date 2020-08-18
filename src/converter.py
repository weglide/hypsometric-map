import os
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as Img
from PIL import ImageFile
from tqdm import tqdm

from .color_schemes import moritz_mix
from .web_mercator import WebMercator

# WGS 84, 1979
pol_radius: float = 6356752.314
equator_radius: float = 6378137.0

ImageFile.MAXBLOCK = 2**20
executor = ThreadPoolExecutor(max_workers=4)
futures = []


class Color:
    dirname = Path('data/terrarium') # path to source files
    hd_dirname = Path('data/hd_terrarium') # path to hd source files

    foldername: str = 'hypsometric' # foldername to replace "terrarium" in output
    hd_foldername: str = 'hd_terrarium' # foldername to replace hd data

    zl_position: int = 2
    x_position: int = 3
    y_position: int = 4

    meters_per_degree = equator_radius * np.pi / 180

    # set color schema here
    schema = moritz_mix

    # set intensity for hillshading here
    intensity = 0.32

    def __init__(self, interpolate: bool=True, hd: bool=False, hillshade: bool=False):
        assert len(self.schema.colors) == len(self.schema.stops)
        self.interpolate = interpolate
        self.hd = hd
        self.hillshade = hillshade

    def get_meters_per_pixel(self, zl: int) -> float:
        """Determine how many meters correspond to one pixel.

        Args:
            zl (int): Zoomlevel starting at 0

        Returns:
            float: Meters per pixel for zoomlevel at equator
        """
        pixel_per_tile = 512
        degree_per_tile = 360 / 2**zl
        degree_per_pixel = degree_per_tile / pixel_per_tile
        return self.meters_per_degree * degree_per_pixel

    def get_elevation(self, array: np.ndarray) -> np.ndarray:
        """Convert terrarium encoded elevation data to actual elevation in meters.

        Args:
            array (np.ndarray): Input array in terrarium encoding

        Returns:
            np.ndarray: Elevation in meters
        """
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
        bins = np.digitize(elevation, self.schema.stops) - 1
        pos = self.normalize(self.schema.stops[bins], self.schema.stops[bins + 1], elevation)
        color1 = self.schema.colors[bins]
        color2 = self.schema.colors[bins + 1]

        # blend rgb values
        for i in range(3):
            if self.interpolate:
                hyp[:,:,i] = self.blend_color_val(color1[:,:,i], color2[:,:,i], pos)
            else:
                hyp[:,:,i] = color1[:,:,i]

        return hyp
        
    def terrarium_to_hypsometric(self, image: Img.Image, zl: int, y: int) -> Img.Image:
        """Convert image in terrarium encoding to rgb elevation image.

        Args:
            image (Img.Image): Input image, terrarium encoding
            zl (int): Zoomlevel
            y (int): Y index of tile

        Returns:
            Img.Image: Image in custom rgb encoding
        """
        # convert image to 3D numpy array (size * size * 3)
        data = np.array(image)
        assert data.shape[0] == data.shape[1]

        elevation = self.get_elevation(data)

        # enforce lower and upper bound
        elevation[elevation <= self.schema.stops[0]] = self.schema.stops[0] + 1
        elevation[elevation >= self.schema.stops[-1]] = self.schema.stops[-1] - 1

        data = self.get_hypsometric_color(elevation)
        if self.hillshade:
            hillshade = self.get_hillshade(elevation, zl, y)
            # iterate rgb layers
            for i in range(3):
                data[:,:,i] = np.minimum(np.maximum(data[:,:,i]+hillshade*255*self.intensity, 0), 255)

        img = Img.fromarray(data, 'RGB')
        return img

    def get_hillshade(self, array: np.ndarray, zl: int, y: int, azimuth: float=315, altitude: float=45) -> np.ndarray:
        """Get hillshade per pixel as float number from -1 to 1

        Args:
            array (np.ndarray): Elevation data per pixel
            azimuth (float, optional): Azimuth angle of the sun, degrees. Defaults to 315.
            altitude (float, optional): Altitude of the sun, degrees. Defaults to 45.

        Returns:
            np.ndarray: Hillshading value from -1 (maximum hillshade) to +1 (minimum hillshade)
        """
        # pixel count is regarding all tiles -> second vertical tile starts at 512
        y_scale = np.arange(int(y)*512, int(y)*512+512)
        meters_per_pixel = self.get_meters_per_pixel(zl)
        scale = meters_per_pixel * WebMercator.lat_factor(y_scale, zl)

        x, y = np.gradient(array)

        # divide matrix by vector containing meters per latitude
        x = (x.T / scale).T
        y = (y.T / scale).T

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

    def merge_tiles_multi(self, y_file: Path):
        """Merge 4 tiles to one tile on higher zoomlevel.

        The resolution of the tiles is increased by a factor of 2

        Args:
            y_file (Path): Path to file
        """
        hd_y_file = self.change_folder_in_path(y_file, 1, self.hd_foldername)
        childs = self.get_childs(y_file)
        new_image = self.merge_single_tile(childs)
        new_image.save(hd_y_file, 'png')
        return

    def merge_tiles(self):
        """Iterate all tiles and merge 4 tiles from zl n to one tile in zl n-1."""
        n: int = 0
        start = time.time()
        zoom_dirs = [f for f in self.dirname.iterdir() if f.is_dir()]
        zoom_dirs = sorted(zoom_dirs, key=lambda x: int(os.path.splitext(x)[0].split('/')[-1]))
        # we can not merge for highest zoom level
        for zoom_dir in zoom_dirs[:-1]:
            x_dirs_lower = [f for f in zoom_dir.iterdir() if f.is_dir()]
            for x_dir in tqdm(x_dirs_lower):
                hd_x_dir = self.change_folder_in_path(x_dir, 1, self.hd_foldername)
                hd_x_dir.mkdir(parents=True, exist_ok=True)

                y_files = [f for f in x_dir.iterdir() if f.is_file()]
                for y_file in y_files:
                    a = executor.submit(self.merge_tiles_multi, y_file)
                    futures.append(a)
                    n += 1
                wait(futures)

            print('----------------------------------------------')
            print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')

    def merge_single_tile(self, childs: List[Path]):
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

    def get_childs(self, path: Path) -> List[Path]:
        """Get the four tiles from zl n+1 which correspond to one tile from zl n.

        Args:
            path (Path): Path of source tile

        Returns:
            List[Path]: List of tiles from child tiles on zl n+1
        """
        childs: List[Path] = []
        parts = list(path.parts)
        zl = int(parts[self.zl_position])
        x = int(parts[self.x_position])
        y, file_ending = os.path.splitext(parts[self.y_position])
        y = int(y)
        parts[self.zl_position] = str(zl+1)

        for i in range(2):
            new_x = x*2 + i
            parts[self.x_position] = str(new_x)
            for j in range(2):
                new_y = y*2 + j
                parts[self.y_position] = str(new_y) + file_ending
                childs.append(Path(*parts))
        return childs


    def convert_tile(self, y_file: Path, zoom_dirs: List[Path], zl: int, i: int):
        """Wrapper around terrarium to hypsometric.

        Checks that file has jpeg ending and saves the file after 
        conversion in appropriate resolution.

        Args:
            y_file (Path): Path of file
            zoom_dirs ([type]): List containing directories at top level.
            zl (int): Current zoom level
            i (int): Index referencing zoom_dirs
        """
        parts = list(y_file.parts)
        y, _ = os.path.splitext(parts[self.y_position])
        
        hyp_y_file = self.change_folder_in_path(y_file, 1, self.foldername)
        if hyp_y_file.with_suffix('.jpeg').is_file():
            return

        # convert to hypsometric
        original_image = Img.open(y_file)
        new_image = self.terrarium_to_hypsometric(original_image, zl, int(y))

        # save last zoomlevel with better quality
        if i == len(zoom_dirs) - 1:
            new_image.save(hyp_y_file.with_suffix('.jpeg'), 'jpeg', quality=50, subsampling=0, optimize=True, progressive=False)
            return
        else:
            new_image.save(hyp_y_file.with_suffix('.jpeg'), 'jpeg', quality=50, subsampling=0, optimize=True, progressive=False)
            return


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
                    a = executor.submit(self.convert_tile, y_file, zoom_dirs, zl, i)
                    futures.append(a)
                    n += 1
                wait(futures)
            
        print('----------------------------------------------')
        print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')



if __name__ == '__main__':
    color = Color(hd=True, hillshade=True)
    # uncomment if hd tiles are required
    # color.merge_tiles()
    # uncomment if hypsometric and hillshade are required
    color.run()
