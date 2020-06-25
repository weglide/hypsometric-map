import time
import os
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as Img
from tqdm import tqdm



class Color:
    dirname = Path('planet/terrarium') # path to source files
    hd_dirname = Path('planet/hd_terrarium') # path to hd source files

    foldername = 'hypsometric' # foldername to replace "terrarium" in output
    hd_foldername = 'hd_terrarium' # foldername to replace hd data

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
    

    def __init__(self, interpolate: bool=True, hd: bool=False, hillshade: bool=False):
        self.interpolate = interpolate
        self.hd = hd
        self.hillshade = hillshade

    def make_stops(self):
        self.land_stops_old = np.array([-8000, -40, 0, 200, 700, 1500, 2500, 3000, 3800, 6800])
        self.land_stops = np.array([-8000, -40, 0, 220, 700, 1300, 2100, 2500, 3000, 3800, 6800])

    def get_elevation(self, array: np.ndarray) -> np.ndarray:
        return (array[:,:,0] * 256 + array[:,:,1] + array[:,:,2] / 256) - 32768

    def normalize(self, lower_bound: np.ndarray, upper_bound: np.ndarray, val: np.ndarray):
        return (val - lower_bound) / (upper_bound - lower_bound)

    def blend_color_val(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        # blended = (1-t)*a + t*b
        blended = np.sqrt((1 - t) * a*a + t * b*b)
        return np.rint(blended).astype(np.uint8)

    def get_hypsometric_color(self, elevation: np.ndarray) -> np.ndarray:
        hyp = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
        # catch invalid elevation values
        # assert self.land_stops[0] < elevation.all() <= self.land_stops[-1]

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
            intensity = self.linear_intensity(zl)
            for i in range(3):
                # data[:,:,i] = data[:,:,i] * (1-intensity + hillshade*intensity)
                data[:,:,i] = np.minimum(np.maximum(data[:,:,i]+hillshade*255*intensity, 0), 255)
        img = Img.fromarray(data, 'RGB')
        return img

    def get_hillshade(self, array: np.ndarray, azimuth: float=315, altitude: float=70) -> np.ndarray:
        # azimuth = 360.0 - azimuth 
        
        # x, y = np.gradient(array)
        # slope = np.arctan(np.sqrt(x * x + y * y))
        # aspect = np.arctan2(-x, y)
        # azimuth_rad = np.radians(azimuth)
        # altitude_rad = np.radians(altitude)
        
        # shaded = np.sin(altitude_rad) * np.sin(slope) \
        # + np.cos(altitude_rad) * np.cos(slope) \
        # * np.cos((azimuth_rad - np.pi / 2.) - aspect)

        x, y = np.gradient(array)
        slope = np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(y, -x)
        azimuth_math = (360. - azimuth + 90.) % 360.
        azimuth_rad = np.radians(azimuth_math)
        zenith = np.radians(90 - altitude)
        shaded = ((np.cos(zenith) * np.cos(slope)) +
                 (np.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect)))

        # return 255 * (shaded + 1) / 2
        # return (shaded + 1) / 2
        return shaded

    def merge_tiles(self):
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
        im1, im2, im3, im4 = [Img.open(c) for c in childs]
        dst = Img.new('RGB', (im1.width * 2, im1.height * 2))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        dst.paste(im3, (im1.width, 0))
        dst.paste(im4, (im1.width, im1.height))
        return dst

    def merge_single_tile_numpy(self, childs):
        im1, im2, im3, im4 = [np.asarray(Img.open(c)) for c in childs]
        hd_image = np.concatenate(
            (np.concatenate((im1, im2)), np.concatenate((im3, im4))),
            axis=1
        )
        return Img.fromarray(hd_image)

    def compress_tile(self, img, size):
        return img.resize((size, size), Img.ANTIALIAS)

    def save(self, img, path):
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp
        # quality is in percent from 0 to 100
        # method is quality/speed tradeoff from 0 (fast) to 6
        # webp works everywhere except for safari. MAybe use jpeg 2000 for safari?
        return img.save(path, 'WEBP', quality=80, method=4) 

    def change_folder_in_path(self, path: Path, position: int, foldername: str) -> Path:
        # split in parts
        parts = list(path.parts)

        # change foldername on position
        parts[position] = foldername
        return Path(*parts)

    def get_childs(self, path: Path, change: int=1) -> List[Path]:
        childs: List[Path] = []
        zl_position: int = 2
        x_position: int = 3
        y_position: int = 4
        parts = list(path.parts)
        zl = int(parts[zl_position])
        x = int(parts[x_position])
        y, file_ending = os.path.splitext(parts[y_position])
        y = int(y)
        parts[zl_position] = str(zl+change)

        for i in range(2):
            new_x = x*2 + i
            parts[x_position] = str(new_x)
            for j in range(2):
                new_y = y*2 + j
                parts[y_position] = str(new_y) + file_ending
                childs.append(Path(*parts))
        return childs

    def run(self):
        
        # determine bins
        self.make_stops()

        n = 0
        start = time.time()
        dirname = self.dirname if not self.hd else self.hd_dirname
        zoom_dirs = [f for f in dirname.iterdir() if f.is_dir()]
        zoom_dirs.sort()
        
        for zoom_dir in zoom_dirs:
            x_dirs = [f for f in zoom_dir.iterdir() if f.is_dir()]

            # determine zoom level
            zl_position: int = 2
            parts = list(zoom_dir.parts)
            zl = int(parts[zl_position])
            print(f'Zoomlevel: {zl}')

            for x_dir in tqdm(x_dirs):
                y_files = [f for f in x_dir.iterdir() if f.is_file()]
                hyp_x_dir = self.change_folder_in_path(x_dir, 1, self.foldername)
                hyp_x_dir.mkdir(parents=True, exist_ok=True)

                for y_file in y_files:
                    hyp_y_file = self.change_folder_in_path(y_file, 1, self.foldername)
                    if hyp_y_file.is_file(): continue
                    # Load Image (JPEG/JPG needs libjpeg to load)
                    # print(f'Start with {y_file}')
                    original_image = Img.open(y_file)

                    # convert to hypsometric
                    new_image = self.terrarium_to_hypsometric(original_image, zl)

                    # create folder if not exists
                    # get path
                    # new_image.save(hyp_y_file, 'png')

                    new_image.save(hyp_y_file.with_suffix('.jpeg'), 'jpeg', quality=46, subsampling=0, optimize=True, progressive=False)

                    n += 1
            
        print('----------------------------------------------')
        print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')


if __name__ == '__main__':
    color = Color(hd=True, hillshade=True)
    # color.merge_tiles()
    color.run()
