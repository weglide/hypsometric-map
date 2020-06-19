import time
from pathlib import Path

import numpy as np
import PIL.Image as Img

dirname = Path('planet/terrarium') # path to source files
foldername = 'hypsometric' # foldername to replace "terrarium" in output

stops = np.array([-11000, -50, 30, 800, 2000, 4800, 7200]) # color stops in meter altitude
colors = np.array([
    [42, 29, 49],
    [70, 133, 155],
    [2, 98, 71],
    [212, 195, 109],
    [125, 38, 36],
    [255, 255, 255],
    [199, 239, 255]
])


def get_elevation(array: np.ndarray) -> np.ndarray:
    return (array[:,:,0] * 256 + array[:,:,1] + array[:,:,2] / 256) - 32768


def normalize(lower_bound: np.ndarray, upper_bound: np.ndarray, val: np.ndarray):
    return (val - lower_bound) / (upper_bound - lower_bound)


def blend_color_val(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    blended = np.sqrt((1 - t) * a*a + t * b*b)
    return np.rint(blended).astype(np.uint8)


def get_hypsometric_color(elevation: np.ndarray) -> np.ndarray:
    hyp = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
    # catch invalid elevation values
    assert stops[0] < elevation.all() <= stops[-1]

    bins = np.digitize(elevation, stops) - 1
    pos = normalize(stops[bins], stops[bins + 1], elevation)
    color1 = colors[bins]
    color2 = colors[bins + 1]

    # blend rgb values
    for i in range(3):
        hyp[:,:,i] = blend_color_val(color1[:,:,i], color2[:,:,i], pos)
        
    return hyp
        
        
def terrarium_to_hypsometric(image: Img.Image) -> Img.Image:
    # convert image to 3D numpy array (size * size * 3)
    data = np.array(image)
    assert data.shape[0] == data.shape[1]

    data = get_elevation(data)

    # enforce lower and upper bound
    data[data <= stops[0]] = stops[0] + 1
    data[data >= stops[-1]] = stops[-1] - 1

    data = get_hypsometric_color(data)
    img = Img.fromarray(data, 'RGB')
    return img


def hillshade(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth 
    
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.
     
 
    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos((azimuthrad - np.pi / 2.) - aspect)
    return 255 * (shaded + 1) / 2


def change_folder_in_path(path: Path, position: int, foldername: str) -> Path:
    # split in parts
    parts = list(path.parts)

    # change foldername on position
    parts[position] = foldername

    return Path(*parts)


if __name__ == '__main__':
    n = 0
    start = time.time()
    zoom_dirs = [f for f in dirname.iterdir() if f.is_dir()]
    
    for zoom_dir in zoom_dirs:
        x_dirs = [f for f in zoom_dir.iterdir() if f.is_dir()]

        for x_dir in x_dirs:
            y_files = [f for f in x_dir.iterdir() if f.is_file()]
            hyp_x_dir = change_folder_in_path(x_dir, 1, foldername)

            for y_file in y_files:
                hyp_y_file = change_folder_in_path(y_file, 1, foldername)
                if hyp_y_file.is_file(): continue
                start1 = time.time()
                # Load Image (JPEG/JPG needs libjpeg to load)
                print(f'Start with {y_file}')
                original_image = Img.open(y_file)
                # print(f'Open: {time.time() - start1:0.4f} seconds')

                # convert to hypsometric
                start2 = time.time()
                new_image = terrarium_to_hypsometric(original_image)
                # print(f'Calculation: {time.time() - start2:0.4f} seconds')

                # create folder if not exists
                start3 = time.time()
                hyp_x_dir.mkdir(parents=True, exist_ok=True)
                # get path
                new_image.save(hyp_y_file, 'png')
                # print(f'Write: {time.time() - start3:0.4f} seconds')
                # print('---------------------------------')
                n = n + 1
        
    print('----------------------------------------------')
    print(f'Converted {n} tiles in {time.time() - start:0.4f} seconds')
