# pip install Pillo
import math
import time
from PIL import Image
from pathlib import Path
import numpy as np

stops = [-11500, -100, 300, 1000, 3000, 9000] # color stops in meter altitude
colors = [
    [42, 29, 49],
    [70, 133, 155],
    [2, 98, 71],
    [212, 195, 109],
    [125, 38, 36],
    [255, 255, 255]
]


# Open an Image
def open_image(path):
    newImage = Image.open(path)
    return newImage

# Save Image
def save_image(image, path):
    image.save(path, 'png')


# Create a new image with the given size
def create_image(i, j):
    image = Image.new("RGB", (i, j), "white")
    return image


# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
        return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel

def get_elevation(pixel):
    return (pixel[0] * 256 + pixel[1] + pixel[2] / 256) - 32768


def normalize(min, max, val):
    return (val - min) / (max - min)


def blend_color_val(a, b, t):
    # ~20% of Calculation workload
    blended = math.sqrt((1 - t) * a*a + t * b*b)
    return round(blended)


def get_hypsometric_color(ele):
    hyp = [0, 0, 0] # result: blended colors
    color1 = [0, 0, 0] # color 1 to blend
    color2 = [0, 0, 0] # color 2 to blend
    pos = 0  # blend position [0..1]
    
    # catch invalid ele values
    if ele <= stops[0] or ele > stops[-1]:
        raise Exception('Elevation value {} does not exist on earth.'.format(ele))
    
    # find colors to blend with stops
    for i in range(len(stops) - 1):
        if ele > stops[i] and ele <= stops[i + 1]:
            pos = normalize(stops[i], stops[i + 1], ele)
            color1 = colors[i]
            color2 = colors[i + 1]

    # blend rgb values
    for i in range(3):
        hyp[i] = blend_color_val(color1[i], color2[i], pos)
        
    return hyp
        
        
def terrarium_to_hypsometric(image):
    # convert image to 3D numpy array (size * size * 3)
    data = np.array(image)
    size = len(data) # assume image is square

    for i in range(size):
        for j in range(size):
            # ~80% of Calculation workload
            # TODO: Vectorize? Does not work out of the box maybe due to dimension reduction and expansion
            data[i][j] = get_hypsometric_color(get_elevation(data[i][j]))

    # return new image
    return Image.fromarray(data, 'RGB')

def change_folder_in_path(path, position, foldername):
    # split in parts
    path = path.parts
    # convert to list
    path = list(path)
    # change foldername on position
    path[position] = foldername
    # convert back to tuple
    path = tuple(path)
    # create path object
    path = Path(*path)

    return path

# Main
if __name__ == "__main__":
    n = 0
    foldername = "hypsometric"
    
    tic = time.perf_counter()

    # Load Image (JPEG/JPG needs libjpeg to load)

    dirname = Path("data/terrarium")
    zoom_dirs = [f for f in dirname.iterdir() if f.is_dir()]
    
    for zoom_dir in zoom_dirs:
        x_dirs = [f for f in zoom_dir.iterdir() if f.is_dir()]

        for x_dir in x_dirs:
            y_files = [f for f in x_dir.iterdir() if f.is_file()]
            hyp_x_dir = change_folder_in_path(x_dir, 1, foldername)

            for y_file in y_files:
                tic1 = time.perf_counter()
                original = open_image(y_file)
                toc1 = time.perf_counter()
                print(f"Open: {toc1 - tic1:0.4f} seconds")

                # convert to hypsometric
                tic2 = time.perf_counter()
                new = terrarium_to_hypsometric(original)
                toc2 = time.perf_counter()
                print(f"Calculation: {toc2 - tic2:0.4f} seconds")

                # create folder if not exists
                tic3 = time.perf_counter()
                hyp_x_dir.mkdir(parents=True, exist_ok=True)
                # get path
                hyp_y_file = change_folder_in_path(y_file, 1, foldername)
                save_image(new, hyp_y_file)
                toc3 = time.perf_counter()
                print(f"Write: {toc3 - tic3:0.4f} seconds")
                print("---------------------------------")
                n = n + 1
    
    toc = time.perf_counter()
    
    print("----------------------------------------------")
    print(f"Converted {n} tiles in {toc - tic:0.4f} seconds")
