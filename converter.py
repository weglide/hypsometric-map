# pip install Pillo
import math
import time
from PIL import Image
from pathlib import Path

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
    # Get R, G, B values (This are int from 0 to 255)
    red =   pixel[0]
    green = pixel[1]
    blue =  pixel[2]

    elevation = (red * 256 + green + blue / 256) - 32768

    return elevation


def normalize(min, max, val):
    return (val - min) / (max - min)


def blend_color_val(a, b, t):
    blended = math.sqrt((1 - t) * a*a + t * b*b)
    return round(blended)


def get_hypsometric_color(ele):
    stops = [-11500, -100, 300, 1000, 3000, 9000] # color stops in meter altitude
    colors = [
        [42, 29, 49],
        [70, 133, 155],
        [2, 98, 71],
        [212, 195, 109],
        [125, 38, 36],
        [255, 255, 255]
    ]
    
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
    width, height = image.size

    # Create new Image and a Pixel Map
    new = create_image(width, height)
    pixels = new.load()

    # Transform to hypsometric
    for i in range(width):
        for j in range(height):
            # get pixel
            pixel = get_pixel(image, i, j)

            # get elevation from pixel rgb values
            elevation = get_elevation(pixel)

            # get hypsometric color form elevation
            hypsometric = get_hypsometric_color(elevation)

            # set pixel in new image
            pixels[i, j] = (hypsometric[0], hypsometric[1], hypsometric[2])

    # return new image
    return new

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
                original = open_image(y_file)

                # convert to hypsometric
                new = terrarium_to_hypsometric(original)
                # create folder if not exists
                
                hyp_x_dir.mkdir(parents=True, exist_ok=True)
                # get path
                hyp_y_file = change_folder_in_path(y_file, 1, foldername)
                save_image(new, hyp_y_file)
                n = n + 1
    
    toc = time.perf_counter()
    
    print(f"Converted {n} tiles in {toc - tic:0.4f} seconds")
