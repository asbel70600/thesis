import os
import numpy as np
from numpy._typing import NDArray
from skimage import io, measure, color, exposure, filters
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

import skimage

input_imgaes = Path("ds")
output_images = Path("images")


def find_longest_contour(image: NDArray) -> NDArray:
    image = color.rgb2hsv(image)[:, :, 1]
    print(np.shape(image))
    image = img_as_ubyte(image)
    print(np.shape(image))
    io.imsave("sat_chan.png",image)

    threshold = filters.threshold_li(image)


    longest_contour = np.array([[], []])
    contours = []

    contours = measure.find_contours(image, 100)
    print(len(contours))

    longest_contour = np.array([[], []])
    for contour in contours:
        if contour.size > longest_contour.size:
            longest_contour = contour

    for coord in longest_contour:
        x = int(coord[0])
        y = int(coord[1])

        for i in range(8):
            image[x+i, y+i] = 255
            image[x+i, y+1+i] = 255
            image[x+1+i, y+i] = 255
            image[x+1+i, y+1+i] = 255


    io.imsave("test.png", image)

    return longest_contour


def find_cropped_coordinates(longest_contour: NDArray[np.float32]) -> dict[str, int]:
    startx = int(longest_contour[:, 0].min() + 500)
    endx = int(longest_contour[:, 0].max() + 500)
    starty = int(longest_contour[:, 1].min() + 500)
    endy = int(longest_contour[:, 1].max() + 500)

    return {"startx": startx, "endx": endx, "starty": starty, "endy": endy}


def crop_image(image_path: Path):

    if image_path.is_dir():
        return None

    image_orig = io.imread(str(image_path))
    longest_contour = find_longest_contour(image_orig)
    coordinates = find_cropped_coordinates(longest_contour)

    image = image_orig[
        coordinates["startx"] : coordinates["endx"],
        coordinates["starty"] : coordinates["endy"],
    ]

    output_dir = f"{output_images}/{image_path.parent.name}"

    # print(f"outimgs: {output_images}")
    # print(f"parent; {image_path.parent.name}")
    # print(f"output: {output_dir}")

    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception as e:
            print(e)

    io.imsave(f"{output_dir}/{image_path.name}.tiff", image, check_contrast=False)
    print(f"saved {image_path.parent}/cropped/{image_path.name}.tiff")


if __name__ == "__main__":
    image = Path(
        "/home/asbel/projects/uclv/tesis/program/dataset_creation/ds/manitov/manitov_108.tiff"
    )
    crop_image(image)
    # pathlist = Path(input_imgaes).glob("*")
    # again = []
    # for path in pathlist:
    #     if os.path.exists(path) and os.path.isdir(path):
    #         for i in path.glob("*"):
    #             again.append(i)
    #
    # with Pool(12) as p:
    #     p.map(crop_image, again)
