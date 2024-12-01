import os
import numpy as np
from numpy._typing import NDArray
from skimage import io, measure, color, exposure
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt

import skimage

input_imgaes = Path("ds")
output_images = Path("images")


def find_longest_contour(image: NDArray) -> NDArray:
    image_cropped = image[500:-500, 500:-500]
    sat_channel = color.rgb2hsv(image_cropped)[:, :, 1]
    longest_contour = np.array([[], []])
    contours = []
    i = 3
    iter = 0

    while len(contours) == 0 and iter < 1000:
        contours = measure.find_contours(sat_channel, i / 10)
        i += 1
        iter += 1

    longest_contour = np.array([[], []])
    for contour in contours:
        if contour.size > longest_contour.size:
            longest_contour = contour

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
    pathlist = Path(input_imgaes).glob("*")
    again = []
    for path in pathlist:
        if os.path.exists(path) and os.path.isdir(path):
            for i in path.glob("*"):
                again.append(i)

    with Pool(12) as p:
        p.map(crop_image, again)


# test_code
# fig, ax = plt.subplots()
# ax.imshow(image_orig[startx:endx,starty:endy])
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
#
#
# for i in range(10):
#     contours = measure.find_contours(image_eq,i/10)
#     if len(contours) == 0:
#         continue
#
#     fig, ax = plt.subplots()
#     ax.imshow(image_cropped)
#     for cont in contours:
#         ax.plot(cont[:,1],cont[:,0])
#
#
#     longest_contour = np.array([[], []])
#     for contour in contours:
#         if contour.size > longest_contour.size:
#             longest_contour = contour
#
#     ax.plot(longest_contour[:,1],longest_contour[:,0],linewidth=7)
#     print(f"i = {i}")
#
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.show()
#
#
# print("#####################################################################################################")
# print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFffffffffffffffffffffffffffffffffuck")
#
