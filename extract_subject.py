import numpy as np
from numpy._typing import NDArray
from skimage import io, measure, color, exposure
from pathlib import Path
from multiprocessing import Pool

images_paths = ["images"]

def find_longest_contour(image: NDArray) -> NDArray:
    image_bw = color.rgb2gray(image)
    image_cropped = image_bw[500:-500,500:-500]
    image_eq = exposure.adjust_sigmoid(image_cropped)
    contours = measure.find_contours(image_eq,0.9)

    longest_contour = np.array([[], []])

    for contour in contours:
        if contour.size > longest_contour.size:
            longest_contour = contour

    return longest_contour


def find_cropped_coordinates(longest_contour: NDArray[np.float32]) -> dict[str, int]:
    startx = int(longest_contour[:, 0].min()+500)
    endx = int(longest_contour[:, 0].max()+500)
    starty = int(longest_contour[:, 1].min()+500)
    endy = int(longest_contour[:, 1].max()+500)

    return {"startx": startx, "endx": endx, "starty": starty, "endy": endy}

def crop_image(image_path: Path):

    if image_path.is_dir():
        return None

    print(f"opening '{str(image_path)}'")
    image_orig = io.imread(str(image_path))
    print("loaded")

    longest_contour = find_longest_contour(image_orig)
    coordinates = find_cropped_coordinates(longest_contour)
    image = image_orig[
        coordinates["startx"] : coordinates["endx"],
        coordinates["starty"] : coordinates["endy"],
    ]
    io.imsave(f"{image_path.parent}/cropped/{image_path.name}.tiff", image, check_contrast=False)
    print(f"saved {image_path.parent}/cropped/{image_path.name}.tiff")


if __name__ == "__main__":
    for path in images_paths:
        pathlist = Path(path).glob("*")

        with Pool(12) as p:
            p.map(crop_image, pathlist)



# test_code
# fig, ax = plt.subplots()
# ax.imshow(image_orig[startx:endx,starty:endy])
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
