import numpy as np
from numpy._typing import NDArray
from skimage import io, measure, color, exposure
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt

images_paths = ["images"]

def find_longest_contour(image: NDArray) -> NDArray:
    # image = exposure.adjust_sigmoid(image)
    image_bw = color.rgb2gray(image)
    image_cropped = image_bw[500:-500,500:-500]
    image_eq = exposure.adjust_sigmoid(image_cropped)

    contours_variants = []
    longest_contour = np.array([[], []])
    count = 0

    image = image_eq

    # contours = measure.find_contours(image,0.1)
    # contours_variants.append(contours)
    # contours = measure.find_contours(image,0.2)
    # contours_variants.append(contours)
    # contours = measure.find_contours(image,0.3)
    # contours_variants.append(contours)
    # contours = measure.find_contours(image,0.4)
    # contours_variants.append(contours)
    # contours = measure.find_contours(image,0.5)
    # contours_variants.append(contours)
    # contours = measure.find_contours(image,0.6)
    # contours_variants.append(contours)
    contours = measure.find_contours(image,0.7)
    contours_variants.append(contours)
    contours = measure.find_contours(image,0.8)
    contours_variants.append(contours)
    contours = measure.find_contours(image,0.9)
    contours_variants.append(contours)
    contours = measure.find_contours(image,1.0)
    contours_variants.append(contours)
    contours = measure.find_contours(image,1.1)
    contours_variants.append(contours)



    fig,ax = plt.subplots(1,5)

    for i in contours_variants:
        ax[count].imshow(image)
        for contour in i:
            if contour.size > longest_contour.size:
                longest_contour = contour

        ax[count].plot(longest_contour[:,1],longest_contour[:,0])
        count += 1

    plt.show()

    return longest_contour


def find_cropped_coordinates(longest_contour: NDArray[np.float32]) -> dict[str, int]:
    startx = int(longest_contour[:, 0].min())
    endx = int(longest_contour[:, 0].max())
    starty = int(longest_contour[:, 1].min())
    endy = int(longest_contour[:, 1].max())

    return {"startx": startx, "endx": endx, "starty": starty, "endy": endy}

def crop_image(image_path: Path):

    if image_path.is_dir():
        return None

    print(f"opening '{str(image_path)}'")
    image_orig = io.imread(str(image_path))
    print("loaded")

    longest_contour = find_longest_contour(image_orig)
    # coordinates = find_cropped_coordinates(longest_contour)
    # image = image_orig[
    #     coordinates["startx"] : coordinates["endx"],
    #     coordinates["starty"] : coordinates["endy"],
    # ]
    # io.imsave(f"{image_path.parent}/cropped/{image_path.name}.tiff", image, check_contrast=False)
    # print(f"saved {image_path.parent}/cropped/{image_path.name}.tiff")


if __name__ == "__main__":
    path = Path("images/_MG_8652.CR2")
    crop_image(path)
    # for path in images_paths:
    #     pathlist = Path(path).glob("*")

        # with Pool(12) as p:
        #     p.map(crop_image, pathlist)



# test_code
# fig, ax = plt.subplots()
# ax.imshow(image_orig[startx:endx,starty:endy])
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
