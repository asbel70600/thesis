from math import sqrt
from pathlib import Path
from numpy._typing import NDArray
from skimage import color, exposure, io, measure
import numpy as np
import scipy.stats
import skimage
from multiprocessing import Pool
import fcntl
import matplotlib.pyplot as plt
import os


def subdivide_hist(img: NDArray) -> NDArray:
    parts = 5
    part = 0
    hist_segments = np.zeros(parts)
    hist = exposure.histogram(img)[0]

    segment_lenght = int(len(hist) / parts)
    sum = 0

    for index, elem in enumerate(hist):
        if (index + 1) % segment_lenght == 0:
            hist_segments[part] = sum
            part += 1
            sum = 0
        else:
            sum += elem

    return hist_segments


def get_boder_data(img: NDArray) -> NDArray:
    w = len(img)
    h = len(img[0])
    d = sqrt(w**2 + h**2)

    foreground = skimage.filters.threshold_li(img)
    contours = measure.find_contours(img, foreground)
    contour_lengths = np.zeros(len(contours))

    for i, contour in enumerate(contours):
        contour_lengths[i] = contour.size / d

    contour_lengths.sort()

    return contour_lengths


def get_misc(img: NDArray) -> NDArray:
    hist = exposure.histogram(img)[0]
    contour_lengths = get_boder_data(img)
    flat_img = img.flatten()
    stats = np.array([])

    stats = np.concatenate((stats, skimage.measure.moments(img).flatten()))
    stats = np.concatenate((stats, skimage.measure.moments_hu(img).flatten()))
    stats = np.concatenate((stats, skimage.measure.inertia_tensor(img).flatten()))
    stats = np.concatenate((stats, skimage.measure.moments_central(img).flatten()))
    stats = np.concatenate((stats, [skimage.measure.blur_effect(img)]))
    stats = np.concatenate((stats, skimage.measure.inertia_tensor_eigvals(img)))
    stats = np.concatenate((stats, skimage.measure.moments_central(img).flatten()))
    stats = np.concatenate((stats, skimage.measure.shannon_entropy(img).flatten()))
    # stats = np.concatenate((stats,skimage.measure.moments_normalized(img).flatten()))

    stats = np.concatenate((stats, np.var(flat_img).flatten()))
    stats = np.concatenate((stats, np.mean(flat_img).flatten()))
    stats = np.concatenate((stats, np.average(flat_img).flatten()))
    stats = np.concatenate((stats, scipy.stats.sem(flat_img).flatten()))
    stats = np.concatenate((stats, [scipy.stats.mode(flat_img).mode]))
    stats = np.concatenate((stats, scipy.stats.skew(flat_img).flatten()))

    stats = np.concatenate((stats, np.var(hist).flatten()))
    stats = np.concatenate((stats, scipy.stats.entropy(hist).flatten()))
    stats = np.concatenate((stats, np.mean(hist).flatten()))
    stats = np.concatenate((stats, np.average(hist).flatten()))
    stats = np.concatenate((stats, scipy.stats.sem(hist).flatten()))
    stats = np.concatenate((stats, [scipy.stats.mode(hist).mode]))
    stats = np.concatenate((stats, scipy.stats.skew(hist).flatten()))

    stats = np.concatenate((stats, np.var(contour_lengths).flatten()))
    stats = np.concatenate((stats, scipy.stats.entropy(contour_lengths).flatten()))
    stats = np.concatenate((stats, np.mean(contour_lengths).flatten()))
    stats = np.concatenate((stats, np.average(contour_lengths).flatten()))
    # stats = np.concatenate((stats,scipy.stats.sem(contour_lengths).flatten()))
    stats = np.concatenate((stats, [scipy.stats.mode(contour_lengths).mode]))
    stats = np.concatenate((stats, scipy.stats.skew(contour_lengths).flatten()))

    return stats


def get_statistics(img: NDArray) -> NDArray:
    moments = np.array([])

    if np.shape(img) != () and len(img[0][0]) < 3:
        print(np.shape(img))

        try:
            plt.imshow(img)
            plt.show()
        except:
            return np.zeros(1130)

        return np.zeros(1130)

    try:
        img_gray = color.rgb2gray(img)
        img_hsv = color.rgb2hsv(img)
        img_xyz = color.rgb2xyz(img)
        img_lab = color.rgb2lab(img)
    except:
        plt.imshow(img)
        plt.show()
        return np.zeros(1130)

    images = np.array(
        [
            img_gray,
            img[:, :, 0],
            img[:, :, 1],
            img[:, :, 2],
            img_hsv[:, :, 0],
            img_hsv[:, :, 1],
            img_hsv[:, :, 2],
            img_xyz[:, :, 0],
            img_xyz[:, :, 1],
            img_xyz[:, :, 2],
            img_lab[:, :, 0],
            img_lab[:, :, 1],
            img_lab[:, :, 2],
        ]
    )

    for j in images:
        moment = get_misc(j)
        hist_section = subdivide_hist(j)
        moments = np.concatenate((moments, moment, hist_section))

    return moments


def main(imgdir: str):
    img_class = Path(imgdir).parent.name
    output_path = f"data/data_{img_class}.csv"
    img = io.imread(imgdir)

    minus_width = int((1 / 6) * len(img))
    minus_height = int((1 / 6) * len(img[0]))

    img = img[minus_width:-minus_width, minus_height:-minus_height, :]

    stats = get_statistics(img)

    fmtlen = len(stats)
    fmt = ["%s"]

    for i in range(fmtlen):
        fmt.append(f"%10.4f")

    stats = np.insert(stats.astype(object), 0, img_class, axis=0)
    stats = stats.reshape(1, -1)
    fmt = tuple(fmt)

    if os.path.exists(output_path) and os.path.isfile(output_path) and not os.path.getsize(output_path) == 0:
        headers = ""
    else:
        headers = "class"
        for i in range(fmtlen):
            headers += f",r{i}"

    with open(output_path, "a") as f:
        fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
        np.savetxt(f, stats, fmt=fmt, header=headers, delimiter=",", comments="")
        fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
    print(f"processed {imgdir}")


if __name__ == "__main__":
    imgs = []

    for i in Path("images").glob("*"):
        for j in Path(f"images/{i.name}").glob("*"):
            imgs.append(j)

    with Pool(12) as p:
        p.map(main, imgs)


# fig,ax = plt.subplots(2)
# ax[0].imshow(img)
# ax[1].imshow(img[minus_width:-minus_width,minus_height:-minus_height,:])
# plt.show()
# print("plotted")

# plt.imshow(img)
# plt.plot()
# plt.imshow(img[minus_width:-minus_width,minus_height:-minus_height,:])
# plt.plot()

# for i in Path("images/cropped/pink").glob("*"):
#     print(i)
#
# for i in Path("images/cropped/yellow").glob("*"):
#     print(i)
# elif len(np.shape(img)) == 3:
#     channels = np.array(
#         [
#             img[:, :, 0].flatten(),
#             img[:, :, 1].flatten(),
#             img[:, :, 2].flatten(),
#         ]
#     )
#     momments = np.zeros((3, 6))
#
#     for i, channel in enumerate(channels):
#         channel_momments = get_moments_of(channel)
#         momments[i] = channel_momments
#
#     return momments
#
# else:
#     print("weird image with shape: {}", np.shape(img))
#     exit(1)

# def get_misc_data(img:NDArray)-> NDArray:
#     potato_polygon = find_potato_polygon(img)
#     centroid = skimage.measure.centroid(potato_polygon)
#
#     blur = skimage.measure.blur_effect(img)
#     hu_moments = skimage.measure.moments_hu(img)
#     scie_moments = skimage.measure.moments(img)
#     central_moments = skimage.measure.moments_central(img)
# def find_potato_polygon(img: NDArray) -> NDArray:
#     w = len(img)
#     h = len(img[0])
#     d = sqrt(w**2 + h**2)
#
#     img_diameter = sqrt(d / 35)
#     foreground = skimage.filters.threshold_li(img)
#     contours = measure.find_contours(img, foreground)
#     longest_contour = np.array([[], []])
#
#     for contour in contours:
#         if contour.size > longest_contour.size:
#             longest_contour = contour
#
#     p_shape = skimage.measure.approximate_polygon(
#         longest_contour, tolerance=img_diameter / 35
#     )
#
#     return p_shape
#
#
# stats = pd.DataFrame(get_statistics(img))
# print(stats.to_csv(index=False,header=False,lineterminator='\n',float_format='%.10f').strip())

#     print(f"saved, len {len(stats)} shape {shape(stats)} checksum {check.sum()}")
#
# for i in Path("images/cropped/yellow").glob("*"):
#     img = io.imread("images/cropped/pink/_MG_8493.CR2.tiff")
#     stats = get_statistics(img)


# try:
#     if random.choice(range(10)) < 2:
#         img = skimage.transform.rescale(
#             img, random.uniform(0.4, 0.9), multichannel=True
#         )
# except Exception as e:
#     print(e)
#
# if random.choice(range(10)) > 7:
#     img = skimage.transform.rotate(img, random.choice(range(360)))
#
# if random.choice(range(10)) < 2:
#     img = skimage.exposure.adjust_gamma(img)
#
# if random.choice(range(10)) < 5:
#     img = skimage.exposure.adjust_sigmoid(img)
#
# if random.choice(range(10)) < 4:
#     img = skimage.util.crop(
#         img,
#         (
#             (random.choice(range(100)), random.choice(range(100))),
#             (random.choice(range(100)), random.choice(range(100))),
#             (0, 0),
#         ),
#     )
#
