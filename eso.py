from math import sqrt
from urllib.request import parse_http_list
from matplotlib.image import pil_to_array
from scipy.ndimage import minimum_position
from skimage import io, filters, morphology, measure
from skimage.feature import blob_dog, peak_local_max
from skimage.restoration import rolling_ball
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage.util import img_as_ubyte

# Load the image
img = io.imread("ds/alouette/alouette_103.tiff")
image = sk.color.rgb2gray(img)
image = sk.transform.rescale(image, 0.1)
image = sk.util.invert(image)
image = img_as_ubyte(image)
print(image)

background_subtracted = rolling_ball(image)

threshold = filters.threshold_otsu(background_subtracted)
binary_mask = image > threshold
cleaned_mask = morphology.remove_small_objects(binary_mask,min_size=400)
foreground = image * cleaned_mask
foreground[foreground > 0] = 1

binary_mask = binary_mask.astype(int)
plt.imshow(binary_mask)
plt.show()
blobs = sk.feature.blob_log(binary_mask)

for i in blobs:
    (x,y,s) = i
    s = sqrt(2) * s
    plt.imshow(image[x:x+s,y:y+s])
    plt.show()



print(binary_mask)

objects = []
for region in measure.regionprops(binary_mask):
    minr, minc, maxr, maxc = region.bbox
    object_image = image[minr:maxr, minc:maxc]
    objects.append(object_image)

for i, j in enumerate(objects):
    io.imsave(f"{i}.png", j)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Background Subtracted")
# plt.imshow(background_subtracted, cmap="gray")
#
# plt.subplot(1, 2, 2)
# plt.title("Labeled Objects")
# plt.imshow(foreground, cmap="nipy_spectral")
# #
# plt.show()
#
# def eso():
#     print("")
#     print("")
#     print("")
