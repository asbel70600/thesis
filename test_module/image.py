from numpy._typing import NDArray
from numpy import array
from skimage import color
from channel import ChannelType, Channel


class ImageRGB:
    rchnl: Channel
    gchnl: Channel
    bchnl: Channel
    image: NDArray
    gray: NDArray

    def __init__(
        self,
        image: NDArray = array([]),
        rchnl: Channel = Channel(),
        gchnl: Channel = Channel(),
        bchnl: Channel = Channel(),
    ) -> None:

        if image.size == 0:
            self.image = image
            self.rchnl = rchnl
            self.gchnl = gchnl
            self.bchnl = bchnl
        else:
            self.image = image
            self.rchnl = Channel(channel_data=image[:,:,0],channel_type=ChannelType.Red)
            self.gchnl = Channel(channel_data=image[:,:,1],channel_type=ChannelType.Green)
            self.bchnl = Channel(channel_data=image[:,:,2],channel_type=ChannelType.Blue)


class ImageHSV:
    hchnl: Channel
    schnl: Channel
    vchnl: Channel
    image: NDArray

    def __init__(
        self,
        image: NDArray = array([]),
        hchnl: Channel = Channel(),
        schnl: Channel = Channel(),
        vchnl: Channel = Channel(),
    ) -> None:

        if image.size == 0:
            self.image = image
            self.hchnl = hchnl
            self.schnl = schnl
            self.vchnl = vchnl
        else:
            hsv_image = color.rgb2hsv(image)
            self.image = hsv_image
            self.hchnl = Channel(channel_data=hsv_image[:,:,0], channel_type=ChannelType.Hue)
            self.schnl = Channel(channel_data=hsv_image[:,:,1], channel_type=ChannelType.Saturation)
            self.vchnl = Channel(channel_data=hsv_image[:,:,2], channel_type=ChannelType.Value)


class ImageLAB:
    lchnl: Channel
    achnl: Channel
    bchnl: Channel
    image: NDArray

    def __init__(
        self,
        image: NDArray = array([]),
        lchnl: Channel = Channel(),
        achnl: Channel = Channel(),
        bchnl: Channel = Channel(),
    ) -> None:

        if image.size == 0:
            self.image = image
            self.lchnl = lchnl
            self.achnl = achnl
            self.bchnl = bchnl
        else:
            lab_image = color.rgb2lab(image)
            self.image = lab_image
            self.lchnl = Channel(channel_data=lab_image[:,:,0], channel_type=ChannelType.Lightness)
            self.achnl = Channel(channel_data=lab_image[:,:,1], channel_type=ChannelType.A)
            self.bchnl = Channel(channel_data=lab_image[:,:,2], channel_type=ChannelType.B)


class ImageXYZ:
    xchnl: Channel
    ychnl: Channel
    zchnl: Channel
    image: NDArray

    def __init__(
        self,
        image: NDArray = array([]),
        xchnl: Channel = Channel(),
        ychnl: Channel = Channel(),
        zchnl: Channel = Channel(),
    ) -> None:

        if image.size == 0:
            self.image = image
            self.xchnl = xchnl
            self.ychnl = ychnl
            self.zchnl = zchnl
        else:
            xyz_image = color.rgb2xyz(image)
            self.image = xyz_image
            self.xchnl = Channel(channel_data=xyz_image[:,:,0], channel_type=ChannelType.X)
            self.ychnl = Channel(channel_data=xyz_image[:,:,1], channel_type=ChannelType.Y)
            self.zchnl = Channel(channel_data=xyz_image[:,:,2], channel_type=ChannelType.Z)


class Image:
    rgh:ImageRGB
    hsv:ImageHSV
    lab:ImageLAB
    xyz:ImageXYZ

    def __init__(self,imageRGB:NDArray) -> None:
        self.rgb = ImageRGB(imageRGB)
        self.hsv = ImageHSV(imageRGB)
        self.lab = ImageLAB(imageRGB)
        self.xyz = ImageXYZ(imageRGB)
