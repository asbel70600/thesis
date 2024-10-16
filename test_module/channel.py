from numpy._typing import NDArray
from numpy import array, floating
from channel_type import ChannelType


class Channel:
    channel_data: NDArray
    channel_type: ChannelType
    mean: floating
    variance: floating
    skewness: float

    def __init__(self, channel_data=array([]), channel_type=ChannelType.Red) -> None:
        self.channel_type = channel_type
        self.channel_data = channel_data
