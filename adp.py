# Copied and modified from https://github.com/archinetai/audio-diffusion-pytorch/blob/v0.0.94/audio_diffusion_pytorch/modules.py under MIT License
# License can be found in LICENSES/LICENSE_ADP.txt

import math
from inspect import isfunction
from math import ceil, floor, log, pi, log2
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from packaging import version

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import Tensor, einsum
from torch.backends.cuda import sdp_kernel
from torch.nn import functional as F
from dac.nn.layers import Snake1d



################################################ Utils ################################################ 


# 定义 ConditionedSequential 类，用于按顺序执行多个模块，并可选择性地使用映射张量
class ConditionedSequential(nn.Module):
    """
    ConditionedSequential 类用于按顺序执行多个模块，并可选择性地使用映射张量。

    初始化参数:
    - *modules: 可变数量的模块，将按顺序执行。
    """
    def __init__(self, *modules):
        super().__init__()
        # 将传入的模块列表存储为 ModuleList
        self.module_list = nn.ModuleList(*modules)

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None):
        """
        前向传播方法，按顺序执行每个模块，并可选择性地使用映射张量。

        参数:
        - x (Tensor): 输入张量。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于每个模块。

        返回:
        - Tensor: 最后一个模块的输出。
        """
        for module in self.module_list:
            # 按顺序执行每个模块，并将输出传递给下一个模块
            x = module(x, mapping)
        return x


# 定义类型变量 T，用于泛型编程
T = TypeVar("T")


# 定义 default 函数，返回可选值或默认值
def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    """
    返回可选值或默认值。

    参数:
    - val (Optional[T]): 需要检查的可选值。
    - d (Union[Callable[..., T], T]): 默认值。如果 d 是可调用对象，则调用它以获取默认值。

    返回:
    - T: 如果 val 存在，则返回 val；否则返回 d 的值或调用 d() 的返回值。
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 定义 exists 函数，检查一个值是否存在（不为 None）
def exists(val: Optional[T]) -> T:
    """
    检查一个值是否存在（不为 None）。

    参数:
    - val (Optional[T]): 需要检查的值。

    返回:
    - T: 如果 val 不为 None，则返回 val。
    """
    return val is not None


# 定义 closest_power_2 函数，找到最接近输入值的2的幂
def closest_power_2(x: float) -> int:
    """
    找到最接近输入值的2的幂。

    参数:
    - x (float): 输入值。

    返回:
    - int: 最接近 x 的2的幂。
    """
    # 计算 x 的以2为底的对数
    exponent = log2(x)
    # 定义距离函数，计算 x 与 2^z 的绝对差值
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    # 找到最接近的整数指数
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


# 定义 group_dict_by_prefix 函数，根据键是否以指定前缀开头，将字典分成两部分
def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    """
    根据键是否以指定前缀开头，将字典分成两部分。

    参数:
    - prefix (str): 前缀字符串，用于判断键是否以此开头。
    - d (Dict): 输入字典。

    返回:
    - Tuple[Dict, Dict]: 返回一个元组，包含两个字典。
                         第一个字典包含不以 prefix 开头的键值对，
                         第二个字典包含以 prefix 开头的键值对。
    """
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


# 定义 groupby 函数，根据键是否以指定前缀开头，将字典分成两部分，并可选择是否保留前缀
def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    """
    根据键是否以指定前缀开头，将字典分成两部分，并可选择是否保留前缀。

    参数:
    - prefix (str): 前缀字符串，用于判断键是否以此开头。
    - d (Dict): 输入字典。
    - keep_prefix (bool, 可选): 是否保留前缀，默认为 False。

    返回:
    - Tuple[Dict, Dict]: 返回一个元组，包含两个字典。
                         如果 keep_prefix 为 False，第一个字典的键不包含 prefix；
                         如果 keep_prefix 为 True，第一个字典的键保留 prefix。
                         第二个字典始终包含以 prefix 开头的键值对。
    """
    # 使用 group_dict_by_prefix 分割字典
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs



################################################ Convolutional Blocks ################################################ 


import typing as tp

# Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py under MIT License
# License available in LICENSES/LICENSE_META.txt

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    """
    计算一维卷积所需的额外填充大小，以确保最后一个窗口是完整的。

    参数:
    - x (torch.Tensor): 输入张量，形状为 (batch_size, channels, length)。
    - kernel_size (int): 卷积核大小。
    - stride (int): 步幅。
    - padding_total (int, 可选): 总填充大小，默认为 0。

    返回:
    - int: 所需的额外填充大小。
    """
    # 获取输入张量的长度
    length = x.shape[-1]
    # 计算卷积后的帧数
    n_frames = (length - kernel_size + padding_total) / stride + 1
    # 计算理想长度，以确保最后一个窗口是完整的
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    # 返回所需的总填充大小
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    """
    为一维卷积填充输入张量，以确保最后一个窗口是完整的。
    额外的填充将添加在末尾。这是为了确保我们可以重建相同长度的输出，
    否则，即使有填充，某些时间步也可能被移除。

    参数:
    - x (torch.Tensor): 输入张量，形状为 (batch_size, channels, length)。
    - kernel_size (int): 卷积核大小。
    - stride (int): 步幅。
    - padding_total (int, 可选): 总填充大小，默认为 0。

    返回:
    - torch.Tensor: 填充后的张量。
    """
    # 计算额外填充大小
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    # 在末尾添加额外填充
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    """
    一维填充的简单封装，仅允许对小型输入进行反射填充。
    如果输入长度小于最大填充大小，则在反射填充之前添加额外的零填充。

    参数:
    - x (torch.Tensor): 输入张量。
    - paddings (Tuple[int, int]): 左右填充大小，例如 (padding_left, padding_right)。
    - mode (str, 可选): 填充模式，默认为 'constant'。
    - value (float, 可选): 填充值，默认为 0。

    返回:
    - torch.Tensor: 填充后的张量。
    """
    # 获取输入张量的长度
    length = x.shape[-1]
    # 获取左右填充大小
    padding_left, padding_right = paddings
    # 确保填充大小为非负数
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        # 获取最大填充大小
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            # 计算额外的零填充大小
            extra_pad = max_pad - length + 1
            # 在末尾添加额外的零填充
            x = F.pad(x, (0, extra_pad))
        # 进行反射填充
        padded = F.pad(x, paddings, mode, value)
        # 计算截断位置
        end = padded.shape[-1] - extra_pad
        # 返回截断后的填充张量
        return padded[..., :end]
    else:
        # 进行其他模式的填充
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    """
    从一维张量中移除填充，处理零填充。

    参数:
    - x (torch.Tensor): 输入张量。
    - paddings (Tuple[int, int]): 左右填充大小，例如 (padding_left, padding_right)。

    返回:
    - torch.Tensor: 移除填充后的张量。
    """
    # 获取左右填充大小
    padding_left, padding_right = paddings
    # 确保填充大小为非负数
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    # 确保填充大小不超过张量长度
    assert (padding_left + padding_right) <= x.shape[-1]
    # 计算移除填充后的结束位置
    end = x.shape[-1] - padding_right
    # 返回移除填充后的张量
    return x[..., padding_left: end]


class Conv1d(nn.Conv1d):
    """
    Conv1d 类继承自 torch.nn.Conv1d，并重写了前向传播方法。
    该类支持因果卷积和非对称填充，以确保卷积输出长度与输入长度一致。

    初始化参数:
    - *args: 传递给 torch.nn.Conv1d 的位置参数。
    - **kwargs: 传递给 torch.nn.Conv1d 的关键字参数。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Tensor, causal=False) -> Tensor:
        """
        前向传播方法，执行一维卷积操作，并进行必要的填充。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, in_channels, length)。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 卷积后的输出张量。
        """
        # 获取卷积核大小
        kernel_size = self.kernel_size[0]
        # 获取步幅
        stride = self.stride[0]
        # 获取膨胀率
        dilation = self.dilation[0]
        # 计算有效卷积核大小，考虑膨胀率
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        # 计算总填充大小
        padding_total = kernel_size - stride

        # 计算额外填充大小，以确保最后一个窗口是完整的
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)

        if causal:
            # Left padding for causal
            # 如果是因果卷积，则在左侧添加填充
            x = pad1d(x, (padding_total, extra_padding))
        else:
            # Asymmetric padding required for odd strides
            # 否则，进行非对称填充（适用于奇数步幅）
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding))
        return super().forward(x)
        
class ConvTranspose1d(nn.ConvTranspose1d):
    """
    ConvTranspose1d 类继承自 torch.nn.ConvTranspose1d，并重写了前向传播方法。
    该类支持因果卷积和非对称填充，以确保转置卷积输出长度与输入长度一致。

    初始化参数:
    - *args: 传递给 torch.nn.ConvTranspose1d 的位置参数。
    - **kwargs: 传递给 torch.nn.ConvTranspose1d 的关键字参数。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, causal=False) -> Tensor:
        """
        前向传播方法，执行一维转置卷积操作，并进行必要的裁剪。

        参数:
        - x (Tensor): 输入张量。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 转置卷积后的输出张量。
        """
        # 获取卷积核大小
        kernel_size = self.kernel_size[0]
        # 获取步幅
        stride = self.stride[0]
        # 计算总填充大小
        padding_total = kernel_size - stride

        # 调用父类的前向传播方法进行转置卷积操作
        y = super().forward(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # 仅裁剪固定填充。`pad_for_conv1d` 中的额外填充将在最后移除，
        # 以确保输出长度正确。
        # 如果在这里移除，需要在编码器中传递相应的长度信息。
        
        # in the encoder.
        if causal:
            # 计算右侧填充大小（向上取整）
            padding_right = ceil(padding_total)
            # 计算左侧填充大小
            padding_left = padding_total - padding_right
            # 移除填充
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            # 如果不是因果卷积，则进行非对称填充（适用于奇数步幅）
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            # 移除填充
            y = unpad1d(y, (padding_left, padding_right))
        # 返回裁剪后的输出张量
        return y
    

def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    """
    创建一维下采样层。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - factor (int): 下采样因子。
    - kernel_multiplier (int, 可选): 卷积核大小的乘数，默认为2。

    返回:
    - nn.Module: 一维下采样层。

    断言:
    - kernel_multiplier 必须为偶数，否则抛出错误。
    """
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor
    )


def Upsample1d(
    in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:
    """
    创建一维上采样层。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - factor (int): 上采样因子。
    - use_nearest (bool, 可选): 是否使用最近邻插值，默认为 False。

    返回:
    - nn.Module: 一维上采样层。

    如果 factor 为 1，则使用 3x1 卷积层进行上采样。
    如果 use_nearest 为 True，则使用最近邻插值和卷积层进行上采样。
    否则，使用转置卷积层进行上采样。
    """

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )

    if use_nearest:
        return nn.Sequential(
            # 最近邻插值上采样
            nn.Upsample(scale_factor=factor, mode="nearest"),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor
        )


class ConvBlock1d(nn.Module):
    """
    ConvBlock1d 类实现了一维卷积块，包含组归一化、激活函数和卷积层。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    - stride (int, 可选): 步幅，默认为1。
    - dilation (int, 可选): 膨胀率，默认为1。
    - num_groups (int, 可选): 组归一化的组数，默认为8。
    - use_norm (bool, 可选): 是否使用组归一化，默认为 True。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
        use_norm: bool = True,
        use_snake: bool = False
    ) -> None:
        super().__init__()
        # 定义组归一化层，如果 use_norm 为 False，则使用恒等映射
        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        # 定义激活函数，如果 use_snake 为 True，则使用 Snake 激活函数；否则，使用 SiLU 激活函数
        if use_snake:
            self.activation = Snake1d(in_channels)
        else:
            self.activation = nn.SiLU() 
        # 定义卷积层
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None, causal=False
    ) -> Tensor:
        """
        前向传播方法，执行组归一化、激活函数和卷积操作。

        参数:
        - x (Tensor): 输入张量。
        - scale_shift (Optional[Tuple[Tensor, Tensor]], 可选): 可选的缩放和偏移量。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 卷积后的输出张量。
        """
        # 应用组归一化
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x, causal=causal)


class MappingToScaleShift(nn.Module):
    """
    MappingToScaleShift 类用于将映射张量转换为缩放和偏移量。

    初始化参数:
    - features (int): 映射张量的特征维度。
    - channels (int): 输出通道数，用于生成缩放和偏移量。
    """
    def __init__(
        self,
        features: int,
        channels: int,
    ):
        super().__init__()
        # 定义一个序列模块，用于将映射张量转换为缩放和偏移量
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            # 全连接层，将特征维度映射到 2 倍的通道数
            nn.Linear(in_features=features, out_features=channels * 2),
        )

    def forward(self, mapping: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播方法，将映射张量转换为缩放和偏移量。

        参数:
        - mapping (Tensor): 输入的映射张量，形状为 (batch_size, features)。

        返回:
        - Tuple[Tensor, Tensor]: 返回一个元组，包含缩放量和偏移量，形状均为 (batch_size, channels, 1)。
        """
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, "b c -> b c 1")
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift


class ResnetBlock1d(nn.Module):
    """
    ResnetBlock1d 类实现了一维残差块，包含两个卷积块和一个残差连接。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    - stride (int, 可选): 步幅，默认为1。
    - dilation (int, 可选): 膨胀率，默认为1。
    - use_norm (bool, 可选): 是否使用组归一化，默认为 True。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    - num_groups (int, 可选): 组归一化的组数，默认为8。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度。如果提供，则使用映射张量生成缩放和偏移量。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        use_snake: bool = False,
        num_groups: int = 8,
        context_mapping_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_mapping = exists(context_mapping_features)

        # 定义第一个卷积块
        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
            use_snake=use_snake
        )

        if self.use_mapping:
            # 确保提供了上下文映射特征维度
            assert exists(context_mapping_features)
            # 定义映射到缩放和偏移量的模块
            self.to_scale_shift = MappingToScaleShift(
                # 上下文映射特征维度、输出通道数
                features=context_mapping_features, channels=out_channels
            )

        # 定义第二个卷积块
        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
            use_snake=use_snake
        )

        # 定义输出映射层，如果输入通道数与输出通道数不同，则使用 1x1 卷积层；否则，使用恒等映射
        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        """
        前向传播方法，执行一维残差块操作。

        参数:
        - x (Tensor): 输入张量。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于生成缩放和偏移量。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 残差块的输出张量。
        """
        assert_message = "context mapping required if context_mapping_features > 0"
        # 确保映射张量的存在性
        assert not (self.use_mapping ^ exists(mapping)), assert_message

        # 应用第一个卷积块
        h = self.block1(x, causal=causal)

        # 初始化缩放和偏移量
        scale_shift = None
        if self.use_mapping:
            # 使用映射张量生成缩放和偏移量
            scale_shift = self.to_scale_shift(mapping)

        # 应用第二个卷积块
        h = self.block2(h, scale_shift=scale_shift, causal=causal)
        # 应用残差连接并返回结果
        return h + self.to_out(x)


class Patcher(nn.Module):
    """
    Patcher 类用于将一维输入张量分割成多个块（patch），并对每个块应用残差块。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - patch_size (int): 每个块的大小。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度。如果提供，则使用映射张量生成缩放和偏移量。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None,
        use_snake: bool = False,
    ):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        # 确保输出通道数可以被块大小整除
        assert out_channels % patch_size == 0, assert_message
        # 存储块大小
        self.patch_size = patch_size

        # 定义残差块，输出通道数除以块大小
        self.block = ResnetBlock1d(
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数除以块大小
            out_channels=out_channels // patch_size,
            # 组归一化的组数
            num_groups=1,
            # 上下文映射特征维度
            context_mapping_features=context_mapping_features,
            # 是否使用 Snake 激活函数
            use_snake=use_snake
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        """
        前向传播方法，将输入张量分割成块并应用残差块。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, in_channels, length)。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于生成缩放和偏移量。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 输出张量，形状为 (batch_size, out_channels, length // patch_size)。
        """
        # 应用残差块
        x = self.block(x, mapping, causal=causal)
        # 将张量重塑为 (batch_size, out_channels // patch_size, length, patch_size)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        # 返回重塑后的张量
        return x


class Unpatcher(nn.Module):
    """
    Unpatcher 类用于将一维输入张量中的块（patch）重新组合成原始张量，并对每个块应用残差块。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - patch_size (int): 每个块的大小。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度。如果提供，则使用映射张量生成缩放和偏移量。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None,
        use_snake: bool = False
    ):
        super().__init__()
        assert_message = f"in_channels must be divisible by patch_size ({patch_size})"
        # 确保输入通道数可以被块大小整除
        assert in_channels % patch_size == 0, assert_message
        # 存储块大小
        self.patch_size = patch_size

        # 定义残差块，输入通道数除以块大小
        self.block = ResnetBlock1d(
            # 输入通道数除以块大小
            in_channels=in_channels // patch_size,
            # 输出通道数
            out_channels=out_channels,
            # 组归一化的组数
            num_groups=1,
            # 上下文映射特征维度
            context_mapping_features=context_mapping_features,
            # 是否使用 Snake 激活函数
            use_snake=use_snake
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        """
        前向传播方法，将输入张量中的块重新组合成原始张量，并对每个块应用残差块。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, in_channels, length // patch_size)。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于生成缩放和偏移量。
        - causal (bool, 可选): 是否进行因果卷积，默认为 False。

        返回:
        - Tensor: 输出张量，形状为 (batch_size, out_channels, length)。
        """
        # 将张量重塑为 (batch_size, in_channels // patch_size, length, patch_size)
        x = rearrange(x, " b (c p) l -> b c (l p) ", p=self.patch_size)
        # 应用残差块
        x = self.block(x, mapping, causal=causal)
        return x



################################################ Attention Components ################################################ 


# 定义前馈网络（FeedForward）层
def FeedForward(features: int, multiplier: int) -> nn.Module:
    """
    创建前馈网络层，包含两个线性层和一个 GELU 激活函数。

    参数:
    - features (int): 输入特征的维度。
    - multiplier (int): 中间层特征的维度乘数。

    返回:
    - nn.Module: 前馈网络层。
    """
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


# 定义添加掩码的函数，用于掩码注意力分数
def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    """
    在注意力分数矩阵中添加掩码。

    参数:
    - sim (Tensor): 注意力分数矩阵，形状为 (batch_size, n, m)。
    - mask (Tensor): 掩码矩阵，形状为 (batch_size, n, m) 或 (n, m)。

    返回:
    - Tensor: 添加掩码后的注意力分数矩阵。
    """
    # 获取批次大小和掩码维度
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        # 如果掩码是 (batch_size, n, m)，则重塑为 (batch_size, 1, n, m)
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        # 如果掩码是 (n, m)，则重复批次维度
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    # 获取数据类型允许的最小值
    max_neg_value = -torch.finfo(sim.dtype).max
    # 将掩码为 False 的位置设置为最小值
    sim = sim.masked_fill(~mask, max_neg_value)
    # 返回添加掩码后的注意力分数矩阵
    return sim


# 定义因果掩码的函数，用于因果注意力
def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    """
    生成因果掩码，阻止模型关注未来的时间步。

    参数:
    - q (Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len, head_features)。
    - k (Tensor): 键张量，形状为 (batch_size, num_heads, seq_len, head_features)。

    返回:
    - Tensor: 因果掩码矩阵，形状为 (batch_size, seq_len, seq_len)。
    """
    # 获取批次大小、序列长度和设备
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    # 生成一个上三角矩阵，值为 False，表示不允许模型关注未来的时间步
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    # 重复批次维度
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask


class AttentionBase(nn.Module):
    """
    AttentionBase 类实现了多头注意力机制的基础功能。

    初始化参数:
    - features (int): 输入特征的维度。
    - head_features (int): 每个注意力头的特征维度。
    - num_heads (int): 注意力头的数量。
    - out_features (Optional[int], 可选): 输出特征的维度。如果未提供，则默认为输入特征的维度。
    """
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        # 缩放因子，用于缩放注意力分数
        self.scale = head_features**-0.5
        # 注意力头的数量
        self.num_heads = num_heads
        # 中间特征的维度
        mid_features = head_features * num_heads
        # 如果未提供输出特征的维度，则默认为输入特征的维度
        out_features = default(out_features, features)

        # 定义输出线性层
        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features
        )
        # 检查是否可以使用 Flash Attention
        self.use_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        if not self.use_flash:
            # 如果不使用 Flash Attention，则返回
            return
        # 获取 CUDA 设备属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            # 如果是 A100 GPU，则使用 Flash Attention
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            # 否则，不使用 Flash Attention
            self.sdp_kernel_config = (False, True, True)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        """
        前向传播方法，执行多头注意力机制。

        参数:
        - q (Tensor): 查询张量。
        - k (Tensor): 键张量。
        - v (Tensor): 值张量。
        - mask (Optional[Tensor], 可选): 可选的掩码张量。
        - is_causal (bool, 可选): 是否进行因果注意力，默认为 False。

        返回:
        - Tensor: 注意力机制的输出张量。
        """
        # 分割多头
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                # 如果进行因果注意力且未提供掩码，则生成因果掩码
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            # 计算相似度矩阵，并添加掩码
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if exists(mask) else sim

            # Get attention matrix with softmax
            # 计算注意力权重
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # Compute values
            # 计算输出
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            with sdp_kernel(*self.sdp_kernel_config):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        # 重塑输出张量
        out = rearrange(out, "b h n d -> b n (h d)")
        # 应用输出线性层并返回结果
        return self.to_out(out)


# 定义 Attention 类，实现多头注意力机制
class Attention(nn.Module):
    """
    Attention 类实现了多头注意力机制，支持上下文特征和因果掩码。

    初始化参数:
    - features (int): 输入特征的维度。
    - head_features (int): 每个注意力头的特征维度。
    - num_heads (int): 注意力头的数量。
    - out_features (Optional[int], 可选): 输出特征的维度。如果未提供，则默认为输入特征的维度。
    - context_features (Optional[int], 可选): 上下文特征的维度。如果未提供，则默认为输入特征的维度。
    - causal (bool, 可选): 是否进行因果注意力，默认为 False。
    """
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        # 存储上下文特征的维度
        self.context_features = context_features
        # 存储是否进行因果注意力
        self.causal = causal
        # 中间特征的维度
        mid_features = head_features * num_heads
        # 如果未提供上下文特征的维度，则默认为输入特征的维度
        context_features = default(context_features, features)

        # 定义输入特征的层归一化
        self.norm = nn.LayerNorm(features)
        # 定义上下文特征的层归一化
        self.norm_context = nn.LayerNorm(context_features)
        # 定义查询（q）线性层
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )       
        # 定义键（k）和值（v）线性层
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        # 定义注意力机制
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor, # [b, n, c]
        context: Optional[Tensor] = None, # [b, m, d]
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
        causal: Optional[bool] = False,
    ) -> Tensor:
        """
        前向传播方法，执行多头注意力机制。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, seq_len, features)。
        - context (Optional[Tensor], 可选): 可选的上下文张量，形状为 (batch_size, context_len, context_features)。
        - context_mask (Optional[Tensor], 可选): 可选的上下文掩码，形状为 (batch_size, context_len)，False 表示被掩码。
        - causal (Optional[bool], 可选): 是否进行因果注意力，默认为 False。

        返回:
        - Tensor: 注意力机制的输出张量，形状为 (batch_size, seq_len, out_features)。
        """
        assert_message = "You must provide a context when using context_features"
        # 确保提供了上下文张量
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        # 如果未提供上下文张量，则使用输入张量
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        # 对输入张量和上下文张量进行层归一化
        x, context = self.norm(x), self.norm_context(context)

        # 计算查询（q）、键（k）和值（v）
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            # 如果提供了上下文掩码，则对键（k）和值（v）进行掩码
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask

        # Compute and return attention
        # 计算注意力并返回结果
        return self.attention(q, k, v, is_causal=self.causal or causal)


# 定义前馈网络（FeedForward）层
def FeedForward(features: int, multiplier: int) -> nn.Module:
    """
    创建前馈网络层，包含两个线性层和一个 GELU 激活函数。

    参数:
    - features (int): 输入特征的维度。
    - multiplier (int): 中间层特征的维度乘数。

    返回:
    - nn.Module: 前馈网络层。
    """
    # 中间层的特征维度
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )



################################################ Transformer Blocks ################################################ 


class TransformerBlock(nn.Module):
    """
    TransformerBlock 类实现了 Transformer 块，包含自注意力机制、交叉注意力机制和前馈网络。

    初始化参数:
    - features (int): 输入特征的维度。
    - num_heads (int): 注意力头的数量。
    - head_features (int): 每个注意力头的特征维度。
    - multiplier (int): 前馈网络中中间层特征的维度乘数。
    - context_features (Optional[int], 可选): 交叉注意力中上下文特征的维度。如果提供，则使用交叉注意力。
    """
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()
        # 判断是否使用交叉注意力
        self.use_cross_attention = exists(context_features) and context_features > 0
        # 定义自注意力机制
        self.attention = Attention(
            # 输入特征的维度
            features=features,
            # 注意力头的数量
            num_heads=num_heads,
            # 每个注意力头的特征维度
            head_features=head_features
        )

        if self.use_cross_attention:
            # 如果使用交叉注意力，则定义交叉注意力机制
            self.cross_attention = Attention(
                # 输入特征的维度
                features=features,
                # 注意力头的数量
                num_heads=num_heads,
                # 每个注意力头的特征维度
                head_features=head_features,
                # 上下文特征的维度
                context_features=context_features
            )
        # 定义前馈网络
        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal: Optional[bool] = False) -> Tensor:
        """
        前向传播方法，执行自注意力、交叉注意力和前馈网络操作。

        参数:
        - x (Tensor): 输入张量。
        - context (Optional[Tensor], 可选): 可选的上下文张量。
        - context_mask (Optional[Tensor], 可选): 可选的上下文掩码。
        - causal (Optional[bool], 可选): 是否进行因果注意力。

        返回:
        - Tensor: Transformer 块的输出张量。
        """
        # 自注意力机制
        # 应用自注意力并添加残差连接
        x = self.attention(x, causal=causal) + x
        if self.use_cross_attention:
            # 应用交叉注意力并添加残差连接
            x = self.cross_attention(x, context=context, context_mask=context_mask) + x
        # 应用前馈网络并添加残差连接
        x = self.feed_forward(x) + x
        return x



################################################ Transformers ################################################ 


class Transformer1d(nn.Module):
    """
    Transformer1d 类实现了基于 Transformer 的 1D 模型，适用于序列数据处理。

    初始化参数:
    - num_layers (int): Transformer 块的数量。
    - channels (int): 输入和输出的通道数。
    - num_heads (int): 注意力头的数量。
    - head_features (int): 每个注意力头的特征维度。
    - multiplier (int): 前馈网络中中间层特征的维度乘数。
    - context_features (Optional[int], 可选): 交叉注意力中上下文特征的维度。如果提供，则使用交叉注意力。
    """
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()
        # 定义输入预处理模块，包括组归一化、1x1 卷积和重排
        self.to_in = nn.Sequential(
            # 组归一化
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            # 重排张量形状为 (batch_size, seq_len, channels)
            Rearrange("b c t -> b t c"),
        )
        # 定义 Transformer 块列表
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    # 输入和输出的特征维度
                    features=channels,
                    # 每个注意力头的特征维度
                    head_features=head_features,
                    # 注意力头的数量
                    num_heads=num_heads,
                    # 前馈网络中中间层特征的维度乘数
                    multiplier=multiplier,
                    # 交叉注意力中上下文特征的维度
                    context_features=context_features,
                )
                # 根据层数创建多个 Transformer 块
                for i in range(num_layers)
            ]
        )
        # 定义输出后处理模块，包括重排和 1x1 卷积
        self.to_out = nn.Sequential(
            # 重排张量形状为 (batch_size, channels, seq_len)
            Rearrange("b t c -> b c t"),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal=False) -> Tensor:
        """
        前向传播方法，执行 Transformer 模型的前向计算。

        参数:
        - x (Tensor): 输入张量。
        - context (Optional[Tensor], 可选): 可选的上下文张量。
        - context_mask (Optional[Tensor], 可选): 可选的上下文掩码。
        - causal (bool, 可选): 是否进行因果注意力，默认为 False。

        返回:
        - Tensor: Transformer 模型的输出张量。
        """
        # 输入预处理
        x = self.to_in(x)
        # 遍历每个 Transformer 块
        for block in self.blocks:
            # 应用 Transformer 块
            x = block(x, context=context, context_mask=context_mask, causal=causal)
        # 输出后处理
        x = self.to_out(x)
        return x



################################################ Time Embeddings ################################################ 


class SinusoidalEmbedding(nn.Module):
    """
    SinusoidalEmbedding 类实现了正弦位置编码，用于将时间步转换为正弦波编码。

    初始化参数:
    - dim (int): 编码的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，将输入时间步转换为正弦波编码。

        参数:
        - x (Tensor): 输入时间步，形状为 (batch_size, seq_len)。

        返回:
        - Tensor: 正弦波编码后的张量，形状为 (batch_size, seq_len, dim)。
        """
        # 计算编码维度的一半
        device, half_dim = x.device, self.dim // 2
        # 计算频率，公式为 log(10000) / (half_dim - 1)
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        # 生成指数衰减的频率
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入时间步和频率进行广播相乘
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        # 计算正弦和余弦编码，并拼接起来
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""
    """
    LearnedPositionalEmbedding 类实现了可学习的位置编码，适用于连续时间。

    初始化参数:
    - dim (int): 编码的维度，必须是偶数。
    """
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        # 计算编码维度的一半
        half_dim = dim // 2
        # 可学习的权重参数
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，将输入时间步转换为可学习的位置编码。

        参数:
        - x (Tensor): 输入时间步，形状为 (batch_size, seq_len)。

        返回:
        - Tensor: 可学习的位置编码后的张量，形状为 (batch_size, seq_len, dim + 1)。
        """
        # 重塑时间步为 (batch_size, 1)
        x = rearrange(x, "b -> b 1")
        # 计算频率，公式为 x * weights * 2 * pi
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        # 计算正弦和余弦编码，并拼接起来
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # 将原始时间步和频率编码拼接起来
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    """
    TimePositionalEmbedding 函数用于创建时间位置编码模块。

    参数:
    - dim (int): 输入时间步的维度。
    - out_features (int): 输出特征的维度。

    返回:
    - nn.Module: 时间位置编码模块，包含可学习的位置编码和线性层。
    """
    return nn.Sequential(
        # 可学习的位置编码
        LearnedPositionalEmbedding(dim),
        # 线性层，将维度映射到 out_features
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )



################################################ Encoder/Decoder Components ################################################ 


class DownsampleBlock1d(nn.Module):
    """
    DownsampleBlock1d 类实现了 1D 下采样块，包含下采样层、残差块、Transformer 块和提取层。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - factor (int): 下采样因子，用于下采样层。
    - num_groups (int): 组归一化的组数，用于残差块和提取层。
    - num_layers (int): 残差块的数量。
    - kernel_multiplier (int, 可选): 卷积核大小的乘数，默认为 2。
    - use_pre_downsample (bool, 可选): 是否在残差块之前进行下采样，默认为 True。
    - use_skip (bool, 可选): 是否使用跳跃连接，默认为 False。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    - extract_channels (int, 可选): 提取层的输出通道数。如果为 0，则不使用提取层。
    - context_channels (int, 可选): 上下文通道数，用于残差块的上下文映射。如果为 0，则不使用上下文。
    - num_transformer_blocks (int, 可选): Transformer 块的数量。如果为 0，则不使用 Transformer。
    - attention_heads (Optional[int], 可选): 注意力头的数量。如果未提供，则根据 attention_features 计算。
    - attention_features (Optional[int], 可选): 每个注意力头的特征维度。如果未提供，则根据 attention_heads 计算。
    - attention_multiplier (Optional[int], 可选): 前馈网络中中间层特征的维度乘数。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度，用于残差块的上下文映射。
    - context_embedding_features (Optional[int], 可选): 上下文嵌入特征维度，用于 Transformer 块。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        use_snake: bool = False,
        extract_channels: int = 0,
        context_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        # 是否在残差块之前进行下采样
        self.use_pre_downsample = use_pre_downsample
        # 是否使用跳跃连接
        self.use_skip = use_skip
        # 是否使用 Transformer
        self.use_transformer = num_transformer_blocks > 0
        # 是否使用提取层
        self.use_extract = extract_channels > 0
        # 是否使用上下文
        self.use_context = context_channels > 0
        # 计算通道数
        channels = out_channels if use_pre_downsample else in_channels
        # 定义下采样层
        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )
        # 定义残差块列表
        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    # 第一层残差块的输入通道数包含上下文通道
                    in_channels=channels + context_channels if i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                # 根据层数创建多个残差块
                for i in range(num_layers)
            ]
        )

        if self.use_transformer:
            # 如果使用 Transformer，则进行assert，确保提供了必要的参数
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                # 计算每个注意力头的特征维度
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                # 计算注意力头的数量
                attention_heads = channels // attention_features

            # 定义 Transformer 块
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features
            )

        if self.use_extract:
            # 计算提取层的组数
            num_extract_groups = min(num_groups, extract_channels)
            # 定义提取层
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        channels: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:
        """
        前向传播方法，执行下采样块的操作。

        参数:
        - x (Tensor): 输入张量。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于残差块的上下文映射。
        - channels (Optional[Tensor], 可选): 可选的上下文通道张量。
        - embedding (Optional[Tensor], 可选): 可选的嵌入张量，用于 Transformer 块。
        - embedding_mask (Optional[Tensor], 可选): 可选的嵌入掩码，用于 Transformer 块。
        - causal (Optional[bool], 可选): 是否进行因果注意力。

        返回:
        - Union[Tuple[Tensor, List[Tensor]], Tensor]: 如果使用跳跃连接，则返回输出张量和跳跃连接列表；否则，返回输出张量。
        """

        if self.use_pre_downsample:
            # 应用下采样层
            x = self.downsample(x)

        if self.use_context and exists(channels):
            # 拼接上下文通道
            x = torch.cat([x, channels], dim=1)

        # 初始化跳跃连接列表
        skips = []
        # 遍历每个残差块
        for block in self.blocks:
            # 应用残差块
            x = block(x, mapping=mapping, causal=causal)
            # 添加跳跃连接
            skips += [x] if self.use_skip else []

        if self.use_transformer:
            # 应用 Transformer 块
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
            # 添加跳跃连接
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            # 应用下采样层
            x = self.downsample(x)

        if self.use_extract:
            # 应用提取层
            extracted = self.to_extracted(x)
            # 返回输出张量和提取结果
            return x, extracted
        # 如果使用跳跃连接，则返回输出张量和跳跃连接列表；否则，返回输出张量
        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    """
    UpsampleBlock1d 类实现了 1D 上采样块，包含上采样层、残差块、Transformer 块和提取层。

    初始化参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - factor (int): 上采样因子，用于上采样层。
    - num_layers (int): 残差块的数量。
    - num_groups (int): 组归一化的组数，用于残差块和提取层。
    - use_nearest (bool, 可选): 是否使用最近邻插值进行上采样，默认为 False。
    - use_pre_upsample (bool, 可选): 是否在上采样之前进行上采样，默认为 False。
    - use_skip (bool, 可选): 是否使用跳跃连接，默认为 False。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    - skip_channels (int, 可选): 跳跃连接通道数，默认为 0。
    - use_skip_scale (bool, 可选): 是否对跳跃连接进行缩放，默认为 False。
    - extract_channels (int, 可选): 提取层的输出通道数。如果为 0，则不使用提取层。
    - num_transformer_blocks (int, 可选): Transformer 块的数量。如果为 0，则不使用 Transformer。
    - attention_heads (Optional[int], 可选): 注意力头的数量。如果未提供，则根据 attention_features 计算。
    - attention_features (Optional[int], 可选): 每个注意力头的特征维度。如果未提供，则根据 attention_heads 计算。
    - attention_multiplier (Optional[int], 可选): 前馈网络中中间层特征的维度乘数。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度，用于残差块的上下文映射。
    - context_embedding_features (Optional[int], 可选): 上下文嵌入特征维度，用于 Transformer 块。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        use_snake: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        extract_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        # 是否使用提取层
        self.use_extract = extract_channels > 0
        # 是否在上采样之前进行上采样
        self.use_pre_upsample = use_pre_upsample
        # 是否使用 Transformer
        self.use_transformer = num_transformer_blocks > 0
        # 是否使用跳跃连接
        self.use_skip = use_skip
        # 跳跃连接缩放因子
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0
        # 计算通道数
        channels = out_channels if use_pre_upsample else in_channels
        # 定义残差块列表
        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                # 根据层数创建多个残差块
                for _ in range(num_layers)
            ]
        )

        if self.use_transformer:
            # 如果使用 Transformer，则进行断言，确保提供了必要的参数
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                # 计算每个注意力头的特征维度
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                # 计算注意力头的数量
                attention_heads = channels // attention_features
            # 定义 Transformer 块
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )
        # 定义上采样层
        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

        if self.use_extract:
            # 计算提取层的组数
            num_extract_groups = min(num_groups, extract_channels)
            # 定义提取层
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        添加跳跃连接。

        参数:
        - x (Tensor): 当前层输出。
        - skip (Tensor): 跳跃连接张量。

        返回:
        - Tensor: 添加跳跃连接后的张量。
        """
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        *,
        skips: Optional[List[Tensor]] = None,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        前向传播方法，执行上采样块的操作。

        参数:
        - x (Tensor): 输入张量。
        - skips (Optional[List[Tensor]], 可选): 可选的跳跃连接列表。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于残差块的上下文映射。
        - embedding (Optional[Tensor], 可选): 可选的嵌入张量，用于 Transformer 块。
        - embedding_mask (Optional[Tensor], 可选): 可选的嵌入掩码，用于 Transformer 块。
        - causal (Optional[bool], 可选): 是否进行因果注意力。

        返回:
        - Union[Tuple[Tensor, Tensor], Tensor]: 如果使用提取层，则返回输出张量和提取结果；否则，返回输出张量。
        """

        if self.use_pre_upsample:
            # 应用上采样层
            x = self.upsample(x)
        # 遍历每个残差块
        for block in self.blocks:
            # 添加跳跃连接
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            # 应用残差块
            x = block(x, mapping=mapping, causal=causal)

        if self.use_transformer:
            # 应用 Transformer 块
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)

        if not self.use_pre_upsample:
            # 应用上采样层
            x = self.upsample(x)

        if self.use_extract:
            # 应用提取层
            extracted = self.to_extracted(x)
            return x, extracted

        return x


class BottleneckBlock1d(nn.Module):
    """
    BottleneckBlock1d 类实现了瓶颈块（Bottleneck Block），包含前处理残差块、Transformer 块和后处理残差块。

    初始化参数:
    - channels (int): 输入和输出的通道数。
    - num_groups (int): 组归一化的组数，用于残差块。
    - num_transformer_blocks (int, 可选): Transformer 块的数量。如果为 0，则不使用 Transformer，默认为 0。
    - attention_heads (Optional[int], 可选): 注意力头的数量。如果未提供，则根据 attention_features 计算。
    - attention_features (Optional[int], 可选): 每个注意力头的特征维度。如果未提供，则根据 attention_heads 计算。
    - attention_multiplier (Optional[int], 可选): 前馈网络中中间层特征的维度乘数。
    - context_mapping_features (Optional[int], 可选): 上下文映射特征维度，用于残差块的上下文映射。
    - context_embedding_features (Optional[int], 可选): 上下文嵌入特征维度，用于 Transformer 块。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    """
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        use_snake: bool = False,
    ):
        super().__init__()
        # 判断是否使用 Transformer
        self.use_transformer = num_transformer_blocks > 0
        # 定义前处理残差块
        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        if self.use_transformer:
            # 如果使用 Transformer，则进行断言，确保提供了必要的参数
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                # 计算每个注意力头的特征维度
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                # 计算注意力头的数量
                attention_heads = channels // attention_features
            # 定义 Transformer 块
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )
        # 定义后处理残差块
        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Tensor:
        """
        前向传播方法，执行瓶颈块的操作。

        参数:
        - x (Tensor): 输入张量。
        - mapping (Optional[Tensor], 可选): 可选的映射张量，用于残差块的上下文映射。
        - embedding (Optional[Tensor], 可选): 可选的嵌入张量，用于 Transformer 块。
        - embedding_mask (Optional[Tensor], 可选): 可选的嵌入掩码，用于 Transformer 块。
        - causal (Optional[bool], 可选): 是否进行因果注意力。

        返回:
        - Tensor: 瓶颈块的输出张量。
        """
        # 应用前处理残差块
        x = self.pre_block(x, mapping=mapping, causal=causal)
        if self.use_transformer:
            # 应用 Transformer 块
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
        # 应用后处理残差块
        x = self.post_block(x, mapping=mapping, causal=causal)
        return x



################################################ UNet ################################################ 


class UNet1d(nn.Module):
    """
    UNet1d 类实现了一个一维的 U-Net 模型，包含编码器、下采样层、瓶颈层、上采样层和解码器。

    初始化参数:
    - in_channels (int): 输入通道数。
    - channels (int): 初始通道数，用于构建 U-Net 的各层。
    - multipliers (Sequence[int]): 每个下采样阶段的通道数乘数列表，用于调整每个阶段的通道数。
    - factors (Sequence[int]): 每个下采样阶段的下采样因子列表，用于调整每个阶段的空间分辨率。
    - num_blocks (Sequence[int]): 每个下采样阶段的残差块数量列表。
    - attentions (Sequence[int]): 每个下采样阶段的 Transformer 块数量列表。如果为 0，则不使用 Transformer。
    - patch_size (int, 可选): 块大小，用于将输入张量分割成块，默认为 1。
    - resnet_groups (int, 可选): 组归一化的组数，用于残差块，默认为 8。
    - use_context_time (bool, 可选): 是否使用时间上下文信息，默认为 True。
    - kernel_multiplier_downsample (int, 可选): 下采样卷积核大小的乘数，默认为 2。
    - use_nearest_upsample (bool, 可选): 是否使用最近邻插值进行上采样，默认为 False。
    - use_skip_scale (bool, 可选): 是否对跳跃连接进行缩放，默认为 True。
    - use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
    - use_stft (bool, 可选): 是否使用 STFT 进行频域特征提取，默认为 False。
    - use_stft_context (bool, 可选): 是否对上下文通道应用 STFT，默认为 False。
    - out_channels (Optional[int], 可选): 输出通道数。如果未提供，则默认为输入通道数。
    - context_features (Optional[int], 可选): 上下文特征的维度。如果提供，则使用上下文特征。
    - context_features_multiplier (int, 可选): 上下文特征的维度乘数，默认为 4。
    - context_channels (Optional[Sequence[int]], 可选): 每个阶段的上下文通道数列表。如果未提供，则不使用上下文通道。
    - context_embedding_features (Optional[int], 可选): 上下文嵌入特征的维度，用于 Transformer 块。
    - **kwargs: 其他关键字参数。
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        use_context_time: bool = True,
        kernel_multiplier_downsample: int = 2,
        use_nearest_upsample: bool = False,
        use_skip_scale: bool = True,
        use_snake: bool = False,
        use_stft: bool = False,
        use_stft_context: bool = False,
        out_channels: Optional[int] = None,
        context_features: Optional[int] = None,
        context_features_multiplier: int = 4,
        context_channels: Optional[Sequence[int]] = None,
        context_embedding_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # 如果未提供输出通道数，则默认为输入通道数
        out_channels = default(out_channels, in_channels)
        # 如果未提供上下文通道数，则默认为空列表
        context_channels = list(default(context_channels, []))
        # 计算 U-Net 的层数
        num_layers = len(multipliers) - 1
        # 是否使用上下文特征
        use_context_features = exists(context_features)
        # 是否使用上下文通道
        use_context_channels = len(context_channels) > 0
        # 初始化上下文映射特征维度
        context_mapping_features = None
        # 从关键字参数中提取注意力相关的参数
        attention_kwargs, kwargs = groupby("attention_", kwargs, keep_prefix=True)

        # 存储 U-Net 的层数
        self.num_layers = num_layers
        # 存储是否使用时间上下文信息
        self.use_context_time = use_context_time
        # 存储是否使用上下文特征
        self.use_context_features = use_context_features
        # 存储是否使用上下文通道
        self.use_context_channels = use_context_channels
        # 存储是否使用 STFT
        self.use_stft = use_stft
        # 存储是否对上下文通道应用 STFT
        self.use_stft_context = use_stft_context

        # 存储上下文特征的维度
        self.context_features = context_features
        # 计算上下文通道数列表的长度
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        # 填充上下文通道数列表
        context_channels = context_channels + [0] * context_channels_pad_length
        # 存储上下文通道数列表
        self.context_channels = context_channels
        # 存储上下文嵌入特征的维度
        self.context_embedding_features = context_embedding_features

        if use_context_channels:
            # 判断每个阶段的上下文通道数是否大于0
            has_context = [c > 0 for c in context_channels]
            # 存储上下文通道数是否存在的列表
            self.has_context = has_context
            # 计算每个阶段的上下文通道索引
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]

        assert (
            # 确保下采样因子列表的长度与层数匹配
            len(factors) == num_layers
            # 确保 Transformer 块数量列表的长度不小于层数
            and len(attentions) >= num_layers
            # 确保残差块数量列表的长度与层数匹配
            and len(num_blocks) == num_layers
        )

        if use_context_time or use_context_features:
            # 计算上下文映射特征维度
            context_mapping_features = channels * context_features_multiplier

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            # 确保上下文映射特征维度存在
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            # 确保上下文特征和上下文映射特征维度存在
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_stft:
            # 从关键字参数中提取 STFT 相关的参数
            stft_kwargs, kwargs = groupby("stft_", kwargs)
            assert "num_fft" in stft_kwargs, "stft_num_fft required if use_stft=True"
            stft_channels = (stft_kwargs["num_fft"] // 2 + 1) * 2
            in_channels *= stft_channels
            out_channels *= stft_channels
            context_channels[0] *= stft_channels if use_stft_context else 1
            assert exists(in_channels) and exists(out_channels)
            self.stft = STFT(**stft_kwargs)

        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"

        self.to_in = Patcher(
            in_channels=in_channels + context_channels[0],
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    context_mapping_features=context_mapping_features,
                    context_channels=context_channels[i + 1],
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    use_snake=use_snake,
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            context_mapping_features=context_mapping_features,
            context_embedding_features=context_embedding_features,
            num_groups=resnet_groups,
            num_transformer_blocks=attentions[-1],
            use_snake=use_snake,
            **attention_kwargs,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    context_mapping_features=context_mapping_features,
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    use_snake=use_snake,
                    skip_channels=channels * multipliers[i + 1],
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[0],
            out_channels=out_channels,
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def get_channels(
        self, channels_list: Optional[Sequence[Tensor]] = None, layer: int = 0
    ) -> Optional[Tensor]:
        """Gets context channels at `layer` and checks that shape is correct"""
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), "Missing context"
        # Get channels index (skipping zero channel contexts)
        channels_id = self.channels_ids[layer]
        # Get channels
        channels = channels_list[channels_id]
        message = f"Missing context for layer {layer} at index {channels_id}"
        assert exists(channels), message
        # Check channels
        num_channels = self.context_channels[layer]
        message = f"Expected context with {num_channels} channels at idx {channels_id}"
        assert channels.shape[1] == num_channels, message
        # STFT channels if requested
        channels = self.stft.encode1d(channels) if self.use_stft_context else channels  # type: ignore # noqa
        return channels

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]
        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        *,
        features: Optional[Tensor] = None,
        channels_list: Optional[Sequence[Tensor]] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False,
    ) -> Tensor:
        channels = self.get_channels(channels_list, layer=0)
        # Apply stft if required
        x = self.stft.encode1d(x) if self.use_stft else x  # type: ignore
        # Concat context channels at layer 0 if provided
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        # Compute mapping from time and features
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping, causal=causal)
        skips_list = [x]

        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i + 1)
            x, skips = downsample(
                x, mapping=mapping, channels=channels, embedding=embedding, embedding_mask=embedding_mask, causal=causal
            )
            skips_list += [skips]

        x = self.bottleneck(x, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        x += skips_list.pop()
        x = self.to_out(x, mapping, causal=causal)
        x = self.stft.decode1d(x) if self.use_stft else x

        return x


################################################ Conditioning Modules ################################################ 


class FixedEmbedding(nn.Module):
    """
    FixedEmbedding 类实现了固定长度的位置嵌入。

    初始化参数:
    - max_length (int): 序列的最大长度。
    - features (int): 嵌入特征的维度。
    """
    def __init__(self, max_length: int, features: int):
        super().__init__()
        # 存储序列的最大长度
        self.max_length = max_length
        # 定义嵌入层，生成固定长度的位置嵌入
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，根据输入张量的长度生成位置嵌入。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, seq_len, ...)。

        返回:
        - Tensor: 位置嵌入张量，形状为 (batch_size, seq_len, features)。
        """
        # 获取批次大小、序列长度和设备
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        # 确保输入序列长度不超过最大长度
        assert length <= self.max_length, assert_message
        # 生成位置张量，范围从 0 到 length-1
        position = torch.arange(length, device=device)
        # 生成位置嵌入，形状为 (seq_len, features)
        fixed_embedding = self.embedding(position)
        # 重复批次维度，形状变为 (batch_size, seq_len, features)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    """
    生成一个随机的布尔张量，其中每个元素为 True 的概率为 proba。

    参数:
    - shape (Any): 张量的形状。
    - proba (float): 每个元素为 True 的概率。
    - device (Optional[Any], 可选): 设备，默认为 None。

    返回:
    - Tensor: 随机的布尔张量。
    """
    if proba == 1:
        # 如果概率为1，则返回全1张量
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        # 如果概率为0，则返回全0张量
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        # 否则，使用伯努利分布生成随机的布尔张量
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class UNetCFG1d(UNet1d):

    """UNet1d with Classifier-Free Guidance"""
    """
    UNetCFG1d 类继承自 UNet1d，添加了分类器引导（Classifier-Free Guidance）功能。

    初始化参数:
    - context_embedding_max_length (int): 上下文嵌入的最大长度。
    - context_embedding_features (int): 上下文嵌入特征的维度。
    - use_xattn_time (bool, 可选): 是否使用时间交叉注意力，默认为 False。
    - **kwargs: 其他传递给 UNet1d 的关键字参数。
    """

    def __init__(
        self,
        context_embedding_max_length: int,
        context_embedding_features: int,
        use_xattn_time: bool = False,
        **kwargs,
    ):
        super().__init__(
            # 传递上下文嵌入特征维度给父类、传递其他关键字参数给父类
            context_embedding_features=context_embedding_features, **kwargs
        )
        # 存储是否使用时间交叉注意力
        self.use_xattn_time = use_xattn_time

        if use_xattn_time:
            # 确保上下文嵌入特征维度存在
            assert exists(context_embedding_features)
            # 定义时间位置编码层，将时间步转换为上下文嵌入特征
            self.to_time_embedding = nn.Sequential(
                TimePositionalEmbedding(
                    dim=kwargs["channels"], out_features=context_embedding_features
                ),
                nn.GELU(),
            )
            # 为时间嵌入增加一个位置
            context_embedding_max_length += 1   # Add one for time embedding
        # 定义固定位置嵌入层
        self.fixed_embedding = FixedEmbedding(
            max_length=context_embedding_max_length, features=context_embedding_features
        )

    def forward(  # type: ignore
        self,
        x: Tensor,
        time: Tensor,
        *,
        embedding: Tensor,
        embedding_mask: Optional[Tensor] = None,
        embedding_scale: float = 1.0,
        embedding_mask_proba: float = 0.0,
        batch_cfg: bool = False,
        rescale_cfg: bool = False,
        scale_phi: float = 0.4,
        negative_embedding: Optional[Tensor] = None,
        negative_embedding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        b, device = embedding.shape[0], embedding.device

        if self.use_xattn_time:
            embedding = torch.cat([embedding, self.to_time_embedding(time).unsqueeze(1)], dim=1)

            if embedding_mask is not None:
                embedding_mask = torch.cat([embedding_mask, torch.ones((b, 1), device=device)], dim=1)

        fixed_embedding = self.fixed_embedding(embedding)

        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            if batch_cfg:
                batch_x = torch.cat([x, x], dim=0)
                batch_time = torch.cat([time, time], dim=0)

                if negative_embedding is not None:
                    if negative_embedding_mask is not None:
                        negative_embedding_mask = negative_embedding_mask.to(torch.bool).unsqueeze(2)

                        negative_embedding = torch.where(negative_embedding_mask, negative_embedding, fixed_embedding)
                    
                    batch_embed = torch.cat([embedding, negative_embedding], dim=0)

                else:
                    batch_embed = torch.cat([embedding, fixed_embedding], dim=0)

                batch_mask = None
                if embedding_mask is not None:
                    batch_mask = torch.cat([embedding_mask, embedding_mask], dim=0)

                batch_features = None
                features = kwargs.pop("features", None)
                if self.use_context_features:
                    batch_features = torch.cat([features, features], dim=0)

                batch_channels = None
                channels_list = kwargs.pop("channels_list", None)
                if self.use_context_channels:
                    batch_channels = []
                    for channels in channels_list:
                        batch_channels += [torch.cat([channels, channels], dim=0)]

                # Compute both normal and fixed embedding outputs
                batch_out = super().forward(batch_x, batch_time, embedding=batch_embed, embedding_mask=batch_mask, features=batch_features, channels_list=batch_channels, **kwargs)
                out, out_masked = batch_out.chunk(2, dim=0)
           
            else:
                # Compute both normal and fixed embedding outputs
                out = super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)
                out_masked = super().forward(x, time, embedding=fixed_embedding, embedding_mask=embedding_mask, **kwargs)

            out_cfg = out_masked + (out - out_masked) * embedding_scale

            if rescale_cfg:

                out_std = out.std(dim=1, keepdim=True)
                out_cfg_std = out_cfg.std(dim=1, keepdim=True)

                return scale_phi * (out_cfg * (out_std/out_cfg_std)) + (1-scale_phi) * out_cfg

            else:

                return out_cfg
                
        else:
            return super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)


# 定义 UNetNCCA1d 类，继承自 UNet1d，添加噪声通道条件增强（Noise Channel Conditioning Augmentation）功能
class UNetNCCA1d(UNet1d):

    """UNet1d with Noise Channel Conditioning Augmentation"""
    """
    UNetNCCA1d 类继承自 UNet1d，添加了噪声通道条件增强功能。

    初始化参数:
    - context_features (int): 上下文特征的维度。
    - **kwargs: 其他传递给 UNet1d 的关键字参数。
    """

    def __init__(self, context_features: int, **kwargs):
        # 调用父类的初始化方法，传递上下文特征维度和其他参数
        super().__init__(context_features=context_features, **kwargs)
        # 初始化 NumberEmbedder，用于嵌入数值特征
        self.embedder = NumberEmbedder(features=context_features)

    def expand(self, x: Any, shape: Tuple[int, ...]) -> Tensor:
        """
        将输入张量或数值扩展到指定的形状。

        参数:
        - x (Any): 输入张量或数值。
        - shape (Tuple[int, ...]): 目标形状。

        返回:
        - Tensor: 扩展后的张量。
        """
        x = x if torch.is_tensor(x) else torch.tensor(x)
        return x.expand(shape)

    def forward(  # type: ignore
        self,
        x: Tensor,
        time: Tensor,
        *,
        channels_list: Sequence[Tensor],
        channels_augmentation: Union[
            bool, Sequence[bool], Sequence[Sequence[bool]], Tensor
        ] = False,
        channels_scale: Union[
            float, Sequence[float], Sequence[Sequence[float]], Tensor
        ] = 0,
        **kwargs,
    ) -> Tensor:
        # 获取批次大小和上下文通道数
        b, n = x.shape[0], len(channels_list)
        # 扩展通道增强参数和缩放参数到 (batch_size, n) 的形状
        channels_augmentation = self.expand(channels_augmentation, shape=(b, n)).to(x)
        channels_scale = self.expand(channels_scale, shape=(b, n)).to(x)

        # Augmentation (for each channel list item)
        # 对每个上下文通道应用增强
        for i in range(n):
            # 计算缩放因子
            scale = channels_scale[:, i] * channels_augmentation[:, i]
            # 重塑形状为 (batch_size, 1, 1)
            scale = rearrange(scale, "b -> b 1 1")
            # 获取当前上下文通道
            item = channels_list[i]
            # 对当前上下文通道应用随机噪声增强
            channels_list[i] = torch.randn_like(item) * scale + item * (1 - scale)  # type: ignore # noqa

        # Scale embedding (sum reduction if more than one channel list item)
        # 对通道缩放参数进行嵌入，并进行求和归约
        # 将缩放参数嵌入到高维空间
        channels_scale_emb = self.embedder(channels_scale)
        # 对批次维度进行求和归约
        channels_scale_emb = reduce(channels_scale_emb, "b n d -> b d", "sum")
        # 调用父类的前向传播方法，传递上下文通道和嵌入后的缩放参数
        return super().forward(
            x=x,
            time=time,
            channels_list=channels_list,
            features=channels_scale_emb,
            **kwargs,
        )


class UNetAll1d(UNetCFG1d, UNetNCCA1d):
    """
    UNetAll1d 类继承自 UNetCFG1d 和 UNetNCCA1d，组合了分类器引导（CFG）和噪声通道条件增强（NCCA）功能。

    初始化参数:
    - *args: 传递给父类的位置参数。
    - **kwargs: 传递给父类的关键字参数。
    """
    
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，传递所有参数
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):  # type: ignore
        """
        前向传播方法，执行分类器引导和噪声通道条件增强。

        参数:
        - *args: 传递给父类的位置参数。
        - **kwargs: 传递给父类的关键字参数。

        返回:
        - Tensor: 模型的输出张量。
        """
        # 调用 UNetCFG1d 的前向传播方法
        return UNetCFG1d.forward(self, *args, **kwargs)


def XUNet1d(type: str = "base", **kwargs) -> UNet1d:
    """
    根据类型创建不同的 U-Net 1D 模型。

    参数:
    - type (str, 可选): 模型类型，支持 "base", "all", "cfg", "ncca"，默认为 "base"。
    - **kwargs: 其他传递给模型构造函数的参数。

    返回:
    - UNet1d: 创建的 U-Net 1D 模型实例。

    异常:
    - ValueError: 如果类型未知，则抛出 ValueError。
    """
    if type == "base":
        # 返回基础 U-Net 1D 模型
        return UNet1d(**kwargs)
    elif type == "all":
        # 返回包含 CFG 和 NCCA 功能的 U-Net 1D 模型
        return UNetAll1d(**kwargs)
    elif type == "cfg":
        # 返回包含分类器引导功能的 U-Net 1D 模型
        return UNetCFG1d(**kwargs)
    elif type == "ncca":
        # 返回包含噪声通道条件增强功能的 U-Net 1D 模型
        return UNetNCCA1d(**kwargs)
    else:
        raise ValueError(f"Unknown XUNet1d type: {type}")

class NumberEmbedder(nn.Module):
    """
    NumberEmbedder 类用于将数值嵌入到高维空间。

    初始化参数:
    - features (int): 嵌入特征的维度。
    - dim (int, 可选): 嵌入空间的维度，默认为 256。
    """
    def __init__(
        self,
        features: int,
        dim: int = 256,
    ):
        super().__init__()
        # 存储嵌入特征的维度
        self.features = features
        # 定义时间位置嵌入层
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        """
        前向传播方法，将数值嵌入到高维空间。

        参数:
        - x (Union[List[float], Tensor]): 输入数值或张量。

        返回:
        - Tensor: 嵌入后的张量。
        """
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        # 获取输入张量形状
        shape = x.shape
        # 重塑张量形状
        x = rearrange(x, "... -> (...)")
        # 应用时间位置嵌入
        embedding = self.embedding(x)
        # 将嵌入后的张量重塑回原始形状，并添加嵌入特征的维度
        x = embedding.view(*shape, self.features)
        return x  # type: ignore



################################################ Audio Transforms ################################################ 


class STFT(nn.Module):
    """Helper for torch stft and istft"""
    """
    STFT 类是一个辅助模块，用于执行短时傅里叶变换（STFT）和逆短时傅里叶变换（iSTFT）。

    初始化参数:
    - num_fft (int, 可选): FFT 窗口大小，默认为 1023。
    - hop_length (int, 可选): 帧移长度，默认为 num_fft // 4。
    - window_length (Optional[int], 可选): 窗口长度。如果未提供，则默认为 num_fft。
    - length (Optional[int], 可选): 输出信号的长度。如果未提供，则根据 hop_length 计算。
    - use_complex (bool, 可选): 是否使用复数表示，默认为 False。
    """

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        # 存储 FFT 窗口大小
        self.num_fft = num_fft
        # 设置默认帧移长度
        self.hop_length = default(hop_length, floor(num_fft // 4))
        # 设置默认窗口长度
        self.window_length = default(window_length, num_fft)
        # 存储输出信号的长度
        self.length = length
        # 注册 Hann 窗口
        self.register_buffer("window", torch.hann_window(self.window_length))
        # 是否使用复数表示
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        """
        对输入音频信号进行 STFT 变换，返回幅度和相位。

        参数:
        - wave (Tensor): 输入音频信号，形状为 (batch_size, channels, time_steps)。

        返回:
        - Tuple[Tensor, Tensor]: 返回一个元组，包含幅度和相位张量，形状均为 (batch_size, channels, freq_bins, time_steps)。
        """
        # 获取批次大小
        b = wave.shape[0]
        # 重塑张量形状为 (batch_size * channels, time_steps)
        wave = rearrange(wave, "b c t -> (b c) t")

        # 执行 STFT 变换，返回复数张量
        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            # 如果使用复数表示，则返回实部和虚部
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            # 否则，返回幅度和相位
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase
        # 重塑张量形状为 (batch_size, channels, freq_bins, time_steps)
        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) -> Tensor:
        """
        对 STFT 变换后的幅度和相位进行逆 STFT 变换，生成音频信号。

        参数:
        - stft_a (Tensor): 幅度张量，形状为 (batch_size, channels, freq_bins, time_steps)。
        - stft_b (Tensor): 相位张量，形状为 (batch_size, channels, freq_bins, time_steps)。

        返回:
        - Tensor: 逆 STFT 变换后的音频信号，形状为 (batch_size, channels, time_steps)。
    """
        # 获取批次大小和时间步数
        b, l = stft_a.shape[0], stft_a.shape[-1]  # noqa
        # 计算输出信号长度
        length = closest_power_2(l * self.hop_length)
        # 重塑张量形状为 (batch_size * channels, freq_bins, time_steps)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> (b c) f l")

        if self.use_complex:
            # 如果使用复数表示，则获取实部和虚部
            real, imag = stft_a, stft_b
        else:
            # 否则，获取幅度和相位
            magnitude, phase = stft_a, stft_b
            # 将幅度和相位转换为实部和虚部
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        # 将实部和虚部堆叠成复数张量
        stft = torch.stack([real, imag], dim=-1)
        # 执行逆 STFT 变换，生成音频信号
        wave = torch.istft(
            stft,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            length=default(self.length, length),
            normalized=True,
        )
        # 重塑张量形状为 (batch_size, channels, time_steps)
        return rearrange(wave, "(b c) t -> b c t", b=b)

    def encode1d(
        self, wave: Tensor, stacked: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        将一维音频信号编码为 STFT 特征。

        参数:
        - wave (Tensor): 输入音频信号，形状为 (batch_size, channels, time_steps)。
        - stacked (bool, 可选): 是否将实部和虚部堆叠在一起，默认为 True。

        返回:
        - Union[Tensor, Tuple[Tensor, Tensor]]: 如果 stacked 为 True，则返回堆叠后的张量；否则，返回 (stft_a, stft_b) 元组。
        """
        # 进行 STFT 变换
        stft_a, stft_b = self.encode(wave)
        # 重塑张量形状为 (batch_size, channels * freq_bins, time_steps)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> b (c f) l")
        # 如果需要堆叠，则将实部和虚部拼接起来
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) -> Tensor:
        """
        将 STFT 特征解码为一维音频信号。

        参数:
        - stft_pair (Tensor): 包含实部和虚部的张量，形状为 (batch_size, 2 * channels * freq_bins, time_steps)。

        返回:
        - Tensor: 逆 STFT 变换后的音频信号，形状为 (batch_size, channels, time_steps)。
        """
        # 计算频率维度大小
        f = self.num_fft // 2 + 1
        # 拆分实部和虚部
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        # 重塑张量形状为 (batch_size, channels, freq_bins, time_steps)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b (c f) l -> b c f l", f=f)
        # 进行逆 STFT 变换
        return self.decode(stft_a, stft_b)
