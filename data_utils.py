import math
import random
import torch

from torch import nn
from typing import Tuple


class PadCrop(nn.Module):
    """
    PadCrop 类用于对音频信号进行填充或裁剪，使其达到指定的样本数。
    如果音频信号长度不足，则进行填充；如果超过，则进行随机裁剪。

    初始化参数:
    - n_samples (int): 目标样本数。
    - randomize (bool, 可选): 是否随机裁剪，默认为 True。
    """
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        """
        对输入音频信号进行填充或裁剪。

        参数:
        - signal (torch.Tensor): 输入音频信号，形状为 (n_channels, n_samples)。

        返回:
        - torch.Tensor: 填充或裁剪后的音频信号，形状为 (n_channels, n_samples)。
        """
        # 获取音频信号的通道数和样本数
        n, s = signal.shape
        # 如果不随机裁剪，则从起始位置开始
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        # 计算结束位置
        end = start + self.n_samples
        # 创建一个全零张量，形状为 (n_channels, n_samples)
        output = signal.new_zeros([n, self.n_samples])
        # 将音频信号复制到输出张量中
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


class PadCrop_Normalized_T(nn.Module):
    """
    PadCrop_Normalized_T 类用于对音频信号进行填充或裁剪，并返回时间戳和填充掩码。
    音频信号被填充或裁剪到指定的样本数，并计算相对时间戳和填充掩码。

    初始化参数:
    - n_samples (int): 目标样本数。
    - sample_rate (int): 采样率，用于计算时间戳。
    - randomize (bool, 可选): 是否随机裁剪，默认为 True。
    """
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        """
        对输入音频信号进行填充或裁剪，并返回时间戳和填充掩码。

        参数:
        - source (torch.Tensor): 输入音频信号，形状为 (n_channels, n_samples)。

        返回:
        - Tuple[torch.Tensor, float, float, int, int, torch.Tensor]: 返回一个元组，包含：
            - 填充或裁剪后的音频信号。
            - 相对开始时间（0到1之间）。
            - 相对结束时间（0到1之间）。
            - 开始时间（秒）。
            - 总时长（秒）。
            - 填充掩码（与音频信号长度相同，1表示有音频，0表示填充）。
        """
        # 获取音频信号的通道数和样本数
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        # 计算填充或裁剪的上界
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        # 如果需要随机裁剪且音频长度超过目标长度，则进行随机裁剪
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            # 随机生成偏移量
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        # 计算相对开始时间和结束时间（0到1之间）
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        # 创建一个全零张量，形状为 (n_channels, n_samples)
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        # 将音频信号复制到输出张量中
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        # 计算开始时间和总时长（秒）
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        # 创建填充掩码，1表示有音频，0表示填充
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        return (
            chunk, # 填充或裁剪后的音频信号
            t_start, # 相对开始时间
            t_end, # 相对结束时间
            seconds_start, # 开始时间（秒）
            seconds_total, # 总时长（秒）
            padding_mask # 填充掩码
        )


class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    """
    PhaseFlipper 类用于随机反转音频信号的相位。

    初始化参数:
    - p (float, 可选): 反转相位的概率，默认为 0.5。
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        """
        对输入音频信号进行相位反转。

        参数:
        - signal (torch.Tensor): 输入音频信号。

        返回:
        - torch.Tensor: 反转相位后的音频信号（以一定概率）或原始信号。
        """
        # 以 self.p 的概率反转信号的相位，否则返回原始信号
        return -signal if (random.random() < self.p) else signal


class Mono(nn.Module):
    """
    Mono 类用于将立体声音频转换为单声道音频。

    调用方法:
    - 如果输入信号的维度大于1，则计算通道的平均值，将其转换为单声道。
    - 如果输入信号已经是单声道，则直接返回原始信号。
    """
    def __call__(self, signal):
        """
        将立体声音频转换为单声道音频。

        参数:
        - signal (torch.Tensor): 输入音频信号。

        返回:
        - torch.Tensor: 单声道音频信号。
        """
        # 如果信号维度大于1，则计算通道的平均值，将其转换为单声道
        return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal


class Stereo(nn.Module):
    """
    Stereo 类用于将单声道音频转换为立体声音频。

    调用方法:
    - 如果输入信号是单声道（维度为1），则重复该信号两次，将其转换为立体声。
    - 如果输入信号已经是立体声（维度为2），则直接返回原始信号。
    - 如果输入信号的通道数超过2，则只保留前两个通道，将其转换为立体声。
    """
    def __call__(self, signal):
        """
        将单声道音频转换为立体声音频。

        参数:
        - signal (torch.Tensor): 输入音频信号。

        返回:
        - torch.Tensor: 立体声音频信号。
        """
        signal_shape = signal.shape
        # Check if it's mono
        # 如果信号是单声道（维度为1），则重复该信号两次，将其转换为立体声
        if len(signal_shape) == 1: # s -> 2, s
            signal = signal.unsqueeze(0).repeat(2, 1)
        elif len(signal_shape) == 2:
            if signal_shape[0] == 1: #1, s -> 2, s
                signal = signal.repeat(2, 1)
            elif signal_shape[0] > 2: #?, s -> 2,s
                signal = signal[:2, :]    
        # 返回转换后的立体声音频信号
        return signal
