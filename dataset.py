import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds

from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # 顶层目录路径，从该目录开始扫描
    ext:list,  # 允许的文件扩展名列表，例如 ['.txt', '.py']
    # max_size = 1 * 1000 * 1000 * 1000  # 可选参数：只扫描小于1 GB的文件
    ):
    """
    快速扫描目录，查找符合条件的子文件夹和文件。
    这是 `glob` 的一个高效替代方案。

    参数:
    - dir (str): 顶层目录路径，从该目录开始扫描。
    - ext (list): 允许的文件扩展名列表，例如 ['.txt', '.py']。

    返回:
    - Tuple[List[str], List[str]]: 返回两个列表，第一个是子文件夹路径列表，第二个是符合条件的文件路径列表。
    """
    subfolders, files = [], []
    # 为每个扩展名添加前导点（.）如果扩展名本身没有点的话
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed

    # 尝试扫描顶层目录，避免“权限被拒绝”错误
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    # 如果是目录，则将其路径添加到子文件夹列表中
                    subfolders.append(f.path)
                elif f.is_file():
                    # 如果是文件，则获取文件扩展名并转换为小写
                    file_ext = os.path.splitext(f.name)[1].lower()
                    # 检查文件是否隐藏（以 . 开头的文件）
                    is_hidden = os.path.basename(f.path).startswith(".")

                    # 如果文件扩展名在允许的扩展名列表中，并且不是隐藏文件，则将其路径添加到文件列表中
                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                # 如果发生任何异常（例如符号链接过多），则跳过该文件
                pass 
    except:
        # 如果扫描顶层目录时发生任何异常（例如权限问题），则跳过该目录
        pass
    
    # 递归扫描子文件夹
    for dir in list(subfolders):
        # 递归调用自身，扫描子文件夹
        sf, f = fast_scandir(dir, ext)
        # 将子文件夹列表扩展到当前子文件夹列表中
        subfolders.extend(sf)
        # 将文件列表扩展到当前文件列表中
        files.extend(f)
    return subfolders, files


def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    """
    快速扫描目录，查找符合条件的子文件夹和文件，并在文件名中搜索指定关键词。
    这是 `glob` 的一个高效替代方案。

    参数:
    - dir (str): 顶层目录路径，从该目录开始扫描。
    - ext (list): 允许的文件扩展名列表，例如 ['.wav', '.mp3']。
    - keywords (list): 需要在文件名中搜索的关键词列表。

    返回:
    - Tuple[List[str], List[str]]: 返回两个列表，第一个是子文件夹路径列表，第二个是符合条件的文件路径列表。
    """
    subfolders, files = [], []
    # make keywords case insensitive
    # 将关键词转换为小写以实现不区分大小写的搜索
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    # 为每个扩展名添加前导点（.）如果扩展名本身没有点的话
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    # 定义禁止包含的关键词列表
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        # 尝试扫描顶层目录，避免“权限被拒绝”错误
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    # 如果是目录，则将其路径添加到子文件夹列表中
                    subfolders.append(f.path)
                elif f.is_file():
                    # 检查文件是否隐藏（以 . 开头的文件）
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    # 检查文件扩展名是否在允许的扩展名列表中
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    # 将文件名转换为小写
                    name_lower = f.name.lower()
                    # 检查文件名中是否包含任意一个关键词
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    # 检查文件名中是否包含任何禁止的关键词
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    # 如果文件扩展名匹配，包含关键词，不包含禁止的关键词，且不是隐藏文件，则将其路径添加到文件列表中
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                # 如果发生任何异常（例如符号链接过多），则跳过该文件
                pass
    except:
        # 如果扫描顶层目录时发生任何异常（例如权限问题），则跳过该目录
        pass
    
    # 递归扫描子文件夹
    for dir in list(subfolders):
        # 递归调用自身，扫描子文件夹
        sf, f = keyword_scandir(dir, ext, keywords)
        # 将子文件夹列表扩展到当前子文件夹列表中
        subfolders.extend(sf)
        # 将文件列表扩展到当前文件列表中
        files.extend(f)
    return subfolders, files


def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    """
    递归获取指定目录及其子目录中符合条件的音频文件名列表。

    参数:
    - paths (list): 需要搜索的目录列表。
    - keywords (list, 可选): 需要在文件名中搜索的关键词列表。如果为 None，则不进行关键词搜索。
    - exts (list, 可选): 允许的音频文件扩展名列表，默认为 ['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']。

    返回:
    - List[str]: 符合条件的音频文件路径列表。
    """
    filenames = []
    # 如果传入的 paths 是字符串，则转换为列表
    if type(paths) is str:
        paths = [paths]
        
    # 遍历每个目录路径
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
            # 如果提供了关键词，则使用 keyword_scandir 进行扫描和搜索
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            # 否则，使用 fast_scandir 进行扫描，不进行关键词搜索
            subfolders, files = fast_scandir(path, exts)
        # 将符合条件的文件路径添加到文件名列表中
        filenames.extend(files)
    return filenames


# 定义 LocalDatasetConfig 类，用于配置本地数据集
class LocalDatasetConfig:
    """
    LocalDatasetConfig 类用于配置本地数据集。

    初始化参数:
    - id (str): 数据集的标识符。
    - path (str): 数据集的路径。
    - custom_metadata_fn (Optional[Callable[[str], str]], 可选): 自定义元数据函数，用于生成自定义元数据。
    """
    def __init__(
        self,
        id: str, # 数据集的标识符
        path: str, # 数据集的路径
        custom_metadata_fn: Optional[Callable[[str], str]] = None # 可选的自定义元数据函数
    ):
        self.id = id # 存储数据集标识符
        self.path = path # 存储数据集路径 
        self.custom_metadata_fn = custom_metadata_fn # 存储自定义元数据函数


# 定义 SampleDataset 类，继承自 torch.utils.data.Dataset，用于加载音频样本
class SampleDataset(torch.utils.data.Dataset):
    """
    SampleDataset 类用于加载音频样本，并进行预处理和增强。

    初始化参数:
    - configs (List[LocalDatasetConfig]): 数据集配置列表。
    - sample_size (int, 可选): 每个样本的样本大小（样本点数），默认为 65536。
    - sample_rate (int, 可选): 采样率，默认为 48000。
    - keywords (List[str], 可选): 需要在文件名中搜索的关键词列表。
    - random_crop (bool, 可选): 是否随机裁剪样本，默认为 True。
    - force_channels (str, 可选): 强制通道数，"stereo" 表示立体声，"mono" 表示单声道，默认为 "stereo"。
    """
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        # 定义数据增强模块，使用 PhaseFlipper 进行相位翻转
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        # 定义 PadCrop_Normalized_T 模块，用于填充和裁剪样本
        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        # 定义编码模块，根据 force_channels 选择 Stereo 或 Mono
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        # 初始化自定义元数据函数字典
        self.custom_metadata_fns = {}

        # 遍历数据集配置列表
        for config in configs:
            # 将根目录路径添加到根目录路径列表中
            self.root_paths.append(config.path)
            # 获取音频文件路径列表，并扩展到文件名列表中
            self.filenames.extend(get_audio_filenames(config.path, keywords))
            if config.custom_metadata_fn is not None:
                # 如果提供了自定义元数据函数，则将其存储到字典中，键为根目录路径
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn
        # 输出找到的文件数量
        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        """
        加载音频文件。

        参数:
        - filename (str): 音频文件路径。

        返回:
        - Tensor: 加载的音频数据。
        """
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                # 读取音频数据
                audio = f.read(f.frames)
                # 转换为 PyTorch 张量
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            # 如果采样率不匹配，则进行重采样
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        """
        返回数据集的大小。

        返回:
        - int: 数据集的大小。
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。

        参数:
        - idx (int): 样本索引。

        返回:
        - Tuple[Tensor, dict]: 返回一个元组，包含音频数据和元数据。
        """
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            # 对音频数据进行填充和裁剪
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Run augmentations on this sample (including random crop)
            # 对音频数据进行编码
            if self.augs is not None:
                audio = self.augs(audio)

            # 裁剪音频数据到 [-1, 1] 范围
            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            # 对音频数据进行编码
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            # 查找文件所在的根目录，并存储相对路径
            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time

            # 应用自定义元数据函数
            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    # 如果元数据中包含 "__reject__" 并且为 True，则随机替换样本
                    return self[random.randrange(len(self))]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]


def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    """
    返回一个生成器函数，该函数迭代输入数据并将键值对分组到样本中。

    参数:
    - data: 输入数据，迭代器形式，包含字典，每个字典有 "fname" 和 "data" 键。
    - keys: 函数，用于将文件名分割为键和扩展名，默认为 base_plus_ext。
    - lcase (bool, 可选): 是否将扩展名转换为小写，默认为 True。
    - suffixes (List[str], 可选): 可选的扩展名列表，用于过滤文件。
    - handler (Callable, 可选): 可选的处理器，用于处理样本。

    Yields:
    - Dict[str, Any]: 分组后的样本，字典形式，包含键和值对。
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        # 如果启用了跟踪，则打印当前键、扩展名和当前样本的键
        if wds.tariterators.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            # 如果当前样本为 None 或者键发生变化，则开始一个新的样本
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample

wds.tariterators.group_by_keys = group_by_keys

# S3 code and WDS preprocessing code based on implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
# S3 代码和 WDS 预处理代码基于 Scott Hawley 的原始实现

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    """
    返回指定 S3 存储桶和目录路径中所有文件的完整 S3 路径列表。

    参数:
    - dataset_path (str): S3 存储桶中的目录路径。
    - s3_url_prefix (Optional[str], 可选): 可选的 S3 URL 前缀。
    - filter (str, 可选): 可选的过滤字符串，用于过滤文件路径。
    - recursive (bool, 可选): 是否递归获取子目录中的文件，默认为 True。
    - debug (bool, 可选): 是否打印调试信息，默认为 False。
    - profile (Optional[str], 可选): 可选的 AWS 配置文件名称。

    返回:
    - List[str]: 包含所有文件完整 S3 路径的列表。
    """
    # 确保 dataset_path 以斜杠结尾
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    # 使用 posixpath 构建 S3 URL 路径
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    # 构建 `aws s3 ls` 命令
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        # 如果提供了配置文件名称，则添加 --profile 参数
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        # 如果需要递归，则添加 --recursive 参数
        cmd.append('--recursive')
    
    # Run the `aws s3 ls` command and capture the output
    # 运行 `aws s3 ls` 命令并捕获输出
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    # 将输出拆分为行，并去除每行的空白字符
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    # 去除以时间戳开头的行中的时间戳
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    # 构建每个文件的完整 S3 路径
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    # 应用过滤字符串（如果指定）
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    # 如果递归获取，则去除冗余的目录名称
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents


def get_all_s3_urls(
    # 有效的 [LAION AudioDataset] 数据集名称列表
    names=[], 
    # list of subsets you want from those datasets, e.g. ['train','valid']
    # 需要从这些数据集中获取的子集列表，例如 ['train','valid']
    subsets=[''],
    # 数据集名称的前缀，如果未指定，则 names 中的每个元素都包含完整的 S3 路径
    s3_url_prefix=None,  # prefix for those dataset names
    # 是否递归列出所有子目录中的 tar 文件
    recursive=True,     # recursively list all tar files in all subdirs
    # 仅获取包含此子字符串的文件
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    # 每个名称对应的配置文件字典，例如 {'dataset1': 'profile1', 'dataset2': 'profile2'}
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    """
    获取多个数据集在单个 S3 存储桶中的碎片（tar 文件）URL。

    参数:
    - names (List[str]): 有效的 [LAION AudioDataset] 数据集名称列表。
    - subsets (List[str]): 需要从这些数据集中获取的子集列表，例如 ['train','valid']。
    - s3_url_prefix (Optional[str]): 数据集名称的前缀。如果未指定，则 names 中的每个元素都包含完整的 S3 路径。
    - recursive (bool, 可选): 是否递归列出所有子目录中的 tar 文件，默认为 True。
    - filter_str (str, 可选): 仅获取包含此子字符串的文件，默认为 'tar'。
    - debug (bool, 可选): 是否打印调试信息，默认为 False。
    - profiles (Dict[str, str], 可选): 每个名称对应的配置文件字典，例如 {'dataset1': 'profile1', 'dataset2': 'profile2'}。

    返回:
    - List[str]: 包含所有碎片（tar 文件）URL 的列表。
    """
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        # 如果未指定 s3_url_prefix，则假设 names 中的每个元素都包含完整的 S3 路径
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            # 使用 posixpath 构建 S3 路径，使用 s3_url_prefix 和当前名称值
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            # 获取当前子集目录中的 tar 文件列表
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                # 对 tar 文件名中的空格和括号进行转义，以便在 shell 命令中使用
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                # 构建当前 tar 文件的 S3 路径
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                # 构建 AWS CLI 命令以下载当前 tar 文件
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    """
    在异常处理程序中调用，用于忽略任何异常，发出警告并继续。

    参数:
    - exn: 异常对象。

    返回:
    - True: 始终返回 True。
    """
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    """
    检查样本是否有效。

    参数:
    - sample (Dict[str, Any]): 样本，字典形式。

    返回:
    - bool: 如果样本包含 "json" 和 "audio" 键，并且不是静音且未被拒绝，则返回 True。
    """
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_silent = is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected


# 定义 S3DatasetConfig 类，用于配置 S3 数据集
class S3DatasetConfig:
    """
    S3DatasetConfig 类用于配置 S3 数据集。

    初始化参数:
    - id (str): 数据集的标识符。
    - s3_path (str): S3 存储桶中的路径。
    - custom_metadata_fn (Optional[Callable[[str], str]], 可选): 自定义元数据函数，用于生成自定义元数据。
    - profile (Optional[str], 可选): AWS 配置文件名称。
    """
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        """
        加载 S3 数据集的 URL 列表。

        使用 get_all_s3_urls 函数获取 S3 存储桶中的所有 tar 文件 URL。

        返回:
        - List[str]: S3 数据集的 URL 列表。
        """
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )

        return self.urls


# 定义 LocalWebDatasetConfig 类，用于配置本地 WebDataset
class LocalWebDatasetConfig:
    """
    LocalWebDatasetConfig 类用于配置本地 WebDataset。

    初始化参数:
    - id (str): 数据集的标识符。
    - path (str): 本地目录路径。
    - custom_metadata_fn (Optional[Callable[[str], str]], 可选): 自定义元数据函数，用于生成自定义元数据。
    - profile (Optional[str], 可选): AWS 配置文件名称。
    """
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):
        """
        加载本地 WebDataset 的 URL 列表。

        使用 fast_scandir 函数扫描本地目录，获取所有 tar 文件路径。

        返回:
        - List[str]: 本地 WebDataset 的 URL 列表。
        """
        # 使用 fast_scandir 扫描目录，获取 tar 文件路径
        self.urls = fast_scandir(self.path, ["tar"])[1]

        return self.urls


# 定义音频解码函数，用于解码音频文件
def audio_decoder(key, value):
    # Get file extension from key
    """
    解码音频文件。

    参数:
    - key (str): 文件名。
    - value (bytes): 文件内容。

    返回:
    - Any: 解码后的音频数据，如果文件扩展名不是音频格式，则返回 None。
    """
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None


# 定义批处理函数，用于将样本批量化
def collation_fn(samples):
    """
    批处理函数，将样本批量化。

    参数:
    - samples (List[Any]): 样本列表。

    返回:
    - List[Any]: 批处理后的样本列表。
    """  
    batched = list(zip(*samples))
    result = []
    # 遍历每个批处理后的样本
    for b in batched:
        # 如果样本是整数或浮点数
        if isinstance(b[0], (int, float)):
            # 转换为 NumPy 数组
            b = np.array(b)
        # 如果样本是 PyTorch 张量
        elif isinstance(b[0], torch.Tensor):
            # 堆叠成一个新的张量
            b = torch.stack(b)
        # 如果样本是 NumPy 数组
        elif isinstance(b[0], np.ndarray):
            # 转换为 NumPy 数组
            b = np.array(b)
        else:
            b = b
        result.append(b)
    return result



class WebDatasetDataLoader():
    """
    WebDatasetDataLoader 类用于从 S3 或本地加载 WebDataset 数据集，并进行数据预处理和批处理。

    初始化参数:
    - datasets (List[S3DatasetConfig]): 数据集配置列表，每个配置包含 S3 路径、自定义元数据函数等。
    - batch_size (int): 每个批次包含的样本数量。
    - sample_size (int): 每个样本的样本大小（样本点数）。
    - sample_rate (int, 可选): 采样率，默认为 48000。
    - num_workers (int, 可选): 数据加载器使用的子进程数量，默认为 8。
    - epoch_steps (int, 可选): 每个 epoch 的步数，默认为 1000。
    - random_crop (bool, 可选): 是否随机裁剪样本，默认为 True。
    - force_channels (str, 可选): 强制通道数，"stereo" 表示立体声，"mono" 表示单声道，默认为 "stereo"。
    - augment_phase (bool, 可选): 是否随机反转相位进行数据增强，默认为 True。
    - **data_loader_kwargs: 其他传递给 WebLoader 的关键字参数。
    """
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase

        # 加载所有数据集的 URL 列表
        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        # 将列表中的列表展平为一个单一的 URL 列表
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        # 随机打乱 URL 列表
        random.shuffle(urls)

        # 构建 WebDataset 数据管道
        self.dataset = wds.DataPipeline(
            # 使用 ResampledShards 随机重采样 URL 列表
            wds.ResampledShards(urls),
            # 将 tar 文件转换为样本，处理异常
            wds.tarfile_to_samples(handler=log_and_continue),
            # 解码音频文件，处理异常
            wds.decode(audio_decoder, handler=log_and_continue),
            # 应用预处理函数，处理异常
            wds.map(self.wds_preprocess, handler=log_and_continue),
            # 选择有效的样本
            wds.select(is_valid_sample),
            # 将样本转换为元组，包含音频和元数据，处理异常
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            # 批处理样本，使用自定义的 collation_fn 函数
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        ).with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        # 创建 WebLoader 实例，用于加载数据
        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):
        """
        对 WebDataset 中的样本进行预处理。

        参数:
        - sample (Dict[str, Any]): 输入样本，字典形式，包含音频数据和元数据。

        返回:
        - Optional[Dict[str, Any]]: 预处理后的样本。如果样本无效，则返回 None。
        """
        # 初始化找到的键和重写键
        found_key, rewrite_key = '', '' 
        for k, v in sample.items():  # print the all entries in dict
            for akey in AUDIO_KEYS:
                if k.endswith(akey):
                    # to rename long/weird key with its simpler counterpart
                    found_key, rewrite_key = k, akey
                    break
            if '' != found_key:
                break
        if '' == found_key:  # got no audio!
            return None  # try returning None to tell WebDataset to skip this one

        audio, in_sr = sample[found_key]
        if in_sr != self.sample_rate:
            resample_tf = T.Resample(in_sr, self.sample_rate)
            audio = resample_tf(audio)

        if self.sample_size is not None:
            # 如果指定了样本大小，则进行填充或裁剪，并获取相对时间戳
            # Pad/crop and get the relative timestamp
            pad_crop = PadCrop_Normalized_T(
                self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                audio)
            sample["json"]["seconds_start"] = seconds_start
            sample["json"]["seconds_total"] = seconds_total
            sample["json"]["padding_mask"] = padding_mask
        else:
            t_start, t_end = 0, 1

        # Check if audio is length zero, initialize to a single zero if so
        # 检查音频长度是否为0，如果是，则初始化为单个零
        if audio.shape[-1] == 0:
            audio = torch.zeros(1, 1)

        # Make the audio stereo and augment by randomly inverting phase
        # 将音频转换为立体声，并随机反转相位进行增强
        augs = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
            PhaseFlipper() if self.augment_phase else torch.nn.Identity()
        )

        audio = augs(audio)

        sample["json"]["timestamps"] = (t_start, t_end)

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        # 应用自定义元数据函数
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
        
            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
            del sample[found_key]

        sample["audio"] = audio

        # Add audio to the metadata as well for conditioning
        # 将音频数据也添加到元数据中，以便进行条件生成
        sample["json"]["audio"] = audio
        
        return sample


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):
    """
    根据数据集配置创建数据加载器。

    参数:
    - dataset_config (dict): 数据集配置字典，包含数据集类型、路径、自定义元数据函数等信息。
    - batch_size (int): 每个批次包含的样本数量。
    - sample_size (int): 每个样本的样本大小（样本点数）。
    - sample_rate (int): 采样率。
    - audio_channels (int, 可选): 音频通道数，默认为2（立体声）。
    - num_workers (int, 可选): 数据加载器使用的子进程数量，默认为4。

    返回:
    - torch.utils.data.DataLoader: 创建好的数据加载器。
    """
    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        # 初始化配置列表
        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            # 创建 LocalDatasetConfig 实例并添加到配置列表中
            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        # 创建 SampleDataset 实例
        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels
        )
        
        # 创建 DataLoader 实例
        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

    # 如果数据集类型是 S3 或 WebDataset（支持 "s3" 以向后兼容）
    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        # 初始化 WebDataset 配置列表
        wds_configs = []

        # 遍历每个 WebDataset 配置
        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                # 如果提供了自定义元数据模块路径，则动态导入模块
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                # 获取自定义元数据函数
                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:
                # 如果配置中有 S3 路径，则创建 S3DatasetConfig 实例
                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    # 如果配置中有本地路径，则创建 LocalWebDatasetConfig 实例
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )
        # 创建 WebDatasetDataLoader 实例
        return WebDatasetDataLoader(
            wds_configs, # WebDataset 配置列表
            sample_rate=sample_rate, # 采样率
            sample_size=sample_size, # 样本大小
            batch_size=batch_size, # 批次大小
            random_crop=dataset_config.get("random_crop", True), # 是否随机裁剪
            num_workers=num_workers, # 子进程数量
            persistent_workers=True, # 保持子进程活跃
            force_channels=force_channels, # 强制通道数
            epoch_steps=dataset_config.get("epoch_steps", 2000)  # 每个 epoch 的步数
        ).data_loader # 返回 DataLoader 实例
    
