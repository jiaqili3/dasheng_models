# Reduced from `xiaomitts/common/audiowebdataset.py`

import json
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union  # type: ignore

import numpy as np
import torch
import torchaudio
import webdataset as wds
from loguru import logger

torchaudio.set_audio_backend("soundfile")

def fast_warn_and_continue(exn):
    if "Format not recognised" in repr(exn):
        raise RuntimeError("Format not recognised")

    warnings.warn(repr(exn))
    return True


def crop_or_pad_audio(wav: torch.Tensor, crop_size: int, pad_last: bool = False):
    n_samples, *_ = wav.shape
    available_crops = n_samples // crop_size
    for i in range(available_crops):
        crop = wav[i * crop_size : (i + 1) * crop_size, ...]
        yield crop

    if (available_crops == 0) and pad_last:
        last_crop = wav[available_crops * crop_size :, ...]
        padded = torch.zeros((crop_size, *last_crop.shape[1:]))
        padded[: last_crop.shape[0]] = last_crop
        yield padded


def _seq_crop_audio(
    data,
    crop_length: None | float,
    mono: bool = True,
    drop_clipped: bool = True,
    pad_last: bool = False,
    handler=None,
):
    """WebDataset crop filter, yields sequential crops
        crop_length: in number of frames
    """
    for sample in data:
        audio, *extra = sample
        audio, sr = audio
        if mono and audio.ndim == 2:
            audio = audio.mean(0)
        if audio.abs().max() >= 0.99 and drop_clipped:
            continue
        if crop_length is not None:
            crops = crop_or_pad_audio(audio.float(), crop_size=(crop_length), pad_last=pad_last)
        else:
            crops = [audio.float()]

        for crop in crops:
            yield (crop, *extra)


class Audiowebdataset(wds.DataPipeline):

    def __init__(
        self,
        urls,
        tar_shuffle: None | int = None,
        resample: bool = False,
        target_sample_rate: None | int = None,
        batch_size: None | int = None,
        filter_function: None | Callable = None,
        rename_keys: Dict[str, str] = dict(audio="flac;mp3;sox;wav;m4a;ogg;wma;", filename="__key__"),
        map_kwargs: None | Dict[str, Callable] = None,
        merge_function: (
            None | Callable
        ) = None,  # merge function is called before batching. In the merge function we can operate on the data in form of a tuple
        handler=fast_warn_and_continue,
    ):
        pipeline: List = [wds.ResampledShards(urls) if resample else wds.SimpleShardList(urls)]

        if tar_shuffle is not None:
            # Tar wise shuffle
            pipeline.extend(
                [
                    wds.detshuffle(
                        bufsize=tar_shuffle,
                        initial=tar_shuffle // 4,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker at each node
                    wds.tarfile_to_samples(handler=handler),
                    wds.shuffle(
                        bufsize=tar_shuffle,
                        initial=tar_shuffle // 4,
                    ),
                ]
            )
        else:
            pipeline.extend([wds.split_by_node, wds.split_by_worker, wds.tarfile_to_samples(handler=handler)])

        # Decode i.e., bytes object to a python-accessible obj.
        pipeline.extend([wds.decode(wds.torch_audio, handler=handler), wds.rename(**rename_keys, handler=handler)])

        if map_kwargs:
            pipeline.extend([wds.map_dict(**map_kwargs)])
        # Filter function takes a sample (key: value) as input and returns True for valid samples, otherwise false
        if filter_function:
            pipeline.extend([wds.select(filter_function)])

        # Resample audio, useful when dataset is not monotonous in sampling rate
        if target_sample_rate:
            assert "audio" in rename_keys.keys(), "target_sample_rate requires key_maps=dict(audio='flac;mp3;wav')"

            def resample_audio(audio_sr: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
                audio, sr = audio_sr
                audio = torchaudio.functional.resample(audio, sr, target_sample_rate)
                return (audio, target_sample_rate)

            pipeline.extend([wds.map_dict(audio=resample_audio)])

        # Webdataset support batching and parallel reading using
        # num_workers only with tuples, not dicts
        pipeline.extend(
            [
                wds.to_tuple(*rename_keys.keys()),
            ]
        )

        if merge_function is not None:
            pipeline.extend([merge_function])

        if batch_size is not None:
            pipeline.append(
                wds.batched(
                    batch_size,
                    collation_fn=partial(
                        wds.filters.default_collation_fn, combine_tensors=False, combine_scalars=False
                    ),
                )
            )

        super().__init__(pipeline)


# Can also replace with wds.Randomix
class BalancedDatasetSampler(wds.DataPipeline, wds.compat.FluidInterface):

    def __init__(self, **datasets):

        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = {k: iter(ds) for k, ds in self.datasets.items()}
        while True:
            for k, source in sources.items():
                try:
                    yield next(source)
                except StopIteration:
                    break

class MultiDatasetLoader:
    def __init__(self, dataloaders, batch_size, weights=None):
        self.dataloaders = dataloaders
        self.iterators = None
        self.batch_size = batch_size
        self.weights = weights if weights is not None else [1.0] * len(dataloaders)
        self.weights = np.array(self.weights) / sum(self.weights)
        assert len(self.weights) == len(dataloaders), "Number of weights must match number of dataloaders"
        
    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        while True:
            try:
                # For each item in the batch, randomly select a dataset
                batch_samples = []
                batch_sources = []  # Track source dataset indices
                for _ in range(self.batch_size):
                    loader_idx = np.random.choice(len(self.dataloaders), p=self.weights)
                    try:
                        sample = next(self.iterators[loader_idx])
                        # WebDataset returns batched data, we need to unbatch it
                        if isinstance(sample, (list, tuple)) and len(sample) > 0:
                            sample = sample[0]
                            target_sample_rate = 16000
                        elif isinstance(sample, dict):
                            target_sample_rate = sample['sample_rate']
                            sample = sample['speech']
                        else:
                            raise NotImplementedError
                        batch_samples.append(sample)
                        batch_sources.append(loader_idx)
                    except StopIteration:
                        if loader_idx == 1:
                            # 1 epoch of general audio
                            raise StopIteration
                        logger.info(f"re-running iteration: {loader_idx}")
                        self.iterators[loader_idx] = iter(self.dataloaders[loader_idx])
                        sample = next(self.iterators[loader_idx])
                        if isinstance(sample, (list, tuple)) and len(sample) > 0:
                            sample = sample[0]
                        batch_samples.append(sample)
                        batch_sources.append(loader_idx)
                    except Exception as e:
                        logger.error(e)
                        continue
                if batch_samples == []:
                    continue
                # Collate the mixed batch
                try:
                    collated = collate_with_lengths_wds(batch_samples, flatten=False, target_sample_rate=target_sample_rate)
                    # Add source indices to the output
                    collated['batch_sources'] = torch.tensor(batch_sources)
                    assert 'speech' in collated.keys()
                    yield collated
                except Exception as e:
                    logger.error(f"Error in batch creation: {e}")
                    logger.error(f"Error in batch creation: {batch_samples}")
                    logger.error(f"Error in batch creation: {batch_sources}")
                    continue
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Error in batch creation: {e}")
                raise e

def expand_with_brace(lists: Iterable[str] | str):
    import braceexpand

    r = []
    for l in lists:
        if "*" in l:
            # Expand using "posix" based *
            r.extend([str(f) for f in Path(l).parent.glob(Path(l).name)])
        else:
            r.extend(braceexpand.braceexpand(l))
        if any(not Path(l).exists() for l in r):
            raise FileNotFoundError(f"One of the paths does not exist: {r}")
    return r


def pad(tensorlist: Sequence[torch.Tensor], padding_value: float = 0.0):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim,) + trailing_dims + (num_raw_samples,)
    out_tensor = torch.full(out_dims, fill_value=padding_value, dtype=tensorlist[0].dtype)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor, torch.as_tensor(lengths)


def collate_with_lengths_wds(
    samples: List[Iterable], combine_scalars: bool = True, flatten: bool = True, combine_tensors: bool = True, 
    target_sample_rate=16000,
):
    batched = list(zip(*samples))
    # result = []
    result = {}
    result['duration'] = 0
    result['sample_rate'] = target_sample_rate

    # result['speech'] = pad(list(batched[0]))
    for i, bat in enumerate(batched):
        if isinstance(bat[0], (int, float)):
            raise NotImplementedError
            if combine_scalars:
                bat = np.array(list(bat))
        elif isinstance(bat[0], torch.Tensor):
            if combine_tensors:
                bat = pad(list(bat))
                # Calculate duration for audio tensors (first element in batch)
                if i == 0:  # Assuming audio is always the first element
                    # b[1] contains the lengths tensor
                    duration = float(bat[1].sum()) / 16000  # Assuming 16kHz sample rate
                    result['duration'] += duration
            result['speech'] = bat[0]
            result['speech_lens'] = bat[1]
        elif isinstance(bat[0], np.ndarray):
            if combine_tensors:
                bat = np.array(list(bat))
        else:
            # nothing here
            bat = list(bat)
        # Do not flatten lists, i.e., some filenames
        # if flatten and not isinstance(b, list):
        #     result.extend(b)
        # else:
        #     result.append(b)
    return result


# Returns (single) dicts with (audio=audio_data, *extra ), useful for only reading audio and keeping other items the same
def create_rawaudio_webdataset(
    urls: List[str] | Dict[str, List[str]],
    target_sample_rate: Optional[int] = 16000,
    mono: bool = True,
    num_workers: int = 4,
    batch_size: int = 64,
    crop_length: float | None = None,
    pad_last: bool = False,  # If only 1 crop available, use padding
    **kwargs,
):

    dataset_kwargs = dict(
        batch_size=batch_size,
        rename_keys=(dict(audio="flac;mp3;sox;wav;m4a;ogg;wma;audio.mp3;clip.mp3", filename="__key__")),
        target_sample_rate=target_sample_rate,
        merge_function=partial(
            _seq_crop_audio, crop_length=crop_length, mono=mono, drop_clipped=False, pad_last=pad_last
        ),
    )
    urls = expand_with_brace(urls)
    dataset = Audiowebdataset(urls, **dataset_kwargs)
    # Set num_workers at most to number of tars, otherwise some processes will do nothing, slowing down dataloading
    dataloader = wds.WebLoader(dataset, num_workers=min(len(urls), num_workers), batch_size=None).unbatched()
    dataloader = dataloader.batched(
        batch_size,
        collation_fn=partial(collate_with_lengths_wds, flatten=False, target_sample_rate=target_sample_rate),
    )
    return dataloader

def create_embedding_webdataset(
    urls: Union[List[str], Dict[str, List[str]]],
    tar_shuffle: None | int = None,
    batch_size: int = 16,
    balanced_sampler: None | bool = False,
    num_workers: int = 4,
    training: bool = False,
    label_processor: None | Callable = None,
    merge_processor: None | Callable = None,
    **kwargs,
):
    dataset_kwargs = dict(
        tar_shuffle=tar_shuffle,
        batch_size=batch_size,
        rename_keys=dict(embedding="npy", target="json", filename="__key__"),
        map_kwargs=dict(
            embedding=lambda x: x.transpose(),
            target=label_processor if label_processor else lambda x: x,
        ),  # Transpose (B,T,D) -> (B,D,T), map the labels if provided
        merge_function=merge_processor,
    )
    if balanced_sampler:
        assert isinstance(urls, dict)
        ds = {k: Audiowebdataset(expand_with_brace(train_data), **dataset_kwargs) for k, train_data in urls.items()}
        dataset = BalancedDatasetSampler(**ds)
    else:
        assert isinstance(urls, list)
        dataset = Audiowebdataset(expand_with_brace(urls), **dataset_kwargs)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    ).unbatched()
    if training:
        dataloader = dataloader.shuffle(512)
    dataloader = dataloader.batched(
        batch_size,
        collation_fn=partial(collate_with_lengths_wds, flatten=False),
    )
    return dataloader


def write_audio_tar(
    audio_paths: List[str],
    labels: List,
    tar_path: str,
    suffix: str = "wav",
    num_shards: int = 20,
    force: bool = False,
    min_length: int = 100,
):
    assert len(audio_paths) == len(labels), "Number of audio files and labels must match."

    assert len(audio_paths) >= num_shards, "Number of shards must be less than number of audio files."
    shard_size = (len(audio_paths) + num_shards - 1) // num_shards

    def make_sample(filename, label=None):
        with open(filename, "rb") as buf:
            raw_data = buf.read()
        fpath = Path(filename)
        stem_name = str(fpath.stem).replace(".", "_")
        suffix = fpath.suffix.replace(".", "")
        ret_data = {
            suffix: raw_data,
            "__key__": f"{stem_name}",  # Just cast to str
        }
        # If we have some labels, also dump a .json file
        if label is not None:
            if isinstance(label, dict):
                ret_data["json"] = json.dumps(label).encode("utf-8")
            elif isinstance(label, str):
                ret_data["json"] = json.dumps({"label": label}).encode("utf-8")
            else:
                raise ValueError("Label must be either dict or str.")
        return ret_data

    for shard in range(num_shards):
        start_index = shard * shard_size
        end_index = start_index + shard_size

        shard_audio_paths = audio_paths[start_index:end_index]
        shard_labels = labels[start_index:end_index]

        sharded_tar_path = tar_path.replace("*", f"0{shard:05d}")
        if not force and Path(sharded_tar_path).exists():
            logger.info(f"Tar file {sharded_tar_path} already exists.")
            continue

        with wds.TarWriter(sharded_tar_path) as ostream:
            for audio_path, label in zip(shard_audio_paths, shard_labels):
                sample = make_sample(audio_path, label)
                if len(sample[suffix]) < min_length:
                    logger.warning(f"Skipping {audio_path} due to short length.")
                    continue
                ostream.write(sample)


if __name__ == '__main__':

    import torch.distributed as dist

    def setup_ddp():
        """Initialize DDP environment."""
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Running DDP with rank {rank}/{world_size}")
        return rank, world_size

    # setup_ddp()

    ds = create_rawaudio_webdataset(
        [
            # '/gluster-ssd-tts/jiaqi_repos/audioset/data/*.tar',
            # '/gluster-ssd-tts/jiaqi_repos/vggsound/vggsound_00.tar',
            # '/ssd2/lijiaqi18/mtg-jamendo/data/val/processed/*.tar',
            # '/gluster-ssd-tts/jiaqi_repos/mtg-jamendo/val_data/*.tar',
            # '/gluster-ssd-tts/jiaqi_repos/free-music-archive/fma_large.tar',
            # '/gluster-ssd-tts/jiaqi_repos/vocalset/vocalset.tar',
            # '/gluster-ssd-tts/jiaqi_repos/million-song-dataset/108427d78e3941708dce02e0dcd293a2/*.tar',
            # '/gluster-ssd-tts/jiaqi_repos/laion_audio_300M/*.tar',
        ],
        batch_size=8,
        target_sample_rate=16000,
        num_workers=0,
        crop_length=12345,
    )
    # ds = iter(ds)
    for data in ds:
        print(data) # [b, t]
        # break
        
