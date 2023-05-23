# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import subprocess
from webdataset import WebLoader

import utils
import torch
from torchvision import transforms
from braceexpand import braceexpand
from .datasets import RawFrameClsDataset, VideoClsDataset
from .masking_generator import (
    RunningCellMaskingGenerator,
    TubeMaskingGenerator,
)
from einops import rearrange
from .pretrain_datasets import HybridVideoMAE, VideoMAE, Video2DatasetWrapper  # noqa: F401
from .transforms import (
    GroupMultiScaleCrop,
    GroupNormalize,
    Stack,
    ToTorchFormatTensor,
)
from torchvision.transforms import ToPILImage
from functools import partial
from utils import multiple_pretrain_samples_collate

import webdataset as wds
from webdataset import WebLoader
from video2dataset.dataloader import get_video_dataset

class DataAugmentationForVideoMAEv2(object):

    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        div = True
        roll = False
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size,
                                                      [1, .875, .75, .66])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=roll),
            ToTorchFormatTensor(div=div),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.encoder_mask_map_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio)
        else:
            raise NotImplementedError(
                'Unsupported encoder masking strategy type.')
        if args.decoder_mask_ratio > 0.:
            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(
                    args.window_size, args.decoder_mask_ratio)
            else:
                raise NotImplementedError(
                    'Unsupported decoder masking strategy type.')

    def __call__(self, images):
        process_data, _ = self.transform(images)
        encoder_mask_map = self.encoder_mask_map_generator()
        if hasattr(self, 'decoder_mask_map_generator'):
            decoder_mask_map = self.decoder_mask_map_generator()
        else:
            decoder_mask_map = 1 - encoder_mask_map
        return process_data, encoder_mask_map, decoder_mask_map

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAEv2,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Encoder Masking Generator = %s,\n" % str(
            self.encoder_mask_map_generator)
        if hasattr(self, 'decoder_mask_map_generator'):
            repr += "  Decoder Masking Generator = %s,\n" % str(
                self.decoder_mask_map_generator)
        else:
            repr += "  Do not use decoder masking,\n"
        repr += ")"
        return repr

   
def custom_collat(samp):
    uncollated = samp["mp4"]
    images = []
    bool_masked_pos = []
    decode_masked_pos = []
    for sample in uncollated:
        images.extend(sample["mp4"])
        bool_masked_pos.extend(sample["encoder_mask"])
        decode_masked_pos.extend(sample["decoder_mask"])
    images = torch.stack(images)
    bool_masked_pos = torch.stack(bool_masked_pos)
    decode_masked_pos = torch.stack(decode_masked_pos)

    return (images, bool_masked_pos, decode_masked_pos)
'''
class DataLoaderWrapper:
    def __init__(self, dl, num_videos):
        self.dl = dl
        self.num_videos = num_videos

    def __iter__(self):
        return iter(self.dl)

    def __len__(self):
        return self.num_videos
'''
def custom_transform_v2d(video_frames, new_step, transform, t, new_length, num_sample):
    print("Starting transform", flush=True)
    video_frames_sampled = video_frames[::new_step, :, :, :].permute(0, 3, 1, 2)
    images = [
        t(tensor) for tensor in video_frames_sampled
    ]
    if num_sample > 1:
        process_data_list = []
        encoder_mask_list = []
        decoder_mask_list = []
        for _ in range(num_sample):
            process_data, encoder_mask, decoder_mask = transform(
                (images, None))
            process_data = process_data.view(
                (new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            process_data_list.append(process_data)
            encoder_mask_list.append(encoder_mask)
            decoder_mask_list.append(decoder_mask)
        out_batch = [[process_data_list, encoder_mask_list, decoder_mask_list]]
    else:
        process_data, encoder_mask, decoder_mask = transform(
            (images, None)
        )
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view(
            (new_length, 3) + process_data.size()[-2:]).transpose(
                0, 1)
        out_batch = [[process_data, encoder_mask, decoder_mask]]
    batch = multiple_pretrain_samples_collate(out_batch, fold=False)
    print("Transform finished", flush=True)
    return {
        'mp4': batch[0],
        'encoder_mask': batch[1],
        'decoder_mask': batch[2]
    }

def get_v2d_dl(args):
    transform = DataAugmentationForVideoMAEv2(args)
    shards = args.data_path
    
    decoder_kwargs = {
        'n_frames': args.num_frames*args.sampling_rate,
        'fps': None,
        'num_threads': 4
    }
    t = ToPILImage()
    transform_dict = {
        'mp4': partial(
            custom_transform_v2d, 
            transform=transform, 
            num_sample=args.num_sample, 
            new_length=args.num_frames,
            new_step=args.sampling_rate,
            t=t
        )
    }
    dl = get_video_dataset(
        urls=shards,
        decoder_kwargs=decoder_kwargs,
        batch_size=1,
        resize_size=(360,640),
        enforce_additional_keys=[],
        handler=wds.warn_and_continue,
        custom_transforms=transform_dict,
    )
    #print('THREADS', torch.get_num_threads(), flush=True)
    torch.set_num_threads(100)
    print('THREADS', torch.get_num_threads(), flush=True)
    dl = wds.WebLoader(
        dl,
        batch_size=args.batch_size,
        num_workers=12,
        collate_fn=custom_collat,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=utils.seed_worker
    ).with_length(args.num_samples_per_worker)

    print("Data Aug = %s" % str(transform))
    return dl

def build_pretraining_dataset(args):
    if args.use_video2dataset:
        dataset = get_v2d_dl(args)
        return dataset

    transform = DataAugmentationForVideoMAEv2(args)
    print("Data Aug = %s" % str(transform))
    
    dataset = VideoMAE(
        root=args.data_root,
        setting=args.data_path,
        train=True,
        test_mode=False,
        name_pattern=args.fname_tmpl,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=1,
        num_crop=1,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=args.num_sample)
    return dataset


def build_dataset(is_train, test_mode, args):
    if is_train:
        mode = 'train'
        anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode:
        mode = 'test'
        anno_path = os.path.join(args.data_path, 'val.csv')
    else:
        mode = 'validation'
        anno_path = os.path.join(args.data_path, 'val.csv')

    if args.data_set == 'Kinetics-400':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 400

    elif args.data_set == 'Kinetics-600':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 600

    elif args.data_set == 'Kinetics-700':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 700

    elif args.data_set == 'Kinetics-710':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 710

    elif args.data_set == 'SSV2':
        dataset = RawFrameClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.fname_tmpl,
            start_idx=args.start_idx,
            args=args)

        nb_classes = 174

    elif args.data_set == 'UCF101':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set == 'HMDB51':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'Diving48':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    elif args.data_set == 'MIT':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 339
    else:
        raise NotImplementedError('Unsupported Dataset')

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
