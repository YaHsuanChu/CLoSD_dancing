import os
from argparse import Namespace
from os.path import join as pjoin

"""AIST++ option parser (key=value format).
Reads aistpp_opt.txt created earlier. Provides fields similar to HumanML get_opt but
without any text-related attributes.
"""

def parse_line(line: str):
    if '=' not in line:
        return None, None
    key, value = line.strip().split('=', 1)
    value = value.strip()
    if value.lower() in ('true', 'false'):
        return key, value.lower() == 'true'
    try:
        if '.' in value:
            return key, float(value)
        return key, int(value)
    except ValueError:
        return key, value


def get_opt(opt_path, device=None):
    opt = Namespace()
    opt_dict = vars(opt)
    with open(opt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, v = parse_line(line)
            if k is not None:
                opt_dict[k] = v

    # required base fields similar to humanml
    opt.dataset_name = 'aistpp'
    opt.device = device
    opt.fps = float(opt_dict.get('fps', 60.0))

    # joints: if converting to SMPL set 22 else keep raw
    opt.joints_num = int(opt_dict.get('n_joints', 22))
    opt.max_motion_length = int(opt_dict.get('max_motion_frames', 2877))
    opt.dim_pose = None  # will be inferred after first motion load

    # paths (allow env override)
    opt.edge_root = os.path.expandvars(opt_dict.get('edge_root', '.'))
    # allow either absolute motion_dir/audio_dir or relative to abs_path provided by caller
    opt.motion_dir = os.path.expandvars(opt_dict.get('motion_dir', opt_dict.get('train_motion_dir', '')))
    opt.audio_dir = os.path.expandvars(opt_dict.get('audio_dir', opt_dict.get('train_audio_dir', '')))
    opt.audio_wav_dir = os.path.expandvars(opt_dict.get('audio_wav_dir', ''))
    opt.audio_feature_type = opt_dict.get('audio_feature_type', 'baseline_feats')

    opt.motion_mean_path = opt_dict.get('motion_mean_path')
    opt.motion_std_path = opt_dict.get('motion_std_path')
    opt.audio_mean_path = opt_dict.get('audio_mean_path')
    opt.audio_std_path = opt_dict.get('audio_std_path')

    opt.pool_audio = bool(opt_dict.get('pool_audio', True))
    opt.audio_time_align = opt_dict.get('audio_time_align', 'linear')

    opt.remap_joints = bool(opt_dict.get('remap_joints', False))
    if 'dim_pose' in opt_dict:
        try:
            opt.dim_pose = int(opt_dict['dim_pose'])
        except Exception:
            opt.dim_pose = None

    # placeholder fields expected downstream (kept for compatibility)
    opt.unit_length = 4  # motion cropping granularity
    opt.fixed_len = int(opt_dict.get('fixed_len', 0))
    opt.return_keys = False
    opt.disable_offset_aug = False

    return opt
