import os
import sys
import numpy as np
from closd.diffusion_planner.data_loaders.get_data import get_dataset

"""
Quick verification for AISTPP setup:
- Loads one sample from the aistpp dataset
- Prints motion per-frame dimension (should be 263 if remap_joints=true)
- Prints motion length and audio embedding shape
- Verifies that motion stats paths exist and match expected dims
Usage:
  python -m closd.diffusion_planner.utils.verify_aistpp_setup \
    --abs_path /train-data-1-ssd/cerosop/data-pred/final/CLoSD_dancing/closd/diffusion_planner
"""

import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abs_path', type=str, required=True,
                    help='Absolute base path to diffusion_planner (where dataset/aistpp_opt.txt resides)')
    args = ap.parse_args()

    ds = get_dataset('aistpp', num_frames=196, split='train', hml_mode='train', abs_path=args.abs_path)
    item = ds[0]
    audio_emb, dummy_pos, caption, token_len, motion, m_len, tokens_joined = item

    print('remap_joints:', ds.opt.remap_joints)
    print('motion shape:', motion.shape, 'per-frame dim:', motion.shape[1])
    print('motion orig length:', m_len)
    print('audio_emb shape:', np.array(audio_emb).shape)
    print('token_len:', token_len)
    print('tokens sample:', tokens_joined[:80] + ('...' if len(tokens_joined)>80 else ''))

    # Check stats files
    m_mean_path = os.path.join(args.abs_path, ds.opt.motion_mean_path)
    m_std_path = os.path.join(args.abs_path, ds.opt.motion_std_path)
    a_mean_path = os.path.join(args.abs_path, ds.opt.audio_mean_path)
    a_std_path = os.path.join(args.abs_path, ds.opt.audio_std_path)

    print('motion_mean_path exists:', os.path.exists(m_mean_path), m_mean_path)
    print('motion_std_path  exists:', os.path.exists(m_std_path), m_std_path)
    print('audio_mean_path  exists:', os.path.exists(a_mean_path), a_mean_path)
    print('audio_std_path   exists:', os.path.exists(a_std_path), a_std_path)

    # Optional dimension check for motion mean/std
    try:
        m_mean = np.load(m_mean_path)
        m_std = np.load(m_std_path)
        print('motion mean/std dims:', m_mean.shape, m_std.shape)
        if m_mean.shape[0] != motion.shape[1]:
            print('[WARN] Motion stats dimension mismatch with loaded motion per-frame dim.')
    except Exception as e:
        print('[WARN] Could not load motion stats:', e)

    try:
        a_mean = np.load(a_mean_path)
        a_std = np.load(a_std_path)
        print('audio mean/std dims:', a_mean.shape, a_std.shape)
    except Exception as e:
        print('[WARN] Could not load audio stats:', e)

if __name__ == '__main__':
    main()
