import os
import random
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
from closd.diffusion_planner.data_loaders.aistpp.utils.get_opt import get_opt
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m

class AISTPPMotionAudioDataset(data.Dataset):
    """Audio->Motion dataset for AIST++ aligned with HumanML3D tuple interface.

    Returns tuple: (audio_embeddings, dummy_pos, caption_placeholder, audio_token_len, motion, motion_len, audio_tokens_joined[, key])
    so that existing t2m_collate can operate with minimal change.
    - audio_embeddings: (N_tokens, F) float32
    - dummy_pos: zeros (N_tokens, 1)
    - caption_placeholder: '' (no text)
    - audio_token_len: int (#tokens)
    - motion: (T, D) normalized
    - motion_len: int (original length before padding)
    - audio_tokens_joined: string e.g. 'a0_a1_...'
    """
    def __init__(self, opt, split='train', device=None):
        self.opt = opt
        self.device = device
        self.split = split
        # stats
        self.motion_mean = np.load(opt.motion_mean_path) if opt.motion_mean_path else None
        self.motion_std = np.load(opt.motion_std_path) if opt.motion_std_path else None
        self.audio_mean = np.load(opt.audio_mean_path) if opt.audio_mean_path else None
        self.audio_std = np.load(opt.audio_std_path) if opt.audio_std_path else None

        self.motion_files = self._gather_files(opt.motion_dir)
        self.audio_files = self._gather_files(opt.audio_dir)
        # build pairing by basename prefix before .npy
        self.pairs = self._match_pairs(self.motion_files, self.audio_files)
        if len(self.pairs) == 0:
            raise RuntimeError(f"No motion/audio pairs found in {opt.motion_dir} and {opt.audio_dir}")

        # infer dim_pose from first motion
        sample_motion = np.load(self.pairs[0][0])
        if sample_motion.ndim == 3:  # (T, J, C)
            t, j, c = sample_motion.shape
            sample_motion = sample_motion.reshape(t, j * c)
        # if we convert to HumanML features, dim_pose becomes 263 (t2m).
        # else keep flattened dimension.
        self.opt.dim_pose = 263 if self.opt.remap_joints else sample_motion.shape[1]

    def _gather_files(self, root):
        if not os.path.isdir(root):
            return []
        return sorted([pjoin(root, f) for f in os.listdir(root) if f.endswith('.npy')])

    def _match_pairs(self, motion_files, audio_files):
        audio_map = {os.path.basename(f).replace('.npy', ''): f for f in audio_files}
        pairs = []
        for m in motion_files:
            base = os.path.basename(m).replace('.npy', '')
            if base in audio_map:
                pairs.append((m, audio_map[base]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_motion(self, path):
        m = np.load(path)
        if m.ndim == 3:
            t, j, c = m.shape
            m = m.reshape(t, j * c)
        return m.astype(np.float32)

    def _load_audio(self, path):
        a = np.load(path)
        # unify to (F, T_a) orientation if needed
        if a.ndim == 1:
            a = a[:, None]  # (F,1)
        if a.shape[0] < a.shape[1]:
            # assume (F, T)
            return a.astype(np.float32)
        else:
            # assume (T, F) -> transpose
            return a.T.astype(np.float32)

    def _time_align_audio(self, audio, target_T):
        # Deprecated: audio is pre-aligned during feature build. Keep as no-op.
        return audio

    def _pool_audio(self, audio):
        # mean pool over time -> (F,)
        return audio.mean(axis=1)

    def _to_humanml_features(self, motion_flat):
        # motion_flat: (T, J*3) -> (1, T, J, 3) -> HumanML feature (1, T-1, 263)
        T, D = motion_flat.shape
        J = D // 3
        pos = motion_flat.reshape(1, T, J, 3).astype(np.float32)
        # extract HumanML features; returns (bs, T-1, 263)
        feats, _ = extract_features_t2m(torch.from_numpy(pos))
        feats = feats.numpy()
        return feats[0]  # (T-1, 263)

    def __getitem__(self, idx):
        motion_path, audio_path = self.pairs[idx]
        motion = self._load_motion(motion_path)
        # optional conversion to HumanML feature representation
        if self.opt.remap_joints:
            motion = self._to_humanml_features(motion)
        orig_len = motion.shape[0]
        # crop to max_motion_length if needed
        start = 0
        if orig_len > self.opt.max_motion_length:
            start = random.randint(0, orig_len - self.opt.max_motion_length)
            motion = motion[start:start + self.opt.max_motion_length]
        # pad if shorter
        if orig_len < self.opt.max_motion_length:
            pad_T = self.opt.max_motion_length - orig_len
            motion = np.concatenate([motion, np.zeros((pad_T, motion.shape[1]), dtype=np.float32)], axis=0)
        # normalize motion (mean/std should match chosen representation)
        m_length = motion.shape[0]
        if self.motion_mean is not None and self.motion_std is not None:
            motion = (motion - self.motion_mean) / self.motion_std

        audio = self._load_audio(audio_path)  # (F, T_a)
        # Expect audio features pre-aligned to original motion length (before crop/pad).
        # Handle minor off-by-one due to representation changes (e.g., HumanML T-1) or rounding.
        F, T_a = audio.shape
        if T_a - 1 != orig_len:
            print(f'[INFO] Audio length {T_a - 1} differs from original motion length {orig_len} for {audio_path}, aligning...')
            if abs(T_a - 1 - orig_len) == 1:
                if T_a - 1 > orig_len:
                    audio = audio[:, :orig_len + 1]
                else:
                    audio = np.concatenate([audio, np.zeros((F, orig_len - T_a + 1), dtype=audio.dtype)], axis=1)
            else:
                # As a safe fallback, linearly resample once to orig_len; expected to be rare if builder aligned.
                xs = np.linspace(0, T_a - 1 - 1, orig_len)
                audio = np.stack([np.interp(xs, np.arange(T_a - 1), audio[f]) for f in range(F)], axis=0)
        # Apply the same crop/pad to audio as applied to motion
        if orig_len > self.opt.max_motion_length:
            audio = audio[:, start:start + self.opt.max_motion_length]
        if orig_len < self.opt.max_motion_length:
            pad_T = self.opt.max_motion_length - orig_len
            audio = np.concatenate([audio, np.zeros((audio.shape[0], pad_T), dtype=audio.dtype)], axis=1)
        audio_tokens = [f'a{i}' for i in range(audio.shape[1])]  # treat time steps as tokens
        audio_embeddings = audio.T  # (T, F)
        token_len = audio_embeddings.shape[0]
        if self.audio_mean is not None and self.audio_std is not None:
            audio_embeddings = (audio_embeddings - self.audio_mean) / self.audio_std

        dummy_pos = np.zeros((token_len, 1), dtype=np.float32)
        caption_placeholder = ''
        tokens_joined = '_'.join(audio_tokens)
        return audio_embeddings, dummy_pos, caption_placeholder, token_len, motion, m_length, tokens_joined

class AISTPP(data.Dataset):
    """Wrapper to mirror HumanML3D interface used by get_data factory."""
    def __init__(self, mode, datapath='./dataset/aistpp_opt.txt', split='train', **kwargs):
        abs_base_path = kwargs.get('abs_path', '.')
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = kwargs.get('device', None)
        opt = get_opt(dataset_opt_path, device)
        opt.fixed_len = kwargs.get('fixed_len', 0)
        if opt.fixed_len > 0:
            opt.max_motion_length = opt.fixed_len
        opt.return_keys = kwargs.get('return_keys', False)
        self.opt = opt
        # Normalize paths to absolute based on abs_base_path
        def _abs(p):
            if p is None:
                return None
            return p if os.path.isabs(p) else pjoin(abs_base_path, p)
        opt.motion_dir = _abs(opt.motion_dir)
        opt.audio_dir = _abs(opt.audio_dir)
        opt.motion_mean_path = _abs(opt.motion_mean_path)
        opt.motion_std_path = _abs(opt.motion_std_path)
        opt.audio_mean_path = _abs(opt.audio_mean_path)
        opt.audio_std_path = _abs(opt.audio_std_path)
        # mean/std already loaded inside dataset per opt file
        self.t2m_dataset = AISTPPMotionAudioDataset(opt, split=split, device=device)
        self.mean_gpu = None
        self.std_gpu = None
        if self.t2m_dataset.motion_mean is not None:
            self.mean_gpu = torch.tensor(self.t2m_dataset.motion_mean, device=device)[None, :, None, None]
            self.mean = torch.tensor(self.t2m_dataset.motion_mean, dtype=torch.float32)
        if self.t2m_dataset.motion_std is not None:
            self.std_gpu = torch.tensor(self.t2m_dataset.motion_std, device=device)[None, :, None, None]
            self.std = torch.tensor(self.t2m_dataset.motion_std, dtype=torch.float32)

    def __len__(self):
        return len(self.t2m_dataset)

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]
