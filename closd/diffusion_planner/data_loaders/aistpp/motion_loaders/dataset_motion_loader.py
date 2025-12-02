import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from closd.diffusion_planner.data_loaders.aistpp.data.dataset import AISTPPMotionAudioDataset
from closd.diffusion_planner.data_loaders.aistpp.utils.get_opt import get_opt

"""Simplified motion loader for AIST++ (no text)."""

def get_aistpp_motion_loader(opt_path, batch_size, device):
    opt = get_opt(opt_path, device)
    dataset = AISTPPMotionAudioDataset(opt, split='test', device=device)
    # Return raw tuples; caller may adapt to evaluation logic
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    return dataloader, dataset
