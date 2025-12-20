# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES
from closd.diffusion_planner.data_loaders.aistpp.data.dataset import AISTPPMotionAudioDataset
import closd.env.tasks.closd
import time

class CLoSDA2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        if not hasattr(AISTPPMotionAudioDataset, 'mean'):
            print("Monkey Patching: adding .mean alias to AISTPPMotionAudioDataset")
            AISTPPMotionAudioDataset.mean = property(lambda self: self.motion_mean)
            AISTPPMotionAudioDataset.std = property(lambda self: self.motion_std)
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.init_state = STATES.TEXT2MOTION    # same as t2m
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_mdm)  # planning_horizon_mdm=_pred_len
        self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        self.audio_embed = None  # store audio embedding info
        self.text_embed = None
        return
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)
        
        # updates prompts and lengths
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data) # re-initialize
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        self.audio_embed = model_kwargs['y']['audio_embed_pred'].to(self.device)
        self.text_embed = model_kwargs['y']['text_embed'].to(self.device)
        for i in env_ids:
            # self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
            # self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]  
            self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]  
            self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]  
            # print(f"=== args: ===\n")
            # for key, value in model_kwargs['y'].items():
            #     if hasattr(value, 'shape'):
            #         print(f"args: {key}, value: {value.shape}")
            #     else:
            #         print(f"args: {key}")
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]  # will be used by the first MDM iteration
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    
    def build_completion_input(self, context_switch_vec=None):
        # input: hml_poses [n_envs, n_frames@mdm_fps, 263]
        # context_switch_vec [n_envs] if not None - indicates which env will use prediction context insted of sim contest
        # output: 
        #   inpainted_motion [bs, 263, 1, max_frames] where hml_poses is the prefix and the rest is zeros
        #   inpainting_mask [bs, 263, 1, max_frames] - true only for the prefix frames

        pose_context = self.pose_buffer  # self.pose_buffer contains last 20 sim poses (for prefix_len=20)
        if self.cfg['env']['dip']['limit_context'] is not None:
            pose_context = pose_context[:, -self.cfg['env']['dip']['limit_context']:]  

        aux_points = None  
        if self.multi_target_cond:
            aux_points = self.calc_cur_target_multi_joint()  # [bs, n_points, 3]     
        
        # pose_context [bs, 30, 24, 3]
        # Real performed motions from the simulator, translated to HML format
        sim_context, translated_aux_points, recon_data = self.rep.pose_to_hml(pose_context, aux_points, fix_ik_bug=True, src_fps=self.isaac_fps, trg_fps=self.mdm_fps)  # [bs, n_frames@mdm_fps, 263], [bs, n_points, 3]
        

        sim_context = sim_context.unsqueeze(2).permute(0, 3, 2, 1)  # [bs, 263, 1, n_frames@mdm_fps]

        if context_switch_vec is not None:
            pred_context = self.cur_mdm_pred[..., -sim_context.shape[-1]:]
            is_pred = context_switch_vec.view(-1, 1, 1, 1)
            hml_context = (is_pred * pred_context) + ((1. - is_pred) * sim_context)
        else:
            hml_context = sim_context

        
        if self.planning_horizon_multiplyer > 1 and self.frame_idx > 0 and self.frame_idx % (self.planning_horizon_30fps * self.planning_horizon_multiplyer) != 0:
            # extand the planning horizon artiffitially. e.g - planning_horizon=40, planning_horizon_multiplyer=2 -> planning horizom will be 80 in practice.
            pred_context = self.cur_mdm_pred[..., -sim_context.shape[-1]:]
            hml_context = pred_context
        
        # only consider motion representation (263d) in this function
        motion_tensor_shape = (self.cfg['env']['num_envs'], 263, self.mdm.nfeats, self.max_frame_mdm)
        context_len = hml_context.shape[-1]
        assert context_len == self.context_len_mdm
        inpainted_motion = torch.zeros(motion_tensor_shape, dtype=torch.float32, device=hml_context.device)
        inpainted_motion[:, :, :, :context_len] = hml_context
        inpainted_motion = inpainted_motion.to(self.mdm_device)
        
        inpainting_mask = torch.zeros(motion_tensor_shape, dtype=torch.bool, device=self.mdm_device)
        inpainting_mask[:, :, :, :context_len] = True  # True means use gt motion

        mask = torch.ones_like(inpainting_mask)[:, [0]]

        aux_entries = {'mask': mask, 'prefix_len': context_len, }

        if self.mdm_cfg_param != 1.:
            aux_entries['scale'] = torch.ones(self.num_envs, device=self.mdm_device) * self.mdm_cfg_param

        # init prefix from humanml real data if exists at the first MDM call
        # used for the text-to-motion task only!
        if hasattr(self, 'hml_prefix_from_data'):
            is_first_iter = self.progress_buf < self.planning_horizon_30fps
            hml_context[is_first_iter] = self.hml_prefix_from_data[is_first_iter, :, :, -self.context_len_mdm:]

        if self.mdm.is_prefix_comp:
            aux_entries.update({'prefix': hml_context})
        else:
            aux_entries.update({'inpainted_motion': inpainted_motion, 'inpainting_mask': inpainting_mask})
        
        if self.multi_target_cond:
            _target_cond = torch.zeros((self.num_envs, self.num_joint_conditions, 3), dtype=translated_aux_points.dtype, device=translated_aux_points.device)
            
            for joint_i in range(2):
                joints_in_use = torch.nonzero(joint_i < self.num_target_joints).squeeze(-1)
                if joints_in_use.shape[0] > 0:
                    _target = translated_aux_points[joints_in_use, joint_i]
                    joint_names_in_use = [self.cur_joint_condition[j][joint_i] for j in joints_in_use]
                    _joint_entry = [self.extended_goal_joint_names.index(j) for j in joint_names_in_use]
                    _target_cond[joints_in_use, _joint_entry] = _target
            
            _target_cond[:, self.extended_goal_joint_names.index('traj'), 1] = 0.   # zero the y axis for the trajectory

            # asign heading
            _target_heading = torch.atan2(translated_aux_points[:, 0, 0], translated_aux_points[:, 0, 2])[:, None]  # heading is according to the first joint
            backward_heading = (self.cur_state == STATES.SIT)
            _target_heading[backward_heading] = _target_heading[backward_heading] % (2*torch.pi) - torch.pi
            _target_cond[self.is_heading, self.extended_goal_joint_names.index('heading'), 0] = _target_heading[self.is_heading][:, 0]

            # update nodel_kwargs
            aux_entries.update({'goal_cond': _target, 'heading_cond': _target_heading})  # for vis
            aux_entries.update({'pred_target_cond': _target_cond, 'target_cond': _target_cond, 'target_joint_names': self.cur_joint_condition, 'is_heading': self.is_heading})
        return aux_entries, recon_data
    
    def get_mdm_next_planning_horizon(self):
        context_switch_vec = None
        if self.context_switch_prob > 0. and self.frame_idx > 0:
            context_switch_vec = torch.bernoulli(self.context_switch_prob * torch.ones([self.num_envs], device=self.device, dtype=self._rigid_body_pos.dtype))
        
        cond_fn = None
        # print(f"[DEBUG] : context_switch_vec = {context_switch_vec}")
        aux_entries, recon_data = self.build_completion_input(context_switch_vec)
        
        # print(f"[DEBUG] : aux_entries = {aux_entries}\nrecon_data = {recon_data}")
        
        # Build MDM inputs
        model_kwargs = {'y': {}}
        model_kwargs['y']['text'] = self.get_text_prompts()
        model_kwargs['y'].update(aux_entries)
        init_image = None
        
        model_kwargs['y']['text_embed'] = self.text_embed
        
        # 算出 audio 的 prefix frame 並 concat 進 prefix motion, 將維度從 [bs, 263, 1, prefix] 變成 [bs, 343, 1, prefix]
        
        starts = torch.clamp(self.progress_buf - self.fake_mdm_args.context_len, min=0).long()
        offsets = torch.arange(20, device=self.device)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, 80)
        audio_prefix = torch.gather(self.audio_embed, 1, gather_indices)
        audio_prefix = audio_prefix.permute(0, 2, 1).unsqueeze(2)
        # print(f"[DEBUG] audio shape {audio_prefix.shape}")
        
        model_kwargs['y']['prefix'] = torch.cat([model_kwargs['y']['prefix'], audio_prefix], dim=1)
        # print(f"[DEBUG] prefix shape {model_kwargs['y']['prefix'].shape}")
        
        # print(f"=== model_kwargs: ===\n")
        # for key, value in model_kwargs['y'].items():
        #     if hasattr(value, 'shape'):
        #         print(f"args: {key}, value: {value.shape}")
        #     else:
        #         print(f"args: {key}")
        
        
        # Run MDM with prefix outpainting
        start_time = time.time()

        sample = self.sample_fn(
            self.mdm,
            self.mdm_tensor_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn,
        )  # [bs, 263, 1, 60]
        # concate 只保留 motion
        sample = sample = sample[:, :263, ...]
        self.cur_mdm_pred = sample  # used in the context_switch feature
        # print(f"[DEBUG] cur_mdm_pred shape: {self.cur_mdm_pred.shape}")
        if self.time_prints:
            print('=== sample mdm for [{}] envs took [{:.2f}] sec'.format(sample.shape[0], time.time() - start_time))

        sample_reshaped = sample.squeeze(2).permute(0, 2, 1)
        sample_xyz = self.rep.hml_to_pose(sample_reshaped, recon_data, sim_at_hml_idx=model_kwargs['y']['prefix_len']-1, src_fps=self.mdm_fps, trg_fps=self.isaac_fps)  # hml rep [bs, n_frames_mdm_fps, 263] -> smpl xyz [bs, n_frames_30fps, 24, 3]

        if self.cfg['env']['dip']['debug_hml']:
            print(f'in get_mdm_next_planning_horizon: prompts={model_kwargs["y"]["text"][:2]}')
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in get_mdm_next_planning_horizon: prompts={self.hml_prompts[:2]}')
            self.visualize(sample[:1],
                               'mdm_debug/prefixComp_{}_{}.mp4'.format(self.frame_idx, self.hml_prompts[0].replace('.', '').replace(' ', '_')),
                               is_prefix_comp=True, model_kwargs=model_kwargs,)

        # Extract the planning horizon
        context_len_30fps = int(model_kwargs['y']['prefix_len'] * self.isaac_fps / self.mdm_fps)
        planning_horizon = sample_xyz[:, context_len_30fps-1:context_len_30fps+self.planning_horizon_30fps]  # [x, -z, y]

        return planning_horizon[:, 0], planning_horizon[:, 1:]
    
    def get_text_prompts(self):
        return 'None'