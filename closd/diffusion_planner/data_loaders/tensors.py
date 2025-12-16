import numpy as np
import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    # attach audio embeddings if present (per-frame features)
    # For non-prefix collate, treat entire audio as prediction window
    if 'audio_emb_pred' in notnone_batches[0] or 'audio_emb' in notnone_batches[0]:
        key_present = 'audio_emb_pred' if 'audio_emb_pred' in notnone_batches[0] else 'audio_emb'
        audio_batch = [b[key_present] for b in notnone_batches]
        audio_stack = collate_tensors(audio_batch)  # (bs, seq, feat) or (bs, 1, feat)
        cond['y'].update({'audio_embed_pred': audio_stack})
        # Optional: include prefix key when provided in items
        if 'audio_emb_prefix' in notnone_batches[0]:
            audio_prefix_batch = [b['audio_emb_prefix'] for b in notnone_batches]
            cond['y'].update({'audio_embed_prefix': collate_tensors(audio_prefix_batch)})
        # Pool to text_embed from pred window for unified text-conditioning interface
        # Determine the time axis by matching motion length T
        if audio_stack.dim() == 3:
            time_len = databatchTensor.shape[-1]
            if audio_stack.shape[1] == time_len and audio_stack.shape[2] != time_len:
                pooled = audio_stack.mean(dim=1)  # (B, F)
            elif audio_stack.shape[2] == time_len and audio_stack.shape[1] != time_len:
                pooled = audio_stack.mean(dim=2)  # (B, F)
            else:
                # Fallback: assume (B, T, F)
                pooled = audio_stack.mean(dim=1)
        else:
            pooled = audio_stack
        cond['y'].update({'text_embed': pooled.unsqueeze(0)})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if 'prefix' in notnone_batches[0]:
        cond['y'].update({'prefix': collate_tensors([b['prefix'] for b in notnone_batches])})

    if 'key' in notnone_batches[0]:
        cond['y'].update({'db_key': [b['key'] for b in notnone_batches]})

    if 'audio' in notnone_batches[0]:
        cond['y'].update({'audio': [b.get('audio') for b in notnone_batches]})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = []
    for b in full_batch:
        # HumanML3D tuple: (word_emb, pos_ohot, caption, sent_len, motion, m_len, tokens, [key])
        # AISTPP tuple: (audio_emb, dummy_pos, '', token_len, motion, m_len, tokens_joined)
        motion = torch.tensor(b[4].T).float().unsqueeze(1)
        item = {
            'inp': motion,
            'lengths': b[5],
            'tokens': b[6] if len(b) > 6 else None,
        }
        extras = list(b[7:]) if len(b) > 7 else []
        raw_audio_meta = next((x for x in extras if isinstance(x, dict)), None)
        key_val = next((x for x in extras if not isinstance(x, dict)), None)
        if key_val is not None:
            item['key'] = key_val
        # Decide dataset tuple type by inspecting caption and pos shape
        caption = b[2]
        pos = b[1]
        is_humanml = isinstance(caption, str) and caption != ''
        is_aistpp = (not is_humanml) and (isinstance(pos, (np.ndarray, list)) and np.array(pos).ndim == 2 and np.array(pos).shape[1] == 1)

        if is_humanml:
            item['text'] = caption
            # Preserve word embeddings & positional one-hots
            if b[0] is not None and isinstance(b[0], (np.ndarray, list)):
                try:
                    we = np.array(b[0], dtype=np.float32)
                    item['word_emb'] = torch.tensor(we)
                except Exception:
                    pass
            if pos is not None and isinstance(pos, (np.ndarray, list)):
                try:
                    po = np.array(pos, dtype=np.float32)
                    item['pos_ohot'] = torch.tensor(po)
                except Exception:
                    pass
        elif is_aistpp:
            # Treat first element as per-frame audio features; orient to (T,F)
            try:
                audio_np = np.array(b[0])
                if audio_np.ndim == 2:
                    T_motion = b[5]
                    h, w = audio_np.shape
                    if h == T_motion:
                        audio_tw = audio_np
                    elif w == T_motion:
                        audio_tw = audio_np.T
                    else:
                        audio_tw = None
                    if audio_tw is not None:
                        item['audio_emb_pred'] = torch.tensor(audio_tw.astype(np.float32))
            except Exception:
                pass
        # Attach audio embeddings only if it looks like per-frame features aligned to motion length
        try:
            audio_np = np.array(b[0])
            if audio_np.ndim == 2 and ('text' not in item):
                T_motion = b[5]
                h, w = audio_np.shape
                if h == T_motion:
                    T, F = h, w
                    audio_tw = audio_np
                elif w == T_motion:
                    T, F = w, h
                    audio_tw = audio_np.T
                else:
                    audio_tw = None
                if audio_tw is not None and F >= 8 and F <= 2048:
                    item['audio_emb_pred'] = torch.tensor(audio_tw.astype(np.float32))
        except Exception:
            pass

        if raw_audio_meta is not None:
            audio_dict = raw_audio_meta.copy()
            waveform_np = audio_dict.get('waveform')
            if waveform_np is not None and not torch.is_tensor(waveform_np):
                audio_dict['waveform'] = torch.tensor(waveform_np, dtype=torch.float32)
            item['audio'] = audio_dict
        adapted_batch.append(item)
    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = []
    for b in batch:
        full = torch.tensor(b[4].T).float().unsqueeze(1)
        item = {
            'inp': full[..., -pred_len:],
            'prefix': full[..., :-pred_len],
            'tokens': b[6] if len(b) > 6 else None,
            'lengths': pred_len,
        }
        extras = list(b[7:]) if len(b) > 7 else []
        raw_audio_meta = next((x for x in extras if isinstance(x, dict)), None)
        key_val = next((x for x in extras if not isinstance(x, dict)), None)
        if key_val is not None:
            item['key'] = key_val
        caption = b[2]
        pos = b[1]
        is_humanml = isinstance(caption, str) and caption != ''
        is_aistpp = (not is_humanml) and (isinstance(pos, (np.ndarray, list)) and np.array(pos).ndim == 2 and np.array(pos).shape[1] == 1)

        if is_humanml:
            item['text'] = caption
            if b[0] is not None and isinstance(b[0], (np.ndarray, list)):
                try:
                    we = np.array(b[0], dtype=np.float32)
                    item['word_emb'] = torch.tensor(we)
                except Exception:
                    pass
            if pos is not None and isinstance(pos, (np.ndarray, list)):
                try:
                    po = np.array(pos, dtype=np.float32)
                    item['pos_ohot'] = torch.tensor(po)
                except Exception:
                    pass
        elif is_aistpp:
            # Slice audio into prefix and prediction windows aligned with motion split
            try:
                audio_np = np.array(b[0])
                if audio_np.ndim == 2:
                    # Orient audio to (T, F)
                    h, w = audio_np.shape
                    audio_tw = audio_np.T
                    T = audio_tw.shape[0]
                    F = audio_tw.shape[1]
                    # Determine context_len from motion prefix length
                    context_len = item['prefix'].shape[-1]
                    if F >= 8 and F <= 4096:
                        audio_prefix = audio_tw[:, :context_len]
                        audio_pred = audio_tw[:, -pred_len:]
                        # print(f'[DEBUG] Sliced audio into prefix {audio_prefix.shape} and pred {audio_pred.shape}, {audio_tw.shape}, {audio_np.shape}')
                        
                        item['audio_emb_pred'] = torch.tensor(audio_pred.astype(np.float32))
                        item['audio_emb_prefix'] = torch.tensor(audio_prefix.astype(np.float32))
            except Exception:
                pass

        if raw_audio_meta is not None:
            audio_dict = raw_audio_meta.copy()
            waveform_np = audio_dict.get('waveform')
            if waveform_np is not None and not torch.is_tensor(waveform_np):
                audio_dict['waveform'] = torch.tensor(waveform_np, dtype=torch.float32)
            item['audio'] = audio_dict
        adapted_batch.append(item)
    return collate(adapted_batch)

