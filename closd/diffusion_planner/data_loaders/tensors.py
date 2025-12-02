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

    # attach audio embeddings if present (for AISTPP)
    if 'audio_emb' in notnone_batches[0]:
        audio_batch = [b['audio_emb'] for b in notnone_batches]
        # Keep audio_emb time-aligned per-frame for conditioning over time
        audio_stack = collate_tensors(audio_batch)  # (bs, seq, feat) or (bs, 1, feat)
        cond['y'].update({'audio_emb': audio_stack})
        # Force text_embed to be pooled (global) version for unified text-conditioning interface
        # Pool across the sequence dimension to get (bs, feat)
        if audio_stack.dim() == 3:
            pooled = audio_stack.mean(dim=1)  # (bs, feat)
        else:
            pooled = audio_stack  # (bs, feat) if already (bs, feat)
        # Convert to tokens-first layout with a single token: (1, bs, feat)
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
            'key': b[7] if len(b) > 7 else None,
        }
        # Optional text for HumanML
        if isinstance(b[2], str):
            item['text'] = b[2]
        # Attach audio embeddings if first element looks like features
        try:
            audio_np = np.array(b[0], dtype=np.float32)
            item['audio_emb'] = torch.tensor(audio_np)
        except Exception:
            pass
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
            'key': b[7] if len(b) > 7 else None,
        }
        if isinstance(b[2], str):
            item['text'] = b[2]
        try:
            audio_np = np.array(b[0], dtype=np.float32)
            item['audio_emb'] = torch.tensor(audio_np)
        except Exception:
            pass
        adapted_batch.append(item)
    return collate(adapted_batch)

