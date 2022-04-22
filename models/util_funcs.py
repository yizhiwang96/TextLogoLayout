import torch

def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)
    
    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)
    
    return (torch.arange(0,max_len,device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

def cal_overlap_loss(imgs_trans, max_seqlen):
    relu = torch.nn.ReLU()
    loss_overlap = 0.0
    rendered_pre = torch.zeros(imgs_trans[..., 0].shape).to(imgs_trans.device)
    for idx in range(max_seqlen):
        loss_overlap += torch.mean(imgs_trans[..., idx] * rendered_pre)
        # update rendered_previous (t = 0, 1, ..., idx) 
        rendered_pre = rendered_pre + imgs_trans[..., idx]
        rendered_pre = 1 - relu(1 - rendered_pre)
    return loss_overlap
