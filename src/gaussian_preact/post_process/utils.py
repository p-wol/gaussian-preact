import torch

def group_by_class(t, n_classes = 10, exclude_identical = False):
    n_samples = t.size(-1)
    
    assert t.dim() >= 2
    assert t.size(-1) == t.size(-2)
    assert n_samples % n_classes == 0
    
    n_samples_by_cl = n_samples // n_classes
    
    new_size = t.size()[:-2] + (n_samples_by_cl,  n_samples_by_cl,)
    new_t = torch.tensor(0.).resize_(new_size)
    
    for i in range(n_classes):
        ii_min = i * n_samples_by_cl
        ii_max = (i + 1) * n_samples_by_cl
        for j in range(n_classes):
            jj_min = j* n_samples_by_cl
            jj_max = (j + 1) * n_samples_by_cl
            new_t[..., i, j] = t[..., ii_min:ii_max, jj_min:jj_max].mean(-1).mean(-1)
    
    if exclude_identical:
        new_t = new_t * n_samples_by_cl / (n_samples_by_cl - 1) - 1 / (n_samples_by_cl - 1)
    
    return new_t
