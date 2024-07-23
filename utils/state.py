import torch

########## STATE ##########
def pack_state(s_self, s_int, s_ext):  # (N, 1) & (N, 8) & (N, 20, 8) -> (N, 169)
    s = torch.cat([s_self, s_int, s_ext.view(s_ext.shape[0], -1)], dim=-1)
    return s

def unpack_state(s):  # (N, 169) -> (N, 1) & (N, 8) & (N, 20, 8)
    s_self, s_int, s_ext = s.split((1, 8, s.shape[1] - 1 - 8), dim=-1)
    s_ext = s_ext.view(s_ext.shape[0], -1, 8)
    return s_self, s_int, s_ext

