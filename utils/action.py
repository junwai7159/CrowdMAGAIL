import torch
import numpy as np

########## ACTION ########## 
def rotate(vec, ang):
    """
    rotate a vector by a angle
    :param vec: (N, 2)
    :param ang: (N,)
    """
    c, s = ang.cos(), ang.sin()
    mat = torch.stack([c, -s, s, c], dim=0).view(2, 2, *vec.shape[:-1])  # (2, 2, N)
    vec_ = torch.einsum('ji...,...i->...j', mat, vec)  # (N, 2)
    return vec_

def xy2ra(xy):
    """
    :param xy: [N, 2]
    :return: ra: [N, 2]
    """
    r = torch.norm(xy, dim=-1)
    a = torch.atan2(xy[:, 1], xy[:, 0])
    ra = torch.stack([r, a], dim=-1)
    return ra

def ra2xy(ra):
    """
    :param ra: [N, 2]
    :return: xy: [N, 2]
    """
    x = ra[:, 0] * torch.cos(ra[:, 1])
    y = ra[:, 0] * torch.sin(ra[:, 1])
    xy = torch.stack([x, y], dim=-1)
    return xy

def xy2rscnt(pos, vel, dir=0):
    """
    get observation states
    :param pos: [N, 2]
    :param vel: [N, 2]
    :return rscnt: [N, 8]
        r: distance
        s: sin(orientation), left(+) right(-)
        c: cos(orientation), front(+) back(-)
        n: depature speed, departure(+) approach(-)
        t: circular velocity, anticlockwise(+), clockwise(-)
        a: orientation
        x: r * c
        y: r * s
    """
    
    r = pos.norm(dim=-1, keepdim=True)
    a = mod2pi(torch.atan2(pos[:, 1], pos[:, 0]).unsqueeze(dim=-1) - dir)
    s = a.sin()
    c = a.cos()
    x = r * c
    y = r * s
    _r = 1. / (r + 1e-8)
    n = (pos * vel).sum(dim=-1, keepdim=True) * _r
    t = torch.diff((pos.flip(-1) * vel), dim=-1) * _r
    rscnt = torch.cat([r, s, c, n, t, a, x, y], dim=-1)
    return rscnt

def mod2pi(delta_angle):
    """
    map a angle in (-2pi, 2pi) into (-pi, pi), used to deal with angle differences
    - -2pi < x < -pi: return x + 2pi
    - -pi < x < +pi: return x
    - +pi < x < +2pi: return x - 2pi
    """
    return torch.remainder(delta_angle + np.pi, 2 * np.pi) - np.pi
