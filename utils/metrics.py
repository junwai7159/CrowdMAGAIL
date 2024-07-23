import cv2
import torch
import numpy as np
import os
import math
import pickle
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from foronoi import Voronoi, Polygon, Visualizer, VoronoiObserver

from configs.config import get_args
from utils.action import mod2pi
from utils.visualization_cv import plot_heatmap

ARGS = get_args()
xrange = {'GC': (5, 25), 'UCY': (-6, 20)}
yrange = {'GC': (15, 35), 'UCY': (-8, 20)}

# find DCA (distance of closest approch), TTCA (time to closest approach)
def get_ttcmd(env, TTC_MAX=20.0, FRIEND_DIS=5.0, FRIEND_RATIO=0.7):
    """
    calculate the TTC and MD between N*N pairs of pedestrians at T steps, return the MD, TTC, and MASK
    - MD: (N, N, T), the closest distance between two pedestrians within future TTC_MAX time
    - TTC: (N, N, T), the time they take to reach their closest distance
    - MASK: (N, N, T), bool, whether the corresponding MD & TTC is valid.
        - when i cannot see j at time t, (i, j, t) is invalid
        - when i and j are friends, (i, j, :) is invalid
    """
    # MD: DCA (Distance of Closest Approach)
    # TTC: TTCA (Time to Closest Approach)
    # TTC_MAX: T_clip

    N, T = env.num_pedestrians, env.num_steps
    time_idx = torch.arange(T, device=env.device) # (t,)
    self_idx = torch.arange(N, device=env.device) # (n,)
    peds_idx = torch.arange(N, device=env.device) # (N,)
    pair_idx = torch.stack(torch.meshgrid(self_idx, peds_idx), dim=-1)  # (n, N, 2)
    pos = env.position[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 2)
    vel = env.velocity[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 2)
    drc = env.direction[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 1)
    xx = (pos[:, :, 0] - pos[:, :, 1]).square().sum(dim=-1)  # (n, N, t)
    xv = ((pos[:, :, 0] - pos[:, :, 1]) * (vel[:, :, 0] - vel[:, :, 1])).sum(dim=-1)  # (n, N, t)
    vv = (vel[:, :, 0] - vel[:, :, 1]).square().sum(dim=-1)  # (n, N, t)

    # calculate MD & TTC
    r2 = 2 * env.ped_radius
    d_now = xx.sqrt() - r2
    d_max = (vv * TTC_MAX**2 + 2 * xv * TTC_MAX + xx).clamp(1e-8).sqrt() - r2
    d_min = (xx - xv ** 2 / vv.clamp(1e-8)).clamp(1e-8).sqrt() - r2
    t_min = -xv / vv.clamp(1e-8)    # ?
    t_col = (-xv - (xv ** 2 - (xx - r2**2) * vv).sqrt()) / vv.clamp(1e-8)   # ?
    md = d_min.clamp(0).where(t_min <= TTC_MAX, d_max.clamp(0)).where(t_min >= 0, d_now.clamp(0.))
    ttc = t_min.clamp(0, TTC_MAX).where(t_col.isnan() | (t_min <= 0), t_col.clamp(0., TTC_MAX))

    # calculate MASK
    dp = pos[:, :, 1] - pos[:, :, 0]  # (n, N, t, 2)
    view = mod2pi(torch.atan2(dp[:, :, :, 1], dp[:, :, :, 0]) - drc[:, :, 0, :, 0]).abs() < np.pi / 2  # (n, N, t)
    # zone = (xx.sqrt() - r2 < 0.45)
    # view |= zone & ((view & zone).cumsum(dim=-1) > 0)
    friend = ((xx < FRIEND_DIS ** 2).float().sum(dim=-1, keepdim=True) / pos.isnan().any(dim=-1).all(dim=2).logical_not().sum(dim=-1, keepdim=True) > FRIEND_RATIO)  # (N, N, 1)
    # into_env = env.mask.clone(); into_env[:, 1:] &= ~into_env[:, :-1]  # (N, T)
    # exit_env = env.mask.clone(); exit_env[:, :-1] &= ~exit_env[:, 1:]  # (N, T)
    # into_env[env.mask.any(dim=-1).logical_not(), -1] = True
    # exit_env[env.mask.any(dim=-1).logical_not(), 0] = True
    # assert (into_env.sum(dim=1) == 1).all() and (exit_env.sum(dim=1) == 1).all(), "A pedestrian enter the environment for more than 1 times!"
    # src = env.position[into_env, :]  # (N, 2)
    # tgt = env.position[exit_env, :]  # (N, 2)
    # friend &= (src[pair_idx, :].diff(dim=2).norm(dim=-1) < FRIEND_DIS) & (tgt[pair_idx, :].diff(dim=2).norm(dim=-1) < FRIEND_DIS)  # (N, 2) -> (N, N, 2, 2) -> (N, N, 1, 2) -> (N, N, 1)
    msk = (view) & (~friend)  # (n, N, t)

    assert (md[msk] >= 0).all() and (ttc[msk] >= 0).all() and (ttc[msk] <= TTC_MAX).all(), "there must be something wrong!"
    return md, ttc, msk


##### Macroscopic Metrics #####
def calc_speed(env_real=None, env_imit=None):
    real_speed = []
    imit_speed = []
    
    if env_real is not None:
        for f in range(env_real.velocity.shape[1]):
            speed = env_real.velocity[env_real.mask[:, f], f].norm(dim=-1).mean().item()
            real_speed.append(0 if np.isnan(speed) else speed)
    for f in range(env_imit.velocity.shape[1]):
        speed = env_imit.velocity[env_imit.mask[:, f], f].norm(dim=-1).mean().item()
        imit_speed.append(0 if np.isnan(speed) else speed)
    if env_real is not None:
        delta_speed =  (torch.tensor(real_speed) - torch.tensor(imit_speed)).abs().mean()
    else:
        delta_speed = torch.tensor(imit_speed).mean()
    return delta_speed.item()

def calc_density(env_real=None, env_imit=None):    
    def calc_voronoi_cell_area(env, bbox_poly):
        density = []
        for f in tqdm(range(env.position.shape[1]), desc='Calculating density ...'):
            vor = Voronoi(bbox_poly)
            # vor.attach_observer(VoronoiObserver())
            position = [tuple(pos) for pos in env.position[env.mask[:, f], f].numpy()]
            vor.create_diagram(points=position)
            # Visualizer(vor, canvas_offset=1).plot_sites(show_labels=True).plot_edges(show_labels=False).plot_vertices().plot_border_to_site().show()
            areas = [point.area()  + 1e-10 for point in vor.sites]
            density.append(np.mean(1.0 / np.array(areas)))
        return density
    
    # define bounding box
    bbox_min_x, bbox_max_x = xrange[ARGS.DATASET]
    bbox_min_y, bbox_max_y = yrange[ARGS.DATASET]
    bbox = np.array([[bbox_min_x, bbox_min_y], [bbox_max_x, bbox_min_y],
                     [bbox_max_x, bbox_max_y], [bbox_min_x, bbox_max_y]])
    bbox_poly = Polygon(bbox)

    
    # generate the voronoi diagram
    if env_real is not None:
        real_density = calc_voronoi_cell_area(env_real, bbox_poly)
    imit_density = calc_voronoi_cell_area(env_imit, bbox_poly)

    if env_real is not None:
        density = (torch.tensor(real_density) - torch.tensor(imit_density)).abs().mean()
    else:
        density = torch.tensor(imit_density).mean()
    return density.item()
    
##### Microscopic Metrics #####
def calc_collision(env_imit):
    N, T, _ = env_imit.position.shape
    num_collided_agents = []

    for t in range(T):
        collision_agent_table = torch.full((N,), False)
        for i in range(N):
            for j in range(i + 1, N):
                if env_imit.mask[i, t] and env_imit.mask[j, t]:
                    if torch.norm(env_imit.position[i, t, :] - env_imit.position[j, t, :], dim=-1) <= (2 * env_imit.ped_radius) ** 2:
                        collision_agent_table[i] = True
                        collision_agent_table[j] = True
        num_collided_agents.append(torch.sum(collision_agent_table).item())

    # return torch.mean(torch.tensor(num_collided_agents) / N)
    return torch.tensor(num_collided_agents).int().sum().item()

def calc_displacement(env_real=None, env_imit=None):
    real_displacement = []
    imit_displacement = []

    if env_real is not None:
        for id, pos in enumerate(env_real.position):
            displacement = pos[env_real.mask[id, :], :].diff(dim=0).norm(dim=1).mean().item()
            real_displacement.append(0 if math.isnan(displacement) else displacement)
    for id, pos in enumerate(env_imit.position):
        displacement = pos[env_imit.mask[id, :], :].diff(dim=0).norm(dim=1).mean().item()
        imit_displacement.append(0 if math.isnan(displacement) else displacement)
    if env_real is not None:
        delta_displacement = (torch.tensor(real_displacement) - torch.tensor(imit_displacement)).abs().mean()
    else:
        delta_displacement = torch.tensor(imit_displacement).mean()
    return delta_displacement.item()

def calc_deviation(env_real=None, env_imit=None):
    real_perp_dev_distance = []
    imit_perp_dev_distance = []

    # dist = |Ax_0 + By_0 + C| / sqrt(A^2 + B^2)
    # A = y_2 - y_1; B = x_1 - x_2; C = x-2 * y_1 - x_1 * y_2
    if env_real is not None:
        for id, pos in enumerate(env_real.position):
            p0 = pos[env_real.mask[id, :], :]
            p1 = p0[0]
            p2 = env_real.destination[id]

            A = p2[1] - p1[1]
            B = p1[0] - p2[1]
            C = p2[0] * p1[1] - p1[0] * p2[1]
            real_perp_dev_distance.append(((A * p0[:, 0] + B * p0[:, 1] + C) / torch.sqrt(A**2 + B**2)).abs().mean().item())

    for id, pos in enumerate(env_imit.position):
        p0 = pos[env_imit.mask[id, :], :]
        p1 = pos[0, :]
        p1 = p0[0]
        p2 = env_imit.destination[id]
        A = p2[1] - p1[1]
        B = p1[0] - p2[1]
        C = p2[0] * p1[1] - p1[0] * p2[1]
        imit_perp_dev_distance.append(((A * p0[:, 0] + B * p0[:, 1] + C) / torch.sqrt(A**2 + B**2)).abs().mean().item())        
    
    if env_real is not None:
        perp_dev_distance = (torch.tensor(real_perp_dev_distance) - torch.tensor(imit_perp_dev_distance)).abs().mean()
    else:
        perp_dev_distance = torch.tensor(imit_perp_dev_distance).mean()
    return perp_dev_distance.item()

def calc_v_locomotion(env_real=None, env_imit=None):
    real_v_locomotion = []
    imit_v_locomotion = []

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity):
            v_locomotion = vel[env_real.mask[id, :], :].norm(dim=1).diff().abs().mean().item()
            real_v_locomotion.append(0 if np.isnan(v_locomotion) else v_locomotion)
    for id, vel in enumerate(env_imit.velocity):
        v_locomotion = vel[env_imit.mask[id, :], :].norm(dim=1).diff().abs().mean().item()
        imit_v_locomotion.append(0 if np.isnan(v_locomotion) else v_locomotion)

    if env_real is not None:
        delta_v_locomotion = (torch.tensor(real_v_locomotion) - torch.tensor(imit_v_locomotion)).abs().mean()
    else:
        delta_v_locomotion = torch.tensor(imit_v_locomotion).mean()

    return delta_v_locomotion.item()

def calc_a_locomotion(env_real=None, env_imit=None):
    real_a_locomotion = []
    imit_a_locomotion = []
    
    # angle_diff = (angle1 - angle2 + pi) % 2pi - pi
    if env_real is not None:
        for id, ang in enumerate(env_real.direction):
            a_locomotion = torch.mean(torch.abs((torch.diff(ang[env_real.mask[id, :], :].squeeze(dim=1)) + np.pi) % (2 * np.pi) - np.pi)).item()
            real_a_locomotion.append(0 if np.isnan(a_locomotion) else a_locomotion)
    for id, ang in enumerate(env_imit.direction):
        a_locomotion = torch.mean(torch.abs((torch.diff(ang[env_imit.mask[id, :], :].squeeze(dim=1)) + np.pi) % (2 * np.pi) - np.pi)).item()
        imit_a_locomotion.append(0 if np.isnan(a_locomotion) else a_locomotion)
    
    if env_real is not None:
        delta_a_locomotion = (torch.tensor(real_a_locomotion) - torch.tensor(imit_a_locomotion)).abs().mean()
    else:
        delta_a_locomotion = torch.tensor(imit_a_locomotion).mean()
    
    return delta_a_locomotion.item()

def calc_energy(env_real=None, env_imit=None):
    real_energy = []
    imit_energy = []

    dt = env_imit.meta_data['time_unit']
    mass = 60
    e_s = 2.23
    e_w = 1.26
    # e_r = 1.0

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity):
            v = torch.norm(vel[env_real.mask[id,:], :], dim=1)
            real_energy.append(torch.mean(mass * (e_s + e_w * v ** 2) * dt).item())
    for id, vel in enumerate(env_imit.velocity):
        v = torch.norm(vel[env_imit.mask[id, :], :], dim=1)
        imit_energy.append(torch.mean(mass * (e_s + e_w * v ** 2) * dt).item())

    if env_real is not None:
        energy = torch.mean(torch.abs(torch.tensor(real_energy) - torch.tensor(imit_energy)))
    else:
        energy = torch.mean(torch.tensor(imit_energy))
    return energy.item()

def calc_steer_energy(env_real=None, env_imit=None):
    real_steer_energy = []
    imit_steer_energy = []

    dt = env_imit.meta_data['time_unit']
    mass = 60
    e_s = 2.23
    e_w = 1.26

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity):
            real_steer_energy.append(torch.mean(mass * (e_s + e_w * torch.norm(torch.diff(vel[env_real.mask[id, :], :], dim=1), dim=1) ** 2) * dt).item())
    for id, vel in enumerate(env_imit.velocity):
        imit_steer_energy.append(torch.mean(mass * (e_s + e_w * torch.norm(torch.diff(vel[env_imit.mask[id, :], :], dim=1), dim=1) ** 2) * dt).item())

    if env_real is not None:
        steer_energy = torch.mean(torch.abs(torch.tensor(real_steer_energy) - torch.tensor(imit_steer_energy)))
    else:
        steer_energy = torch.mean(torch.tensor(imit_steer_energy))
    return steer_energy.item()


##### Other Metrics #####
def calc_SSIM(env_real, env_imit):
    heatmap_real = plot_heatmap(env_real, xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET], return_array=True)
    heatmap_imit = plot_heatmap(env_imit, xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET], return_array=True)
    
    gray_real = cv2.cvtColor(heatmap_real, cv2.COLOR_BGR2GRAY)
    gray_imit = cv2.cvtColor(heatmap_imit, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray_real, gray_imit, full=True)

    return score