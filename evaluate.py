import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.ppo import PPO
from model.sfm import SFM
from model.orca import ORCA
from envs.pedsim import Pedsim
from configs.config import get_args, set_seed
from utils.state import pack_state
from utils.action import   mod2pi
from utils.metrics import *
from utils.visualization_cv import generate_gif, plot_traj, plot_heatmap
from utils.preprocess import RawData

xrange = {'GC': (5, 25), 'UCY': (-6, 20)}
yrange = {'GC': (15, 35), 'UCY': (-8, 20)}

if __name__ == '__main__':
    ARGS = get_args()
    set_seed(ARGS.SEED)
    data = RawData(f'./configs/data_configs/pretrain_{ARGS.DATASET}.yaml')
    test_data = (data.raw_data['test'][0])
    N = test_data['num_pedestrians']
    T = test_data['num_steps']
    env_real = test_data['env']

    # init RL model
    if ARGS.MODEL in ['TECRL', 'MAGAIL']:
        model = PPO(ARGS).to(ARGS.DEVICE)
        state_dict  = torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE))
        model.load_state_dict(state_dict, strict=False)
        # print(model)

    # metrics
    metrics_keys = ['Speed', 'Displacement', 'Deviation', 'Velocity', 'Angle', 'Energy', 'Steer', 'SSIM']
    metrics = {metric_key: None for metric_key in metrics_keys}

    agent_set = set()
    pbar = tqdm(range(T))
    for t in pbar:
        if t == 0:
            # filter agent from env_real to env_imit
            imit_agent_mask = (~torch.isnan(env_real.position[:, t, :]).any(dim=1)).nonzero().squeeze() # (N_imit,)
            agent_set.update(imit_agent_mask.tolist())

            # initialize  env_imit
            env_imit = Pedsim(env_real.args)
            env_imit.meta_data = env_real.meta_data
            env_imit.add_pedestrian(env_real.position[imit_agent_mask, t, :], env_real.velocity[imit_agent_mask, t, :], env_real.destination[imit_agent_mask, :], init=True)
            env_imit.add_obstacle(env_real.obstacle)

            if ARGS.MODEL == 'SFM':
                model = SFM(env_imit, ARGS)
            elif ARGS.MODEL == 'ORCA':
                model = ORCA(env_imit, ARGS)
            env_imit.ppo = model

        else:
            imit_agent_mask = (~torch.isnan(env_real.position[:, t, :]).any(dim=1)).nonzero().squeeze() # (N_imit,)
            pre_agent_set = agent_set.copy()
            agent_set.update(imit_agent_mask.tolist())
            
            if len(agent_set - pre_agent_set) > 0:
                new_agent_mask = list(agent_set - pre_agent_set)
                env_imit.add_pedestrian(env_real.position[new_agent_mask, t, :], env_real.velocity[new_agent_mask, t, :], env_real.destination[new_agent_mask, :], init=False)
                if ARGS.MODEL in ['SFM', 'ORCA']:
                    model.add_agent(env_real.position[new_agent_mask, t, :], env_real.destination[new_agent_mask, :])

        if t < T - 1:
            mask = env_imit.mask[:, -1] & ~env_imit.arrive_flag[:, -1] 
            if ARGS.MODEL in ['TECRL', 'MAGAIL']:
                action = torch.full((env_imit.num_pedestrians, 2), float('nan'), device=env_imit.device)
                if mask.any():
                    action[mask, :], _ = model(pack_state(*env_imit.get_state())[mask])
                env_imit.action(action[:, 0], action[:, 1], enable_nan_action=True)
            elif ARGS.MODEL in ['SFM', 'ORCA']:
                if mask.any():  
                    model()

    ## Metrics ##
    metrics['Speed'] = calc_speed(env_real, env_imit)
    # metrics['Density'] = calc_density(env_real, env_imit)
    metrics['Displacement'] = calc_displacement(env_real, env_imit)
    metrics['Deviation'] = calc_deviation(env_real, env_imit)
    metrics['Velocity'] = calc_v_locomotion(env_real, env_imit)
    metrics['Angle'] = calc_a_locomotion(env_real, env_imit)
    metrics['Energy'] = calc_energy(env_real, env_imit)
    metrics['Steer'] = calc_steer_energy(env_real, env_imit)
    metrics['SSIM'] = calc_SSIM(env_real, env_imit)
    metrics = np.around(np.array(list(metrics.values())), decimals=4)
    print(metrics)
    
    # df = pd.DataFrame(metrics.reshape(-1, 1).T)
    # df.columns = metrics_keys
    # df.to_csv(f'./result/{ARGS.DATASET}/data/{ARGS.MODEL}_data.csv', index=True, header=True)

    ## plot gif ##
    generate_gif(env_real, save_path=f'./result/{ARGS.DATASET}/gif/real.gif', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])
    generate_gif(env_imit, save_path=f'./result/{ARGS.DATASET}/gif/imit_{ARGS.MODEL}.gif', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])

    ## plot trajectory ##
    plot_traj(env_real, save_path=f'./result/{ARGS.DATASET}/trajectory/real.png', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])
    plot_traj(env_imit, save_path=f'./result/{ARGS.DATASET}/trajectory/imit_{ARGS.MODEL}.png', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])

    ## plot heatmap ##
    plot_heatmap(env_real, save_path=f'./result/{ARGS.DATASET}/heatmap/real_heatmap.png', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])
    plot_heatmap(env_imit, save_path=f'./result/{ARGS.DATASET}/heatmap/imit_{ARGS.MODEL}_heatmap.png', xrange=xrange[ARGS.DATASET], yrange=yrange[ARGS.DATASET])
