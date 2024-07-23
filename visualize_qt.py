"""
This script is used for visualization
- Rely on PyQt5 and pyqtgraph
- execute `python visualize.py --LOAD_MODEL <MODEL>` to visualize the performance of MODEL
    - For example, execute `python visualize.py --LOAD_MODEL ./checkpoint/demonstration/model_final.bin`
"""
import torch
import numpy as np
import pandas as pd

from envs.pedsim import Pedsim
from model.ppo import PPO
from model.sfm import SFM
from model.orca import ORCA

from configs.config import get_args, set_seed
from utils.visualization_qt import Visualization
from utils.visualization_cv import *
from utils.metrics import *
from utils.envs import init_env
from utils.state import pack_state

if __name__ == '__main__':
    ARGS = get_args()
    set_seed(ARGS.SEED)

    env = Pedsim(ARGS)
    _, _, _, spawn_collision = init_env(env, ARGS)
    if ARGS.MODEL in ['TECRL', 'MAGAIL']:
        model = PPO(ARGS).to(ARGS.DEVICE)
        if ARGS.LOAD_MODEL is not None:
            model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))
        env.ppo = model
    else:
        if ARGS.MODEL == 'ORCA':
            model = ORCA(env, ARGS)
        elif ARGS.MODEL == 'SFM':
            model = SFM(env, ARGS)
        model()
    
    qt = False
    if qt:
        def update():
            with torch.no_grad():
                mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
                if not mask.any():
                    return False
                if ARGS.MODEL in ['TECRL', 'MAGAIL']:
                    action = torch.full((env.num_pedestrians, 2), float('nan'), device=env.device)
                    state = pack_state(*env.get_state())
                    action[mask, :], _ = model(state[mask], explore=True)
                    env.action(action[:, 0], action[:, 1], enable_nan_action=True)
                elif ARGS.MODEL in ['SFM', 'ORCA']:
                    model()
            return True
        env.update = update
        Visualization(env, model=model).play()
    
    else:
        while True:
            with torch.no_grad():
                mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
                if not mask.any():
                    break
                if ARGS.MODEL in ['TECRL', 'MAGAIL']:
                    action = torch.full((env.num_pedestrians, 2), float('nan'), device=env.device)
                    state = pack_state(*env.get_state())
                    action[mask, :], _ = model(state[mask], explore=True)
                    env.action(action[:, 0], action[:, 1], enable_nan_action=True)
                elif ARGS.MODEL in ['SFM', 'ORCA']:
                    model()

    
    # save metrics
    metrics_keys = ['Time', 'Collision']#, 'Speed', 'Displacement', 'Deviation', 'Velocity', 'Angle', 'Energy', 'Steer']
    metrics = {metric_key: None for metric_key in metrics_keys}
    metrics['Time'] = env.num_steps
    metrics['Collision'] = calc_collision(env) - spawn_collision
    # metrics['Speed'] = calc_speed(env_imit=env)
    # metrics['Displacement'] = calc_displacement(env_imit=env)
    # metrics['Deviation'] = calc_deviation(env_imit=env)
    # metrics['Velocity'] = calc_v_locomotion(env_imit=env)
    # metrics['Angle'] = calc_a_locomotion(env_imit=env)
    # metrics['Energy'] = calc_energy(env_imit=env)
    # metrics['Steer'] = calc_steer_energy(env_imit=env)
    metrics = np.around(np.array(list(metrics.values())), decimals=4)
    print(metrics)
    df = pd.DataFrame(metrics.reshape(-1, 1).T)
    df.columns = metrics_keys
    df.to_csv(f'./result/{ARGS.SCENARIO.lower().capitalize()}/data/{ARGS.MODEL}_data.csv', index=True, header=True)

    # plot results
    generate_gif(env, save_path=f'./result/{ARGS.SCENARIO.lower().capitalize()}/gif/{ARGS.MODEL.lower()}.gif', xrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV), yrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV))
    plot_traj(env, save_path=f'./result/{ARGS.SCENARIO.lower().capitalize()}/trajectory/{ARGS.MODEL.lower()}.png', xrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV), yrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV))
    plot_heatmap(env, save_path=f'./result/{ARGS.SCENARIO.lower().capitalize()}/heatmap/{ARGS.MODEL.lower()}.png', xrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV), yrange=(-ARGS.SIZE_ENV, ARGS.SIZE_ENV))
