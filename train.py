"""
This sciprt is used for train the model.
- Execute `python train.py`, the model will be saved in `./checkpoint/testproj/model_final.bin`
- See the function `get_args()` in `utils/utils.py` for default parameters.
"""
import os
import copy
import torch
import json
import logging
import random
import numpy as np

from model.ppo import PPO
from model.sfm import SFM
from envs.pedsim import Pedsim
from configs.config import get_args, set_seed
from utils.state import pack_state
from utils.envs import init_env, init_ped_obs_random
from utils.visualization_cv import generate_gif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
from utils.preprocess import RawData

if __name__ == '__main__':
    # initialization
    ARGS = get_args()
    set_seed(ARGS.SEED)

    # include dynamic agents
    if ARGS.DYNAMIC:
        rl_agent_mask = np.ones(ARGS.NUM_PED, dtype=bool)
        rl_agent_mask[:ARGS.NUM_PED // 2] = False
        np.random.shuffle(rl_agent_mask)
    else:
        rl_agent_mask = np.ones(ARGS.NUM_PED, dtype=bool)
    # fixed settings
    positions, destinations, obstacles = init_ped_obs_random(ARGS)

    # load model
    model = PPO(ARGS, rl_agent_mask).to(ARGS.DEVICE)
    print(model)
        
    # load expert dataset
    if ARGS.MODEL == 'MAGAIL':
        data = RawData(f'./configs/data_configs/pretrain_{ARGS.DATASET}.yaml')
        expert_dataset_loader = data.load_dataset()
    else: 
        expert_dataset_loader=None

    # load and log model
    if ARGS.LOAD_MODEL is not None:
        if ARGS.MODEL == 'MAGAIL':
            pretrained_dict = torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE))
            filtered_dict = {k: v for k, v in pretrained_dict.items() if 'd.' not in k}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            state_dict = torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE))
            model.load_state_dict(state_dict, strict=False)
    save_path = os.path.join(ARGS.SAVE_DIRECTORY, ARGS.UUID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with(open(os.path.join(save_path, f'args.log'), 'w')) as f:
        json.dump(ARGS.__dict__, f, indent=2)
    with(open(os.path.join(save_path, f'rewards.log'), 'w')) as f:
        f.write('REWARD ARRIVE ENERGY WORK COLL MENTAL SMOOTH_V SMOOTH_W\n')
    logging.getLogger().setLevel(logging.INFO)

    # env for demonstration
    demo_env = None

    # train for N episodes
    for episode in range(1, ARGS.MAX_EPISODES + 1):
        # reset environment
        env = Pedsim(ARGS, rl_agent_mask, model)
        init_env(env, ARGS, pos=positions, des=destinations, obs=obstacles)
        dynamic_env = Pedsim(ARGS, rl_agent_mask)
        dynamic_env.add_pedestrian(env.position[:, 0, :], env.velocity[:, 0, :], env.destination[:, :], init=True)
        dynamic_env.add_obstacle(env.obstacle)
        dynamic_model  = SFM(dynamic_env, ARGS)
        env.dynamic_model = dynamic_model

        # run & train
        total_reward, arrive_num, total_detail_reward = model.run_episode(env, episode, train=True, expert=expert_dataset_loader)
        # model.visualize()

        # log
        logging.info(f'[Epi{episode}] #Arrive: {arrive_num}, Reward: {total_reward:7.2f} [ {", ".join([f"{total_detail_reward[r]:.1f}({r})" for r in total_detail_reward])} ]')
        open(os.path.join(save_path, f'rewards.log'), 'a').write(f'{str(total_reward)} {" ".join([str(r) for r in total_detail_reward.values()])}\n')
        
        # save (at epoch 100, 200, ..., 900, 1000, 2000, ..., 9000, ...)
        if episode >= 100 and str(episode)[1:] == '0' * len(str(episode)[1:]):
            if episode == 100:
                demo_env = env
            # save model
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{episode}.bin'))
            
            # save GIF visualization
            for t in range(200):
                mask = demo_env.mask[rl_agent_mask, -1] & ~demo_env.arrive_flag[rl_agent_mask, -1]
                if not mask.any(): break
                action = torch.full((rl_agent_mask.sum(), 2), float('nan'), device=demo_env.device)
                s_self, s_int, s_ext = demo_env.get_state()
                s_self, s_int, s_ext = s_self[rl_agent_mask], s_int[rl_agent_mask], s_ext[rl_agent_mask]
                action[mask, :], _ = model(pack_state(s_self, s_int, s_ext)[mask], explore=True)
                _,  run = demo_env.action(action[:, 0], action[:, 1], enable_nan_action=True)
            generate_gif(demo_env, os.path.join(save_path, f'demo_{episode}.gif'), rl_agent_mask=rl_agent_mask)
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_final.bin'))

        if episode % 10000 == 0:
            model.ARGS.ENTROPY /= 10.0
    
    model.writer.close()
