import yaml
from collections import defaultdict
import torch
from tqdm import tqdm
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import numpy as np

from envs.pedsim import Pedsim
from configs.config import get_args, set_seed
from utils.state import pack_state

def trajectories_split(trajectories):
    split_trajectories = []
    for traj in trajectories:
        tensor_traj = torch.tensor(traj)
        if(torch.all(torch.diff(tensor_traj[:, 2]) == 1)):
            split_trajectories.append(traj)
        else:
            left = 0
            for right in range(1, tensor_traj.shape[0]):
                if(tensor_traj[right, 2] - tensor_traj[right - 1, 2] > 1):
                    split_trajectories.append(traj[left:right])
                    left = right
            split_trajectories.append(traj[left:right])

    return split_trajectories

class ExpertDataset(Dataset):
    def __init__(self, state, action):
        self.state = state.view(-1, 169)
        self.action = action.view(-1, 2)
    
    def __len__(self):
        return self.state.size(0)
    
    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]

class RawData(object):
    def __init__(self, data_config_path):
        self.ARGS = get_args()
        self.data_config_path = data_config_path
        self.raw_data = self.load_raw_data()  # {train: {N, T, postion, velocity, destination, obstacle}, valid, test}
        self.expert_state = list()
        self.expert_action = list()
    
    def load_trajectory_data(self, data_path):
        meta_data, trajectories, des, obs = np.load(data_path, allow_pickle=True)
        N = len(trajectories)
        T = np.max([t for traj in trajectories for x, y, t in traj]) + 1
        position = torch.full((N, T, 2), float('nan'), device=self.ARGS.DEVICE)  # (N, T, 2)
        for p, traj in enumerate(trajectories):
            for x, y, t in traj:
                position[p, t, 0] = x
                position[p, t, 1] = y
        nan_flag = position.isnan().any(dim=-1)  # (N, T)
        into_env = (~nan_flag) & (nan_flag.roll(shifts=1, dims=1))  # (N, T)
        exit_env = (nan_flag) & (~nan_flag.roll(shifts=1, dims=1))  # (N, T)
        into_env[nan_flag.logical_not().all(dim=-1), 0] = True
        exit_env[nan_flag.logical_not().all(dim=-1), 0] = True
        assert (into_env.sum(dim=1) == 1).all() and (exit_env.sum(dim=1) == 1).all(), "A pedestrian enter the env for more/less than 1 times!"
        time = torch.arange(T, device=self.ARGS.DEVICE)
        into_time = torch.masked_select(time, into_env)  # (N,)
        exit_time = torch.masked_select(time, exit_env)  # (N,)
        exit_time[exit_time == 0] = T

        velocity = position.diff(dim=1, prepend=position[:, (0,), :]) / meta_data['time_unit']  # (N, T, 2)
        velocity[:, into_time, :] = velocity.roll(shifts=-1, dims=1)[:, into_time, :]
        one_step_flag = (into_time + 1 == exit_time)
        velocity[one_step_flag, into_time[one_step_flag], :] = 0.

        destination = torch.FloatTensor(des)[:, 0, :2]  # (N, 2)
        obstacle = torch.FloatTensor(obs)   # (M, 2)

        env = Pedsim(self.ARGS)
        env.init_ped(position, velocity, destination)
        env.add_obstacle(obstacle)

        return {'env':env, 'num_pedestrians': N, 'num_steps': T, 'data_path': data_path}

    def load_raw_data(self):
        with open(self.data_config_path, 'r') as stream:
            data_paths = yaml.load(stream, Loader=yaml.FullLoader)

        data = defaultdict(list)
        for key in data_paths.keys():
            for path in data_paths[key]:
                trajectory_data = self.load_trajectory_data(path)
                data[key].append(trajectory_data)
        
        return data        

    def load_dataset(self):
        for train_data in self.raw_data['train']:
            print(f"Loading from '{train_data['data_path']}'...")
            env_real = train_data['env']
            N = train_data['num_pedestrians']
            T = train_data['num_steps']

            # load from env_real to expert_dataset  
            expert_state = torch.zeros((N, T, 169))
            expert_action = torch.zeros((N, T, 2))
            for i in tqdm(range(T - 1)):
                # State
                s_self, s_int, s_ext = env_real.get_state(index=i)
                state_t = pack_state(s_self, s_int, s_ext)  # (N, 169)
                state_t[torch.isnan(state_t)] = 0.0
                expert_state[:, i, :] = state_t

                # Action
                w0 = env_real.direction[:, i, 0]
                w_w0 = env_real.direction[:, i + 1, 0]
                v = torch.norm(env_real.velocity[:, i + 1, :], dim=-1).view(-1, 1)
                v[torch.isnan(v)] = 0.0
                w =(w_w0 - w0).view(-1, 1)
                w[torch.isnan(w)] = 0.0
                expert_action[:, i, :] = torch.cat((v, w), dim=-1)
            
            self.expert_state.append(expert_state)
            self.expert_action.append(expert_action)
        
        self.expert_state = torch.cat(self.expert_state, dim=0)
        self.expert_action = torch.cat(self.expert_action, dim=0)
        expert_dataset = ExpertDataset(state=self.expert_state, action=self.expert_action)
        expert_dataset_loader = DataLoader(dataset=expert_dataset,
                                        batch_size=self.ARGS.BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=multiprocessing.cpu_count() // 2,
                                        drop_last=True)
        return expert_dataset_loader

    def data_augmentation(self):
        pass


