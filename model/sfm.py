import torch
import numpy as np
from model import pysocialforce as psf
from model.pysocialforce.scene import PedState

class SFM(torch.nn.Module):
  def __init__(self, env, ARGS):
    super(SFM, self).__init__()
    self.env = env
    self.ARGS = ARGS
    self.N = self.env.num_pedestrians
    self.M = self.env.num_obstacles
    self.initial_state = self.init_state()  # (N, 6)
    self.obstacle = self.init_obstacle()  # (M * 4, 4)
    self.simulator = self.init_simulator()

  def add_agent(self, position, destination):
    n = position.shape[0]
    self.N += n

    # update state
    initial_state = torch.zeros((self.N, 6))
    initial_state[:self.N-n, 0:6] = torch.tensor(self.simulator.get_states()[0][-1, :, 0:6])
    initial_state[-n:, 0:2] = position
    initial_state[-n:, 4:6] = destination

    self.simulator.peds.state = initial_state


    # for agent_id in range(self.N):
    #   p = initial_state[:, 0:2][agent_id]
    #   g =  initial_state[:, 4:6][agent_id]
    #   initial_state[agent_id, 2:4] = (g - p) / torch.norm(g - p)
    # self.initial_state = initial_state
    
  def init_state(self):
    # (px, py, vx, vy, gx, gy)
    initial_state = torch.zeros((self.N, 6))
    initial_state[:, 0:2] = self.env.position[:, -1, :]
    initial_state[:, 4:6] = self.env.destination

    for agent_id in range(self.N):
      p = self.env.position[agent_id, -1, :]
      g = self.env.destination[agent_id, :]
      initial_state[agent_id, 2:4] = (g - p) / torch.norm(g - p)

    return initial_state
  
  def init_obstacle(self):
    if self.M == 0:
      return None
    
    # (startx, endx, starty, endy)
    obstacle = []
    half_side = self.env.obstacle_radius * np.sqrt(2) / 2

    for obs_id in range(self.M):
      x, y = self.env.obstacle[obs_id].tolist()

      top_left_vertex = (x - half_side, y + half_side)
      top_right_vertex = (x + half_side, y + half_side)
      bottom_left_vertex = (x - half_side, y - half_side)
      bottom_right_vertex = (x + half_side, y - half_side)

      top_side = [coord for pair in zip(top_left_vertex, top_right_vertex) for coord in pair]
      bottom_side = [coord for pair in zip(bottom_left_vertex, bottom_right_vertex) for coord in pair]
      left_side = [coord for pair in zip(top_left_vertex, bottom_left_vertex) for coord in pair]
      right_side = [coord for pair in zip(top_right_vertex, bottom_right_vertex) for coord in pair]

      obstacle.extend([top_side, bottom_side, left_side, right_side])

    return obstacle

  def init_simulator(self):
    simulator = psf.Simulator(self.initial_state,
                              groups=None,
                              obstacles=self.obstacle,
                              config_file='./configs/sfm.toml')
    
    return simulator
  
  def forward(self, index=-1, rl_mask=None, ppo_step=None):
    # (x, y, v_x, v_y, d_x, d_y, [tau])
    self.simulator.step_once()
    state = torch.tensor(self.simulator.get_states()[0], dtype=torch.float32).permute(1, 0, 2).to(self.ARGS.DEVICE)  # (N, T, 7)
    position_ = state[:, -1, 0:2] # (N, 2)
    velocity_ = state[:, -1, 2:4] # (N, 2)

    arrive_flag_ = torch.norm(position_ - self.env.destination, dim=-1) < self.env.ped_radius
    mask_ = ~arrive_flag_
    
    position_[~mask_] = float('nan')
    velocity_[~mask_] = float('nan')
    direction_ = torch.atan2(velocity_[:, 1], velocity_[:, 0]).unsqueeze(1)  # (N, 1)
    
    if rl_mask is not None:
      position_[rl_mask] = ppo_step[0][rl_mask]
      velocity_[rl_mask] = ppo_step[1][rl_mask]
      arrive_flag_[rl_mask] = ppo_step[2][rl_mask]
      mask_[rl_mask] = ppo_step[3][rl_mask].view(-1)
      direction_[rl_mask] = ppo_step[4] [rl_mask].view(-1, 1)
      
      state_ = torch.tensor(self.simulator.get_states()[0][-1, :, 0:6]).to(self.ARGS.DEVICE)
      state_[rl_mask, 0:2] = position_[rl_mask].type(torch.float64)
      state_[rl_mask, 2:4] = velocity_[rl_mask].type(torch.float64)
      self.simulator.peds.state = state_.cpu()

    self.env.position = torch.cat([self.env.position, position_.unsqueeze(1)], dim=1) # (N, T, 2)
    self.env.velocity = torch.cat([self.env.velocity, velocity_.unsqueeze(1)], dim=1) # (N, T, 2)
    self.env.arrive_flag = torch.cat([self.env.arrive_flag, arrive_flag_.unsqueeze(1)], dim=1)  # (N,T)
    self.env.mask = torch.cat([self.env.mask, mask_.unsqueeze(1)], dim=1) # (N, T)
    self.env.direction = torch.cat([self.env.direction, direction_.unsqueeze(1)], dim=1)  # (N, T)
    
    self.env.num_steps += 1

    if rl_mask is not None:
      return position_[~rl_mask], velocity_[~rl_mask],  arrive_flag_[~rl_mask], mask_[~rl_mask], direction_[~rl_mask]
    else:
      return None