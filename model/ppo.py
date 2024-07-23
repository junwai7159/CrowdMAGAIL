import os
import logging

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion

from utils.state import pack_state, unpack_state
from utils.action import  xy2rscnt
from model.networks import MLP, Attention, Actor, Critic, Discriminator


class PPO(torch.nn.Module):
    def __init__(self, ARGS, rl_mask=None):
        super(PPO, self).__init__()
        self.ARGS = ARGS
        self.feature = MLP((8, *eval(ARGS.H_FEATURE), 128), act=nn.Tanh, act_out=nn.Identity)
        self.feature2 = MLP((1 + 8, *eval(ARGS.H_FEATURE), 32), act=nn.Tanh, act_out=nn.Identity)
        self.attention = Attention((128, *eval(ARGS.H_ATTENTION), 1), act=nn.Tanh, act_out=nn.Identity)
        self.pi = Actor(h_seq=eval(ARGS.H_SEQ))
        self.v = Critic(h_seq=eval(ARGS.H_SEQ))
        self.memory = dict(s=[], a=[], r=[], v=[], p=[], d=[], f=[])
        self.optimizer = torch.optim.Adam([
            {'params': self.feature.parameters(), 'lr': ARGS.LR_0},
            {'params': self.feature2.parameters(), 'lr': ARGS.LR_0},
            {'params': self.attention.parameters(), 'lr': ARGS.LR_0},
            {'params': self.pi.parameters(), 'lr': ARGS.LR_A},
            {'params': self.v.parameters(), 'lr': ARGS.LR_C} 
        ], eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6400, gamma=0.9, verbose=False)

        if ARGS.MODEL == 'MAGAIL':
            self.d = Discriminator(h_seq=eval(ARGS.H_DISCRIMINATOR), args=ARGS)
            self.optimizer_d = torch.optim.Adam(params=self.d.parameters(), lr=ARGS.LR_D, betas=(0.5, 0.999))
            self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95, verbose=False)

        # TensorBoard
        save_path = os.path.join(ARGS.SAVE_DIRECTORY, ARGS.UUID)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.writer = SummaryWriter(save_path)

        # Episode
        self.episode = 0
        self.time_step = 0
        self.rl_mask = rl_mask   #(N,), mask to select rl agents

    def generate_s(self, state):
        s_self, s_int, s_ext = unpack_state(state)                                              # (N, 1) & (N, 8) & (N, 20, 8)
        s_int_ = torch.cat([s_self, s_int], dim=-1)                                              # (N, 9)
        s_int_merge = self.feature2(s_int_)                                                         # (N, 32)
        mask = torch.any(torch.isnan(s_ext), dim=-1, keepdim=True)    # (N, 20, 1)
        s_ext_ = s_ext.masked_fill(mask, 0.)                                                       # (N, 20, 8)
        s_ext_feat = self.feature(s_ext_).masked_fill(mask, 0.)                   # (N, 20, 128)
        s_ext_merge = self.attention(s_ext_feat, ~mask)                               # (N, 128)
        return torch.cat([s_int_merge, s_ext_merge], dim=-1)                   # (N, 160)

    def forward(self, state, explore=False, mask=None):
        with torch.no_grad():
            if mask is not None:
                action = torch.full((state.shape[0], 2), float('nan'), device=self.ARGS.DEVICE)
                logprob = torch.full((state.shape[0], 1), float('nan'), device=self.ARGS.DEVICE)
                if mask.any():
                    action[mask, :], logprob[mask, :] = self.pi(self.generate_s(state[mask, :]), explore=explore)
            else:
                action, logprob = self.pi(self.generate_s(state), explore=explore)
        return action, logprob

    def get_logprob(self, state, action):
        with torch.no_grad():
            _, logprob = self.pi(self.generate_s(state), action, explore=True)
        return logprob

    def learn(self, env, expert=None):
        ########## update discriminator ##########
        if (self.ARGS.MODEL == 'MAGAIL') and (expert is not None):
            print('Updating Discriminator ...')
            gen_batch_state = torch.stack(self.memory['s'], dim=0).view(-1, 169).to(self.ARGS.DEVICE)  # (memory_capacity, 169)
            gen_batch_state[torch.isnan(gen_batch_state)] = 0.0
            gen_batch_action = torch.stack(self.memory['a'], dim=0).view(-1, 2).to(self.ARGS.DEVICE)    # (memory_capacity, 2)
            gen_batch_action[torch.isnan(gen_batch_action)] = 0.0

            for _ in tqdm(range(self.ARGS.D_EPOCH)):
                expert_batch_state, expert_batch_action = next(iter(expert))    # sample random batch
                expert_batch_state = expert_batch_state.to(self.ARGS.DEVICE)
                expert_batch_action = expert_batch_action.to(self.ARGS.DEVICE)
                
                """Vanilla GAN"""
                # gen_r = torch.sigmoid(self.d(gen_batch_state[torch.randperm(len(gen_batch_state))[:self.ARGS.BATCH_SIZE]], 
                #                              gen_batch_action[torch.randperm(len(gen_batch_action))[:self.ARGS.BATCH_SIZE]]))
                # expert_r = torch.sigmoid(self.d(expert_batch_state, expert_batch_action))

                # # if (gen_r < 0.5).float().mean().item() > 0.8:
                # #     break
       
                # # label smoothing
                # gen_labels = torch.zeros_like(gen_r)
                # expert_labels = torch.ones_like(expert_r)
                # smooting_rate = 0.1
                # expert_labels *= (1- smooting_rate)
                # gen_labels += torch.ones_like(gen_r) * smooting_rate

                # # loss
                # g_loss = F.binary_cross_entropy(gen_r, gen_labels)
                # e_loss = F.binary_cross_entropy(expert_r, expert_labels)
                # d_loss = g_loss + e_loss

                """WGAN with Gradient Penalty"""
                gen_r = self.d(gen_batch_state[torch.randperm(len(gen_batch_state))[:self.ARGS.BATCH_SIZE]], 
                                gen_batch_action[torch.randperm(len(gen_batch_action))[:self.ARGS.BATCH_SIZE]])
                expert_r = self.d(expert_batch_state, expert_batch_action)

                d_loss = gen_r.mean() - expert_r.mean()
                wasserstein_distance = - d_loss
                gradient_penalty = self.d.calc_gradient_penalty(gen_batch_state, gen_batch_action, expert_batch_state, expert_batch_action)
                d_loss += 10 * gradient_penalty

                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()
                self.scheduler_d.step()
            
            # GAIL reward
            # with torch.no_grad():
            #     r_d = - torch.log(torch.clamp(torch.sigmoid(gen_r), 1e-10, 1)).view(-1, 1)

            self.writer.add_scalar('GAN/d_loss', d_loss.item(), self.episode)
            # self.writer.add_scalar('GAN/g_loss', g_loss.item(), self.episode)
            # self.writer.add_scalar('GAN/e_loss', e_loss.item(), self.episode)
            self.writer.add_scalar('GAN/wasserstein_distance', wasserstein_distance.item(), self.episode)
            self.writer.add_scalar('GAN/g_r', torch.sigmoid(gen_r).mean().item(), self.episode)
            self.writer.add_scalar('GAN/e_r', torch.sigmoid(expert_r).mean().item(), self.episode)
            self.writer.add_scalar('GAN/g_acc', (torch.sigmoid(gen_r) < 0.5).float().mean().item(), self.episode)
            self.writer.add_scalar('GAN/e_acc', (torch.sigmoid(expert_r) > 0.5).float().mean().item(), self.episode)


        ########## update PPO (generator) ##########
        print('Updating Generator ...')
        with torch.no_grad():
            # v_: bdr 的最后一个元素是 v[-1], 因此 advantages 的最后一个元素是 v[-1] - v[-1] = 0, 因此要取这个初始值使得第一个算出的 delta 为 0
            # 可以的话其实最好是根据 s[-1] 和 a[-1] 得到下一个状态 s_, 以及对应的 v_ = v(s_), 但是不太好搞, 所以这里就这样了
            v_ = (self.memory['v'][-1] - self.memory['r'][-1]) / self.ARGS.GAMMA
            adv = 0.0
            advantages = []

            for index in reversed(range(len(self.memory['s']))):
                r = self.memory['r'][index] 
                # # Add discriminator reward
                # if self.ARGS.MODEL == 'MAGAIL':
                #     r += self.ARGS.RW_GAIL * r_d[index] 
                d = self.memory['d'][index]
                f = self.memory['f'][index]
                v = self.memory['v'][index]
                delta = (r + self.ARGS.GAMMA * v_ - v)
                v_ = v
                adv = (1 - d) * (delta + (self.ARGS.GAMMA * self.ARGS.LAMBDA) * adv) + d * f * (r - v)
                advantages.insert(0, adv)

        badv = torch.stack(advantages, dim=1).view(-1, 1)  # (N, 1) * T -> (N, T, 1) -> (N * T, 1)
        bv = torch.stack(self.memory['v'], dim=1).view(-1, 1)  # 同上
        bdr = badv + bv
        badv = (badv - badv.mean()) / badv.std().clamp(1e-5)
        bs = torch.stack(self.memory['s'], dim=1).view(-1, 1 + 8 + 20 * 8)  # 同上, 下同
        ba = torch.stack(self.memory['a'], dim=1).view(-1, 2)
        bp = torch.stack(self.memory['p'], dim=1).view(-1, 1)

        pg_loss_array = np.array([])
        vf_loss_array = np.array([])
        en_loss_array = np.array([])

        for _ in tqdm(range(self.ARGS.G_EPOCH)):
            bs_ = self.generate_s(bs)
            entropy, new_logp = self.pi(bs_, ba)
            value = self.v(bs_)

            ratio = torch.exp(new_logp - bp)
            surr1 = ratio * badv
            surr2 = torch.clamp(ratio, 1 - self.ARGS.EPSILON, 1 + self.ARGS.EPSILON) * badv

            pg_loss = torch.min(surr1, surr2).mean()
            vf_loss = F.mse_loss(bdr, value).mean()
            en_loss = entropy.mean()
            pg_loss_array = np.concatenate([pg_loss_array, [pg_loss.item()]])
            vf_loss_array = np.concatenate([vf_loss_array, [vf_loss.item()]])
            en_loss_array = np.concatenate([en_loss_array, [en_loss.item()]])

            loss = -1.0 * pg_loss + 0.5 * vf_loss - self.ARGS.ENTROPY * en_loss
            assert not loss.isinf(), "Loss is Inf!"
            assert not loss.isnan(), "Loss is NaN!"
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
        
        return np.mean(pg_loss_array), np.mean(vf_loss_array), np.mean(en_loss_array)
        
    def run_episode(self, env, episode, train=True, expert=None):
        if env.obstacle is not None and env.obstacle.numel() > 0:
            assert not torch.any(torch.isnan(env.obstacle)), f"obstacle nan! {env.obstacle}"
        total_reward = 0.0
        total_detail_reward = {}
        done = False
        
        pg_loss, vf_loss, en_loss = None, None, None
        total_value = 0.0

        self.episode = episode
        for step in range(self.ARGS.MAX_EP_STEPS):
            self.time_step += 1

            with torch.no_grad():
                s_self, s_int, s_ext = env.get_state()
                s_self, s_int, s_ext = s_self[self.rl_mask], s_int[self.rl_mask], s_ext[self.rl_mask]
                assert not torch.any(torch.isnan(s_int)), f"s_int nan! {s_int}"
                state = pack_state(s_self, s_int, s_ext)
                s = self.generate_s(state)
                action, action_logprob = self.pi(s)
                assert not torch.any(torch.isnan(action)), f"action nan! {action}"
                value = self.v(s)
                assert not torch.any(torch.isnan(value)), f"value nan! {value}"
                total_value += torch.mean(value)

            reward, detail_rewards = env.action(action[:, 0], action[:, 1])
            assert not torch.any(torch.isnan(reward)), f"reward nan! {reward}"

            total_reward += torch.mean(reward).item()
            for key, rwd in detail_rewards.items():
                if key not in total_detail_reward:
                    total_detail_reward[key] = 0.0
                total_detail_reward[key] += torch.mean(rwd['VALUE']).item()
                
            if torch.any(env.arrive_flag[:, -1]):
                done = True
            if train:
                self.memory['s'].append(state)
                self.memory['a'].append(action)
                self.memory['p'].append(action_logprob)
                self.memory['v'].append(value)
                self.memory['r'].append(reward)
                self.memory['f'].append(env.arrive_flag[:, (-1,)][self.rl_mask].float())
                d = (step == self.ARGS.MAX_EP_STEPS - 1) or done
                self.memory['d'].append(torch.full_like(reward, d))
                if len(self.memory['s'])  >= self.ARGS.MEMORY_CAPACITY:
                    pg_loss, vf_loss, en_loss = self.learn(env, expert)
                    for k in self.memory.keys():
                        del self.memory[k][:]
            if done:
                logging.debug('有人到达目的地, 提前结束')
                break


        # TensorBoard: track training
        self.writer.add_scalar('Environment/Cumulative Reward', total_reward, self.episode)
        self.writer.add_scalar('Environment/Energy', total_detail_reward['ENERGY'], self.episode)
        self.writer.add_scalar('Environment/Work', total_detail_reward['WORK'], self.episode)
        self.writer.add_scalar('Environment/Mental', total_detail_reward['MENTAL'], self.episode)
        self.writer.add_scalar('Environment/Smooth_V', total_detail_reward['SMOOTH_V'], self.episode)
        self.writer.add_scalar('Environment/Smooth_W', total_detail_reward['SMOOTH_W'], self.episode)
        self.writer.add_scalar('Environment/Episode Length', step, self.episode)

        if pg_loss:
            self.writer.add_scalar('Losses/Policy Loss', pg_loss, self.episode)
        if vf_loss:
            self.writer.add_scalar('Losses/Value Loss', vf_loss, self.episode)

        if en_loss:
            self.writer.add_scalar('Policy/Entropy', en_loss, self.episode)
        self.writer.add_scalar('Policy/Learning Rate', self.optimizer.param_groups[0]['lr'], self.episode)
        self.writer.add_scalar('Policy/Value Estimate', total_value, self.episode)
        
        return total_reward, int(torch.sum(env.arrive_flag[:, -1]).item()), total_detail_reward

    def visualize(self, smooth=True):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import numpy as np

        def get_xy(xyrange=(-10, 10)):
            x = np.linspace(*xyrange, 100)
            x, y = np.meshgrid(x, x)
            x = torch.from_numpy(np.ravel(x)).float()
            y = torch.from_numpy(np.ravel(y)).float()
            xy = torch.stack([x, y], dim=-1)
            return xy

        def draw(ax, xy, z, title='', des=None, obs=None, **kwargs):
            x = xy[:, 0].reshape(100, 100).cpu().numpy()
            y = xy[:, 1].reshape(100, 100).cpu().numpy()
            z = z.detach().cpu().numpy().reshape(100, 100)
            pm = ax.pcolormesh(x, y, z, **kwargs)
            if des is not None: ax.scatter(*des.T.cpu().numpy(), marker='x', color='r', alpha=0.5)
            if obs is not None: ax.scatter(*obs.T.cpu().numpy(), marker='o', color='b', alpha=0.5)
            ax.grid('on', linestyle='--', alpha=0.5, color='w')
            ax.axis('equal')
            if title:
                ax.set_title(title)
            return pm

        
        xy = get_xy().to(self.ARGS.DEVICE)
        v = 1.0
        v0 = torch.tensor([[v, 0.]], device=self.ARGS.DEVICE)
        p0 = torch.tensor([[0., 0.]], device=self.ARGS.DEVICE)
        p1 = torch.tensor([[5., 1.]], device=self.ARGS.DEVICE)
        p2 = torch.tensor([[1., 2.]], device=self.ARGS.DEVICE)

        s_self = torch.full((10000, 1), v, device=self.ARGS.DEVICE)

        s_int0 = xy2rscnt(pos=p0 - xy, vel=-v0, dir=0)
        s_int1 = xy2rscnt(pos=p1 - xy, vel=-v0, dir=0)
        s_int2 = xy2rscnt(pos=p2 - xy, vel=-v0, dir=0)

        s_ext0 = xy2rscnt(pos=xy - p0, vel=-v0, dir=0)
        s_ext1 = xy2rscnt(pos=xy - p1, vel=-v0, dir=0)
        s_ext2 = xy2rscnt(pos=xy - p2, vel=-v0, dir=0)

        s_ext_v0 = xy2rscnt(pos=p0 - xy, vel=-v0 * 0., dir=0)
        s_ext_v1 = xy2rscnt(pos=p0 - xy, vel=-v0 * .5, dir=0)
        s_ext_v2 = xy2rscnt(pos=p0 - xy, vel=-v0 * 1., dir=0)
        s_ext_v3 = xy2rscnt(pos=p0 - xy, vel=-v0 * 2., dir=0)

        # value
        fig, axes = plt.subplots(2, 2)
        with torch.no_grad():
            v0 = self.v(self.generate_s(pack_state(s_self, s_int0, s_ext1)))
            v1 = self.v(self.generate_s(pack_state(s_self, s_int0, s_ext2)))
            v2 = self.v(self.generate_s(pack_state(s_self, s_int1, s_ext0)))
            v3 = self.v(self.generate_s(pack_state(s_self, s_int2, s_ext0)))
        vmax = torch.max(torch.cat([v0, v1, v2, v3]))
        vmin = torch.min(torch.cat([v0, v1, v2, v3]))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        pm0 = draw(axes[0, 0], xy, v0, title='v', norm=norm, des=p0, obs=p1, cmap=plt.cm.bone)
        pm1 = draw(axes[0, 1], xy, v1, title='v', norm=norm, des=p0, obs=p2, cmap=plt.cm.bone)
        pm2 = draw(axes[1, 0], xy, v2, title='v', norm=norm, des=p1, obs=p0, cmap=plt.cm.bone)
        pm3 = draw(axes[1, 1], xy, v3, title='v', norm=norm, des=p2, obs=p0, cmap=plt.cm.bone)
        fig.colorbar(pm0, ax=np.ravel(axes))
        plt.show()

        # force
        fig, axes = plt.subplots(2, 2)
        with torch.no_grad():
            a0, _ = self.pi(self.generate_s(pack_state(s_self, s_int0, s_ext1)), explore=not smooth)
            a1, _ = self.pi(self.generate_s(pack_state(s_self, s_int0, s_ext2)), explore=not smooth)
            a2, _ = self.pi(self.generate_s(pack_state(s_self, s_int1, s_ext0)), explore=not smooth)
            a3, _ = self.pi(self.generate_s(pack_state(s_self, s_int2, s_ext0)), explore=not smooth)
        halfrange = torch.cat([a0, a1, a2, a3])[:, 0].abs().max()
        norm = colors.CenteredNorm(halfrange=halfrange)
        pm0 = draw(axes[0, 0], xy, a0[:, 0], title='force[0]', norm=norm, des=p0, obs=p1, cmap=plt.cm.PiYG)
        pm1 = draw(axes[0, 1], xy, a1[:, 0], title='force[0]', norm=norm, des=p0, obs=p2, cmap=plt.cm.PiYG)
        pm2 = draw(axes[1, 0], xy, a2[:, 0], title='force[0]', norm=norm, des=p1, obs=p0, cmap=plt.cm.PiYG)
        pm3 = draw(axes[1, 1], xy, a3[:, 0], title='force[0]', norm=norm, des=p2, obs=p0, cmap=plt.cm.PiYG)
        fig.colorbar(pm0, ax=np.ravel(axes))
        plt.show()

        # action w
        fig, axes = plt.subplots(2, 2)
        vmax = torch.max(torch.abs(torch.cat([a0, a1, a2, a3])[:, 1]))
        vmin = -vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        pm0 = draw(axes[0, 0], xy, a0[:, 1], title='action w', norm=norm, des=p0, obs=p1, cmap=plt.cm.seismic)
        pm1 = draw(axes[0, 1], xy, a1[:, 1], title='action w', norm=norm, des=p0, obs=p2, cmap=plt.cm.seismic)
        pm2 = draw(axes[1, 0], xy, a2[:, 1], title='action w', norm=norm, des=p1, obs=p0, cmap=plt.cm.seismic)
        pm3 = draw(axes[1, 1], xy, a3[:, 1], title='action w', norm=norm, des=p2, obs=p0, cmap=plt.cm.seismic)
        fig.colorbar(pm0, ax=np.ravel(axes))
        plt.show()

        # attention
        fig, axes = plt.subplots(2, 2)
        with torch.no_grad():
            weight0 = self.attention.get_weight(self.feature(s_ext_v0.view(10000, 8))).exp()
            weight1 = self.attention.get_weight(self.feature(s_ext_v1.view(10000, 8))).exp()
            weight2 = self.attention.get_weight(self.feature(s_ext_v2.view(10000, 8))).exp()
            weight3 = self.attention.get_weight(self.feature(s_ext_v3.view(10000, 8))).exp()
        vmax = torch.max(torch.cat([weight0, weight1, weight2, weight3]))
        vmin = torch.min(torch.cat([weight0, weight1, weight2, weight3]))
        norm = colors.LogNorm(vmin=1, vmax=vmax - vmin + 1)
        pm0 = draw(axes[0, 0], xy, weight0 - vmin + 1, title='weight(v=0.0)', norm=norm, obs=p0, cmap=plt.cm.hot)
        pm1 = draw(axes[0, 1], xy, weight1 - vmin + 1, title='weight(v=0.5)', norm=norm, obs=p0, cmap=plt.cm.hot)
        pm2 = draw(axes[1, 0], xy, weight2 - vmin + 1, title='weight(v=1.0)', norm=norm, obs=p0, cmap=plt.cm.hot)
        pm3 = draw(axes[1, 1], xy, weight3 - vmin + 1, title='weight(v=1.5)', norm=norm, obs=p0, cmap=plt.cm.hot)
        fig.colorbar(pm0, ax=np.ravel(axes))
        plt.show()
