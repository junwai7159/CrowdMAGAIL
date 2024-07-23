import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from utils.action import xy2ra, ra2xy

class MLP(nn.Module):
    def __init__(self, shape: tuple, act=nn.Tanh(), act_out=nn.Identity(), gain_out=1.0):
        super(MLP, self).__init__()
        s = []
        for i, j in zip(shape[:-1], shape[1:]):
            linear_layer = nn.Linear(i, j)
            nn.init.orthogonal_(linear_layer.weight)
            nn.init.constant_(linear_layer.bias, 0)
            s.append(linear_layer)
            s.append(act())
        #     s.append(nn.Dropout(0.2))
        # nn.init.orthogonal_(s[-3].weight, gain=gain_out)
        nn.init.orthogonal_(s[-2].weight, gain=gain_out)
        s[-1] = act_out()
        self.seq = nn.Sequential(*s)

    def forward(self, feature):
        return self.seq(feature)


class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        将形如 [batch_size, obstacle_num, feature_num] 的装量加权得到 [batch_size, feature_num] 的张量, 权值根据 feature_num 维
        度决定. 
        【已弃用】若某个 (batch_size, obstacle_num) 对应的 feature 含有 NaN, 则对应权值为 0
        【代替】 若 mask (形如 [batch_size, obstacle_num, 1]) 给出 False, 则对应权值为 0
        """
        super(Attention, self).__init__()
        self.seq = MLP(*args, **kwargs)

    def forward(self, s, mask):
        shape = s.shape
        log_weight = self.seq(s)  # (N, 20, 128) -> (N, 20, 1)
        log_weight_ = log_weight.masked_fill(~mask, -float('inf'))
        weight = torch.softmax(log_weight_, dim=-2)  # N, 20, 1
        # 若 mask 中全是 False (即一个东西也没有) , 则 softmax 将得到一组 nan, 这里将 nan 全部设置为 0.
        weight_ = weight.masked_fill(~mask, 0)  # N, 20, 1
        s_ = torch.einsum('nof,no->nf', s, weight_.squeeze(-1)) # N, 128
        return s_

    def get_weight(self, s):
        return self.seq(s)

class Actor(torch.nn.Module):
    def __init__(self, h_seq=(128, 128, 128, 128), gmm_num=3):
        super(Actor, self).__init__()
        self.gmm_num = gmm_num
        self.seq = MLP((32 + 128, *h_seq, 6 * gmm_num), act=nn.Tanh, act_out=torch.nn.Identity, gain_out=0.1)

    def forward(self, s, action=None, explore=True):
        N = s.shape[0]
        K = self.gmm_num
        h = self.seq(s)  # .view(N, 6 * K)
        p = torch.softmax(h[..., :K], dim=-1) + 1e-4  # (N, K), mixture weights
        m = h[..., K:3*K].view(-1, K, 2)  # (N, K, 2), mean 
        s = torch.zeros([N, K, 2, 2], device=s.device)  # (N, K, 2, 2), covariance matrix
        s[:, :, 0, 0] = explore * F.softplus(h[:, 3*K:4*K]) + 1e-4
        s[:, :, 1, 1] = explore * F.softplus(h[:, 4*K:5*K]) + 1e-4
        s[:, :, 1, 0] = explore * h[:, 5*K:6*K]
        dist_1 = Categorical(p)  # sample -> (N,)
        dist_2 = MultivariateNormal(m, scale_tril=s)  # sample -> (N, K, 2)
        dist = MixtureSameFamily(dist_1, dist_2)  # sample -> (N, 2)
        if action is None:
            xy = dist.sample()  # (N, 2)
            action = xy2ra(xy)  # (N, 2)
            action_logprob = dist.log_prob(xy).view(-1, 1)  # (N, 1)
            return action, action_logprob
        else:  # 若输入中给定动作, 则可以给出采样的熵和采样得到此动作的概率值
            entropy = dist_1.entropy() + torch.sum(p * dist_2.entropy(), dim=-1)
            action_logprob = dist.log_prob(ra2xy(action)).view(-1, 1)
            return entropy, action_logprob


class Critic(torch.nn.Module):
    def __init__(self, h_seq=(128, 128, 128, 128)):
        super(Critic, self).__init__()
        self.seq = MLP((32 + 128, *h_seq, 1), act=nn.Tanh, act_out=torch.nn.Identity, gain_out=0.1)

    def forward(self, s):
        v = self.seq(s)
        return v


class Discriminator(torch.nn.Module):
    def __init__(self, h_seq=(128, 128), args=None):
        super(Discriminator, self).__init__()
        # input = state_dim + action_dim
        self.seq = MLP((169+ 2, *h_seq, 1), act=nn.Tanh, act_out=torch.nn.Identity, gain_out=0.1)
        self.args = args
    
    def forward(self, s, action):
        sa = torch.cat([s, action], dim=-1)
        sa += torch.normal(0, 0.1, size=sa.shape, device=self.args.DEVICE)
        return self.seq(sa)
    
    def calc_gradient_penalty(self, gen_batch_state, gen_batch_action, expert_batch_state, expert_batch_action):
        grad_collect_func = lambda d: torch.cat([grad.view(-1) for grad in torch.autograd.grad(d, self.parameters(), retain_graph=True)]).unsqueeze(0)
        
        differences_batch_state = (gen_batch_state[:expert_batch_state.size(0)]- expert_batch_state).to(self.args.DEVICE)
        differences_batch_action = (gen_batch_action[:expert_batch_action.size(0)] - expert_batch_action).to(self.args.DEVICE)
        alpha = torch.rand(expert_batch_state.size(0), 1).to(self.args.DEVICE)
        interpolates_batch_state = gen_batch_state[:expert_batch_state.size(0)] + (alpha * differences_batch_state)
        interpolates_batch_action = gen_batch_action[:expert_batch_action.size(0)] + (alpha * differences_batch_action)
        gradients = torch.cat([x for x in map(grad_collect_func, self.forward(interpolates_batch_state, interpolates_batch_action))])
        slopes = torch.norm(gradients, p=2, dim=-1)
        gradient_penalty = torch.mean((slopes - 1.) ** 2)

        return gradient_penalty