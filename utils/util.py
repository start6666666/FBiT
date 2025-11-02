import numpy as np
import torch
import random
from torch_geometric.data import Data
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import collections
import supply_chain_env
import math

def prepare_training_env(env_id: str, env_mode: int, seed: int = 2024,
                         use_norm: bool = False, n_envs: int = 1, result_path=None):
    
    model_env = make_vec_env(env_id, n_envs=n_envs, seed=seed,
                             env_kwargs={"env_mode": env_mode},
                             vec_env_cls=DummyVecEnv, monitor_dir=result_path)
    if use_norm:
        model_env = VecNormalize(model_env, norm_obs=True, norm_reward=False, clip_obs=20)

    return model_env

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe)  

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)

class ReplayBuffer():
    def __init__(self, buffer_limit, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.stack(s_lst).to(self.device), torch.tensor(np.array(a_lst), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device), torch.stack(s_prime_lst).to(self.device), \
            torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)

    def size(self):
        return len(self.buffer)
    
class EpisodeReplayBuffer():
    def __init__(self, buffer_limit, epi_length, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.current_episode = []
        self.device = device
        self.epi_length = epi_length

    def put(self, transition):
        self.current_episode.append(transition)
        if len(self.current_episode) >= self.epi_length:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, n):
        if len(self.buffer) < n:
            return None

        sampled_episodes = random.sample(self.buffer, n)

        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for episode in sampled_episodes:
            for transition in episode:
                s, a, r, s_prime, done = transition
                s_list.append(s)
                a_list.append(torch.tensor(a, dtype=torch.float))
                r_list.append(torch.tensor([r], dtype=torch.float))
                s_prime_list.append(s_prime)
                done_mask = 0.0 if done else 1.0
                done_mask_list.append(torch.tensor([done_mask], dtype=torch.float))

        s_batch = torch.stack(s_list).to(self.device)
        a_batch = torch.stack(a_list).to(self.device)
        r_batch = torch.stack(r_list).to(self.device)
        s_prime_batch = torch.stack(s_prime_list).to(self.device)
        done_mask_batch = torch.stack(done_mask_list).to(self.device)

        return s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch

    def size(self):
        return len(self.buffer)

class EpisodeReplayBuffer_G():
    def __init__(self, buffer_limit, epi_length, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.epi_length = epi_length
        self.current_episode = []
        self.device = device

    def put(self, transition):
        self.current_episode.append(transition)
        if len(self.current_episode) >= self.epi_length:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, n):
        if len(self.buffer) < n:
            return None

        sampled_episodes = random.sample(self.buffer, n)

        s_list, a_list, r_list, s_prime_list, done_mask_list, next_demand_list, next_next_demand_list = [], [], [], [], [], [], []

        for episode in sampled_episodes:
            for transition in episode:
                s, a, r, s_prime, done, next_demand, next_next_demand = transition
                s_list.append(s)
                a_list.append(torch.tensor(a, dtype=torch.float))
                r_list.append(torch.tensor([r], dtype=torch.float))
                s_prime_list.append(s_prime)
                done_mask = 0.0 if done else 1.0
                done_mask_list.append(torch.tensor([done_mask], dtype=torch.float))
                next_demand_list.append(next_demand)
                next_next_demand_list.append(next_next_demand)

        a_batch = torch.stack(a_list).to(self.device)
        r_batch = torch.stack(r_list).to(self.device)

        done_mask_batch = torch.stack(done_mask_list).to(self.device)
        next_demand_batch = torch.stack(next_demand_list).to(self.device)
        next_next_demand_batch = torch.stack(next_next_demand_list).to(self.device)
        return s_list, a_batch, r_batch, s_prime_list, done_mask_batch, next_demand_batch, next_next_demand_batch

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()  
        self.current_episode.clear()  

class EpisodeReplayBuffer_G_BIFN():
    def __init__(self, buffer_limit, epi_length, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.current_episode = []
        self.device = device
        self.epi_length = epi_length

    def put(self, transition):
        self.current_episode.append(transition)
        if len(self.current_episode) >= self.epi_length:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, n):
        if len(self.buffer) < n:
            return None

        sampled_episodes = random.sample(self.buffer, n)

        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for episode in sampled_episodes:
            for transition in episode:
                s, a, r, s_prime, done = transition
                s_list.append(s)
                a_list.append(torch.tensor(a, dtype=torch.float))
                r_list.append(torch.tensor([r], dtype=torch.float))
                s_prime_list.append(s_prime)
                done_mask = 0.0 if done else 1.0
                done_mask_list.append(torch.tensor([done_mask], dtype=torch.float))

        a_batch = torch.stack(a_list).to(self.device)
        r_batch = torch.stack(r_list).to(self.device)
        done_mask_batch = torch.stack(done_mask_list).to(self.device)

        return s_list, a_batch, r_batch, s_prime_list, done_mask_batch

    def size(self):
        return len(self.buffer)



class g_env0_utils:
    def __init__(self, env, features_num, device):
        n_stores = env.get_attr("n_retailers")[0]
        n_warehouse = env.get_attr("n_warehouses")[0]
        n_factory = env.get_attr("n_producers")[0]
        n_node = env.get_attr("n_nodes")[0]
        features_num = features_num
        n_edge = env.get_attr("n_edges")[0]
        n_goods = 1
        self.edge_index = env.get_attr("edge_index")[0]
        self.n_edge = n_edge + n_factory
        self.features_num = features_num
        self.n_factory = n_factory
        self.n_goods = n_goods
        self.n_node = n_node
        self.n_stores = n_stores
        self.n_warehouse = n_warehouse
        self.device = device

    def generate_graphs_store_lack_demand(self, s):
        M, W, R = self.n_factory, self.n_warehouse, self.n_stores

        splits = np.split(s, [R, 2*R, 3*R, 3*R+M])
        demand_old1, lack_demand, manu_stocks, wh_stocks, reta_stocks = splits
        
        stocks = np.hstack([manu_stocks, wh_stocks, reta_stocks,
                            np.zeros(M + W), lack_demand,
                            np.zeros(M + W), demand_old1,
                            ])
        x = torch.from_numpy(stocks).float().unsqueeze(-1).view(-1, W+M+R).t().to(self.device)
        u_np = np.hstack([demand_old1])
        graph_attr = torch.from_numpy(u_np).float().to(self.device)
        edge_index = torch.from_numpy(self.edge_index).to(self.device)  
        reverse_edge_index = edge_index.clone()
        reverse_edge_index[[0, 1]] = reverse_edge_index[[1, 0]]  
        return Data(x=x, edge_index=edge_index, reverse_edge_index=reverse_edge_index, graph_attr=graph_attr, num_nodes=self.n_node)

    def generate_state(self, s):
        M, W, R = self.n_factory, self.n_warehouse, self.n_stores
        splits = np.split(s, [R, 2*R, 3*R, 3*R+M])
        demand_old1, lack_demand, manu_stocks, wh_stocks, reta_stocks = splits
        
        stocks = np.hstack([manu_stocks, wh_stocks, reta_stocks,
                            np.zeros(M + W), lack_demand,
                            np.zeros(M + W), demand_old1,
                            ])
        x = torch.from_numpy(stocks).float().unsqueeze(-1).to(self.device)

        return x.view(1, -1)

    def return_info(self):

        return self.features_num, self.n_factory, self.n_goods, self.n_node, self.n_stores, self.n_edge, self.n_warehouse
