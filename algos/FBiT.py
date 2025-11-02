import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import torch
import numpy as np
import torch_geometric

seed = 7548

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch_geometric.seed_everything(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
from utils.util import *
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_num, n_factory, n_goods, n_node, n_stores, n_edge = 0, 0, 0, 0, 0, 0
writer = None
result_path = None
epi_length = 52

# Hyperparameters
lr_pi = 0.0005
lr_q = 0.0005
init_alpha = 0.01
gamma = 0.99
batch_size = 5
buffer_limit = 20000
tau = 0.005
target_entropy = -1
lr_alpha = 0.001

class predictorNet(nn.Module):
    def __init__(self, learning_rate):
        super(predictorNet, self).__init__()
        self.d_model = 64
        self.mlp = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=4,
                                                   dim_feedforward=self.d_model * 2,
                                                   batch_first=False,
                                                   norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.pos_enc = PositionalEncoding(self.d_model, max_len=55)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.predict_demand = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )
        self.predict_binary = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=600,  
            gamma=0.7  
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, g):
        b = max(1, int(g.num_graphs / epi_length))
        t = int(g.num_graphs) if b == 1 else 52
        t = max(1, min(t, 53))
        src_mask = torch.triu(torch.ones(t, t), diagonal=1).bool()

        s_Lstm = self.mlp(g.graph_attr.view(b, t, n_stores, -1)).permute(1, 0, 2, 3).reshape(t, b * n_stores, -1)

        s_Lstm = self.pos_enc(s_Lstm)
        s_Lstm = self.transformer(s_Lstm, mask=src_mask.to(device))
        s_Lstm = s_Lstm.reshape(t, b, n_stores, -1).permute(1, 0, 2, 3)
        predict_demand = self.predict_demand(s_Lstm)
        predict_demand[:, :3] = 0

        next_demand = predict_demand

        next_demand = F.pad(next_demand, ((0, 0, n_node - n_stores, 0,)), mode='constant', value=0)

        return next_demand, predict_demand

    def train_net(self, mini_batch):
        g, a, r, g_prime, done_mask, _, _ = mini_batch
        g_batch = Batch.from_data_list(g)
        g_prime_batch = Batch.from_data_list(g_prime)
        _, predict_demand = self.forward(g_batch)
        predict_demand = predict_demand.view(-1, epi_length, n_stores, 1)
        target_demand = g_prime_batch.graph_attr.view(-1, epi_length, n_stores, 1)
        loss = F.smooth_l1_loss(predict_demand[:, 3:], target_demand[:, 3:])
        writer.add_scalar("rollout/predictor_loss", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def ensemble_predict(models, g_batch):
        next_d = []
        with torch.no_grad():
            for m in models:
                m.eval()
                next_demand, pred_demand = m.forward(g_batch)
                next_d.append(next_demand)
        next_d = torch.stack(next_d, dim=0)  
        mean = next_d.mean(dim=0)        
        return mean

class PolicyNet(nn.Module):
    def __init__(self, learning_rate, in_channels, hidden_channels, out_channels, heads, prediction_net, hidden1=512,
                 hidden2=256):
        super(PolicyNet, self).__init__()

        self.gat1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.gat3 = GATv2Conv(in_channels + 1, hidden_channels, heads=heads, concat=True)
        self.gat4 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.fc = nn.Linear(out_channels * 2 * 2, hidden1)
        self.fc1 = nn.Linear(hidden1, hidden2)

        self.fc_mu = nn.Linear(hidden2, n_goods)
        self.fc_std = nn.Linear(hidden2, n_goods)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, g, next_demand):
        x = torch.cat([g.x, next_demand.view(g.num_nodes, -1)], dim=1)

        xb = self.gat3(x, g.reverse_edge_index)
        xb = self.leaky_relu(xb)
        xb = self.gat4(xb, g.reverse_edge_index)
        xb = self.leaky_relu(xb)

        xf = self.gat1(xb, g.edge_index)
        xf = self.leaky_relu(xf)
        xf = self.gat2(xf, g.edge_index)
        xf = self.leaky_relu(xf)

        s = torch.cat([xf, xb], dim=1)

        s = torch.cat([
            s[g.edge_index[0]],
            s[g.edge_index[1]]
        ], dim=1)
        s = F.relu(self.fc(s))
        s = F.relu(self.fc1(s))
        mu = self.fc_mu(s)
        std = F.softplus(self.fc_std(s))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).view(g.num_graphs, n_edge * n_goods).sum(-1, keepdim=True)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7).view(g.num_graphs,
                                                                                        n_edge * n_goods).sum(-1,
                                                                                                              keepdim=True)
        return real_action.view(-1), real_log_prob

    def select_action(self, g, next_demand, test=False):
        x = torch.cat([g.x, next_demand[:, -1, :, :].view(g.num_nodes, -1)], dim=1)

        xb = self.gat3(x, g.reverse_edge_index)
        xb = self.leaky_relu(xb)
        xb = self.gat4(xb, g.reverse_edge_index)
        xb = self.leaky_relu(xb)

        xf = self.gat1(xb, g.edge_index)
        xf = self.leaky_relu(xf)
        xf = self.gat2(xf, g.edge_index)
        xf = self.leaky_relu(xf)

        s = torch.cat([xf, xb], dim=1)
        
        s = torch.cat([
            s[g.edge_index[0]],
            s[g.edge_index[1]]
        ], dim=1)

        s = F.relu(self.fc(s))
        s = F.relu(self.fc1(s))

        mu = self.fc_mu(s)
        std = F.softplus(self.fc_std(s))
        dist = Normal(mu, std)
        if test:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).view(g.num_graphs, n_edge * n_goods).sum(-1, keepdim=True)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7).view(g.num_graphs,
                                                                                        n_edge * n_goods).sum(-1,
                                                                                                              keepdim=True)
        return real_action.view(-1), real_log_prob

    def train_net(self, q1, q2, predict_demand, mini_batch):
        g, _, _, _, _, next_demand, _ = mini_batch
        g_batch = Batch.from_data_list(g)
        a, log_prob = self.forward(g_batch, next_demand)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(g_batch, a, next_demand), q2(g_batch, a, next_demand)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  
        writer.add_scalar("rollout/actor_loss", loss.mean())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        writer.add_scalar("rollout/alpha_loss", alpha_loss)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    def __init__(self, learning_rate, in_channels, hidden_channels, out_channels, heads, prediction_net, hidden1=1024,
                 hidden2=512):
        super(QNet, self).__init__()
        self.out_channels = out_channels
        self.gat1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=True, edge_dim=in_channels+n_goods)
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.gat3 = GATv2Conv(in_channels + 1, hidden_channels, heads=heads, concat=True, edge_dim=in_channels+n_goods)
        self.gat4 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.fc_s = nn.Linear(out_channels * 2 * n_node, hidden1)
        self.fc_a = nn.Linear(n_edge * n_goods, hidden1)
        self.fc_cat = nn.Linear(hidden1 + hidden1, hidden1)
        self.fc_out = nn.Linear(hidden1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, g, a, next_demand):
        a_graph = a.view(g.num_graphs, n_goods * n_edge)
        x = torch.cat([g.x, next_demand.view(g.num_nodes, -1)], dim=1)
        ub = self.gat3(x, g.reverse_edge_index)
        ub = self.leaky_relu(ub)
        ub = self.gat4(ub, g.reverse_edge_index)
        ub = self.leaky_relu(ub)

        uf = self.gat1(ub, g.edge_index)
        uf = self.leaky_relu(uf)
        uf = self.gat2(uf, g.edge_index)
        uf = self.leaky_relu(uf)

        s = torch.cat([uf, ub], dim=1)
        s = s.view(g.num_graphs, -1)
        h1 = self.fc_s(s)
        h2 = self.fc_a(a_graph)
        cat = torch.cat([h1, h2], dim=1)

        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, predict_demand, mini_batch):
        g, a, r, g_prime, done_mask, next_demand, _ = mini_batch
        g_batch = Batch.from_data_list(g)
        critic_loss = F.smooth_l1_loss(self.forward(g_batch, a, next_demand), target).mean()
        writer.add_scalar("rollout/critic_loss", critic_loss)
        loss = critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, predict_demand, mini_batch):
    g, a, r, g_prime, done_mask, _, next_demand = mini_batch
    g_prime_batch = Batch.from_data_list(g_prime)

    with torch.no_grad():
        a_prime, log_prob = pi(g_prime_batch, next_demand)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(g_prime_batch, a_prime, next_demand), q2(g_prime_batch, a_prime, next_demand)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done_mask * (min_q + entropy)

    return target

class SAC_FBiT():
    def __init__(self, epi_num=5000, test_num=100, batch_size=batch_size, buffer_limit=buffer_limit):
        self.epi_num = epi_num
        self.test_num = test_num
        self.batch_size = batch_size
        self.buffer_limit = buffer_limit

    def learn(self, env_mode, env):
        global result_path
        base_path = f'results/{env_id}/{seed}/{os.path.splitext(os.path.basename(__file__))[0]}/mode{env_mode}/learn'
        counter = 1
        result_path = f'{base_path}{counter}'
        while os.path.exists(result_path):
            counter += 1
            result_path = f'{base_path}{counter}'

        os.makedirs(result_path, exist_ok=True)
        global writer
        writer = SummaryWriter(result_path)
        print(result_path)

        global features_num, n_factory, n_goods, n_node, n_stores, n_edge
        self.utils = g_env0_utils(env, 3, device)

        features_num, n_factory, n_goods, n_node, n_stores, n_edge, _ = self.utils.return_info()
        global memory, target_entropy
        memory = EpisodeReplayBuffer_G(buffer_limit, epi_length, device)
        target_entropy = -n_edge
        ensemble_size = 1
        predict_demand_ensemble = [
            predictorNet(0.001).to(device)
            for _ in range(ensemble_size)
        ]
        q1, q2, q1_target, q2_target = (QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device))
        pi = PolicyNet(lr_pi, features_num, 64, 128, 3, prediction_net=predict_demand_ensemble).to(device)

        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        score = 0.0
        print_interval = 10
        max_score = float('-inf')
        train_num = 0

        for n_epi in range(1, self.epi_num + 1):
            s = env.reset()
            done = False
            truncated = [False]
            count = 0
            n_epi_score = 0
            g = self.utils.generate_graphs_store_lack_demand(s[0])
            g_list = []
            g_list.append(g)
            next_demand = predictorNet.ensemble_predict(predict_demand_ensemble, Batch.from_data_list(g_list))
            while count < 500 and not done and not truncated[0]:

                a, log_prob = pi.select_action(Batch.from_data_list([g]), next_demand)
                a = a.cpu().detach().numpy()
                s_prime, r, truncated, info = env.step([a])
                if truncated[0]:
                    next_g = self.utils.generate_graphs_store_lack_demand(info[0]['terminal_observation'])
                else:
                    next_g = self.utils.generate_graphs_store_lack_demand(s_prime[0])
                g_list.append(next_g)
                next_next_demand = predictorNet.ensemble_predict(predict_demand_ensemble, Batch.from_data_list(g_list))
                memory.put((g, a, r[0] / 100, next_g, truncated[0], next_demand[0, -1, :, :].detach(), next_next_demand[0, -1, :, :].detach()))
                n_epi_score += r[0]
                g = next_g
                next_demand = next_next_demand
                count += 1
            score += n_epi_score

            if memory.size() > 20:
                for i in range(10):
                    mini_batch = memory.sample(batch_size)
                    td_target = calc_target(pi, q1_target, q2_target, predict_demand_ensemble, mini_batch)
                    q1.train_net(td_target, predict_demand_ensemble, mini_batch)
                    q2.train_net(td_target, predict_demand_ensemble, mini_batch)
                    entropy = pi.train_net(q1, q2, predict_demand_ensemble, mini_batch)
                    q1.soft_update(q1_target)
                    q2.soft_update(q2_target)
                    for model in predict_demand_ensemble:
                        model.train_net(mini_batch)
            else:       
                for i in range(100):
                    mini_batch = memory.sample(1)
                    for model in predict_demand_ensemble:
                        model.train_net(mini_batch)

            writer.add_scalar("rollout/ep_rew_iter", n_epi_score, n_epi * epi_length)
            if max_score < n_epi_score:
                ensemble_state_dicts = [net.state_dict() for net in predict_demand_ensemble]
                save_dict = {
                'ensemble': ensemble_state_dicts,
                'q1': q1.state_dict(),
                'q2': q2.state_dict(),
                'q1_target': q1_target.state_dict(),
                'q2_target': q2_target.state_dict(),
                'pi': pi.state_dict()
                }
                torch.save(save_dict, f'{result_path}/best_model.pth')
                print("save best model")
                max_score = n_epi_score
            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                                 pi.log_alpha.exp()))
                writer.add_scalar("rollout/ep_rew_mean", score / print_interval, n_epi * epi_length)
                score = 0.0

    def test(self, env_mode, env, model_path=None, n_eval_episodes=100, render=False):
        global result_path
        if model_path is None:
            base_path = f'results/{env_id}/{seed}/{os.path.splitext(os.path.basename(__file__))[0]}/mode{env_mode}/learn1'
            model_path = f'{base_path}/best_model.pth' 

        assert os.path.exists(model_path), f"模型文件 {model_path} 不存在，请先训练！"

        self.utils = g_env0_utils(env, 3, device)
        global features_num, n_factory, n_goods, n_node, n_stores, n_edge
        features_num, n_factory, n_goods, n_node, n_stores, n_edge, _ = self.utils.return_info()

        ensemble_size = 1
        predict_demand_ensemble = [
            predictorNet(0.001).to(device)
            for _ in range(ensemble_size)
        ]
        q1, q2, q1_target, q2_target = (QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device),
                                        QNet(lr_q, features_num, 64, 128, 3, predict_demand_ensemble).to(device))
        pi = PolicyNet(lr_pi, features_num, 64, 128, 3, prediction_net=predict_demand_ensemble).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        for model, state_dict in zip(predict_demand_ensemble, checkpoint['ensemble']):
            model.load_state_dict(state_dict)
        q1.load_state_dict(checkpoint['q1'])
        q2.load_state_dict(checkpoint['q2'])
        q1_target.load_state_dict(checkpoint['q1_target'])
        q2_target.load_state_dict(checkpoint['q2_target'])
        pi.load_state_dict(checkpoint['pi'])

        print(f"成功加载模型: {model_path}")

        returns = []
        for ep in range(n_eval_episodes):
            s = env.reset()
            done = False
            truncated = [False]
            g = self.utils.generate_graphs_store_lack_demand(s[0])
            g_list = [g]
            next_demand = predictorNet.ensemble_predict(predict_demand_ensemble, Batch.from_data_list(g_list))
            # next_demand += torch.rand(1, 1, n_node, 1).to(device)
            # next_demand = torch.rand(1, 1, n_node, 1).to(device)

            ep_return = 0
            step = 0
            while not done and not truncated[0] and step < 500:
                a, _ = pi.select_action(Batch.from_data_list([g]), next_demand, True)
                a = a.cpu().detach().numpy()
                s_prime, r, truncated, info = env.step([a])
                ep_return += r[0]
                if render:
                    env.render()

                if truncated[0]:
                    next_g = self.utils.generate_graphs_store_lack_demand(info[0]['terminal_observation'])
                else:
                    next_g = self.utils.generate_graphs_store_lack_demand(s_prime[0])
                g_list.append(next_g)
                next_demand = predictorNet.ensemble_predict(predict_demand_ensemble, Batch.from_data_list(g_list))
                # next_demand += torch.rand(1, 1, n_node, 1).to(device)
                # next_demand = torch.rand(1, 1, n_node, 1).to(device)
                g = next_g
                step += 1
            returns.append(ep_return)
            print(f"Episode {ep+1}: Return = {ep_return:.2f}")

        avg_return = sum(returns) / len(returns)
        print(f"平均回报 (over {n_eval_episodes} episodes): {avg_return:.2f}")

        env.close()
        csv_path = os.path.join(base_path, "test_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Return"])
            for i, r in enumerate(returns, start=1):
                writer.writerow([i, r])
        return returns, avg_return


if __name__ == '__main__':
    env_mode_list = [1]
    epi_num = 5000
    env_id = 'SupplyChainG-v0'
    for env_mode in env_mode_list:
        env = prepare_training_env(env_id, env_mode=env_mode, seed=seed, result_path=result_path)
        sac_fbit = SAC_FBiT(epi_num)
        sac_fbit.learn(env_mode=env_mode, env=env)
        env = prepare_training_env('SupplyChainG-v0', env_mode=env_mode, seed=200, result_path=result_path)
        sac_fbit.test(env_mode=env_mode, env=env)
