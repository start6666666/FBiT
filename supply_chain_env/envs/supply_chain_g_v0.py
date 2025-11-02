import numpy as np
import pandas as pd
import json
import os
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Tuple
import random


class SupplyChainGV0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_mode: int = 1, env_para_dict: dict = None, render_mode: str = None):

        super().__init__()
        self.render_mode = render_mode

        config_file = os.path.join(os.path.dirname(__file__), "config", "v2_config.json")
        with open(config_file, "r") as f:
            all_config = json.load(f)

        self.mode = env_mode
        if 1 <= self.mode <= 4:
            config = all_config[f"mode{env_mode}_env_para"]
        else:
            assert config is None

        self.nodes_config = config["nodes"]
        self.edges_config = config["edges"]

        self.n_nodes = len(self.nodes_config)
        self.n_edges = len(self.edges_config)

        self.node_id_to_idx = {node['id']: i for i, node in enumerate(self.nodes_config)}
        self.nodes = [{'idx': i, **node} for i, node in enumerate(self.nodes_config)]

        self.producer_nodes = [n for n in self.nodes if n['type'] == 'PRODUCER']
        self.retailer_nodes = [n for n in self.nodes if n['type'] == 'RETAILER']

        self.n_producers = len(self.producer_nodes)

        self.n_retailers = len(self.retailer_nodes)

        self.n_warehouses = self.n_nodes - self.n_retailers - self.n_producers

        self.producer_action_idx_to_node_idx = {i: p['idx'] for i, p in enumerate(self.producer_nodes)}

        # Map edge index in action space to edge tuple (from, to)
        self.edge_action_idx_to_edge = {i: (edge[0], edge[1]) for i, edge in enumerate(self.edges_config)}
        # Map edge tuple (from, to) to its index in the action space
        self.edge_to_action_idx = {(edge[0], edge[1]): i for i, edge in enumerate(self.edges_config)}


        self.max_node_cap = np.array([n['capacity'] for n in self.nodes])
        self.max_prod = np.array([n.get('max_production', 0) for n in self.producer_nodes])
        self.max_reta_demand = np.array([n['max_demand'] for n in self.retailer_nodes])

        self.price = config["price"]
        self.prod_costs = np.array([n.get('production_cost', 0) for n in self.producer_nodes])
        self.storage_costs = np.array([n['storage_cost'] for n in self.nodes])
        self.transport_costs = np.array([edge[2]['cost'] for edge in self.edges_config])

        self.penalty_cost = config["penalty_cost"]
        self.waste_cost = config["waste_cost"]
        self.max_truck_cap = config["max_truck_cap"]
        self.episode_length = config["episode_length"]
        self.demand_type = config["demand_type"]

        # --- State and Action Space ---
        self.state_dim = self.n_retailers * 2 + self.n_nodes * 1
        self.action_dim = self.n_producers + self.n_edges

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # --- Action Scaling ---
        self.scale_action = np.ones(self.action_dim)
        # --- Production scaling ---
        self.scale_action[:self.n_producers] = self.max_prod

        for i in range(self.n_edges):
            u_id, v_id, _ = self.edges_config[i]
            u_idx = self.node_id_to_idx[u_id]
            v_idx = self.node_id_to_idx[v_id]
            scale = min(self.max_node_cap[u_idx], self.max_node_cap[v_idx])
            self.scale_action[self.n_producers + i] = scale

        source_nodes_idx = []
        destination_nodes_idx = []
        for producer_node in self.producer_nodes:
            p_idx = producer_node['idx']
            source_nodes_idx.append(p_idx)
            destination_nodes_idx.append(p_idx)
        for edge in self.edges_config:
            u_id, v_id, _ = edge
            u_idx = self.node_id_to_idx[u_id]
            v_idx = self.node_id_to_idx[v_id]
            source_nodes_idx.append(u_idx)
            destination_nodes_idx.append(v_idx)
        self.edge_index = np.array([source_nodes_idx, destination_nodes_idx], dtype=np.int64)

        # --- Dynamic State Variables ---
        self.t = 0
        self.node_stocks = np.zeros(self.n_nodes, dtype=np.float32)

        self.retailer_id_to_idx = {r['id']: i for i, r in enumerate(self.retailer_nodes)}
        self.lack_store = np.zeros(self.n_nodes - self.n_retailers, dtype=np.float32)
        self.lack_store_old = np.zeros(self.n_nodes - self.n_retailers, dtype=np.float32)
        self.lack_demand = np.zeros(self.n_retailers, dtype=np.float32)
        self.lack_demand_old = np.zeros(self.n_retailers, dtype=np.float32)
        self.demand = np.zeros(self.n_retailers, dtype=np.float32)
        self.demand_old1 = np.zeros(self.n_retailers, dtype=np.float32)
        self.demand_offset = 0

        if self.demand_type == 1:
            dir_path = os.path.dirname(__file__)
            csv_path = os.path.join(dir_path, 'demand_processed.csv')
            df = pd.read_csv(csv_path, encoding="gbk")
            self.retailer_demands = []
            for index, row in df.iterrows():
                try:
                    demands = eval(row['demands'])
                    self.retailer_demands.append(demands)
                except (SyntaxError, TypeError):
                    print(f"Error converting demands in row {index}: {row['demands']}")
        
    def _get_obs(self):
        return np.hstack((self.demand_old1.copy(), self.lack_demand.copy(),
                          self.node_stocks.copy()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        self.node_stocks = np.random.random(self.n_nodes) * self.max_node_cap * 0.5

        self.lack_demand.fill(0)
        self.lack_demand_old = self.lack_demand.copy()

        self.demand_offset = np.random.randint(0, self.episode_length)
        self.demand = self.update_demand()
        self.demand_old1 = np.zeros(self.n_retailers, dtype=np.float32)


        return self._get_obs(), {}

    def step(self, action):
        # --- 1. Action Processing ---
        true_action = (np.clip(action, -1, 1) + 1) / 2 * self.scale_action
        true_prod_action = true_action[:self.n_producers]
        true_ship_action = true_action[self.n_producers:]

        waste_produce = 0

        # --- 2. Distribution Phase ---
        shipments = {self.edge_action_idx_to_edge[i]: amount for i, amount in enumerate(true_ship_action)}

        outgoing_shipments = {node_idx: 0 for node_idx in range(self.n_nodes)}
        for (u_id, v_id), amount in shipments.items():
            u_idx = self.node_id_to_idx[u_id]
            outgoing_shipments[u_idx] += amount
        self.lack_store.fill(0)

        final_shipments = shipments.copy()
        for u_idx, total_outgoing in outgoing_shipments.items():
            if total_outgoing > self.node_stocks[u_idx]:
                self.lack_store[u_idx] = total_outgoing - self.node_stocks[u_idx]
                scale_factor = self.node_stocks[u_idx] / total_outgoing
                for edge, amount in shipments.items():
                    if self.node_id_to_idx[edge[0]] == u_idx:
                        final_shipments[edge] *= scale_factor

        arrivals = {node_idx: 0 for node_idx in range(self.n_nodes)}
        for edge, amount in final_shipments.items():
            u_id, v_id = edge
            u_idx, v_idx = self.node_id_to_idx[u_id], self.node_id_to_idx[v_id]

            self.node_stocks[u_idx] = max(0, self.node_stocks[u_idx] - amount)
            arrivals[v_idx] += amount

        for node_idx, arrival_amount in arrivals.items():
            self.node_stocks[node_idx] += arrival_amount
            if self.node_stocks[node_idx] > self.max_node_cap[node_idx]:
                waste_produce += self.node_stocks[node_idx] - self.max_node_cap[node_idx]
                self.node_stocks[node_idx] = self.max_node_cap[node_idx]

        # --- 3. Sales Phase ---
        sell_produce = np.zeros(self.n_retailers, dtype=np.float32)
        self.lack_demand.fill(0)

        for reta_idx, retailer in enumerate(self.retailer_nodes):
            node_idx = retailer['idx']
            total_demand = self.demand[reta_idx]
            if self.node_stocks[node_idx] >= total_demand:
                sell_produce[reta_idx] = total_demand
                self.node_stocks[node_idx] -= total_demand
            else:  # Insufficient supply
                sell_produce[reta_idx] = self.node_stocks[node_idx]
                self.lack_demand[reta_idx] = total_demand - self.node_stocks[node_idx]
                self.node_stocks[node_idx] = 0

        # --- 4. Production Phase ---
        for prod_act_idx, amount in enumerate(true_prod_action):
            node_idx = self.producer_action_idx_to_node_idx[prod_act_idx]
            self.node_stocks[node_idx] += amount
            if self.node_stocks[node_idx] > self.max_node_cap[node_idx]:
                waste_produce += self.node_stocks[node_idx] - self.max_node_cap[node_idx]
                self.node_stocks[node_idx] = self.max_node_cap[node_idx]

        # --- 5. Reward Calculation ---
        if self.demand_type in [0, 1]:
            prices = self.update_price()
        else:
            prices = np.full(self.n_retailers, self.price)

        reward = (np.sum(sell_produce * prices)
                  - np.sum(true_prod_action * self.prod_costs)
                  - np.sum(self.node_stocks * self.storage_costs)
                  - np.sum(self.lack_demand * self.penalty_cost)
                  - np.sum(np.ceil(true_ship_action / self.max_truck_cap) * self.transport_costs)
                  - waste_produce * self.waste_cost)

        # --- 6. State Update ---
        self.t += 1
        self.lack_demand_old = self.lack_demand.copy()
        self.lack_store_old = self.lack_store.copy()
        self.demand_old1 = self.demand.copy()

        terminated = False
        truncated = self.t >= self.episode_length

        if not truncated:
            self.demand = self.update_demand()

        observation = self._get_obs()
        return observation, reward, terminated, truncated, {}

    def update_demand(self):
        demand = np.zeros(self.n_retailers, dtype=np.float32)




        return demand

    def update_price(self):
        # Price fluctuates with demand relative to max demand
        return self.price * self.max_reta_demand * 1.5 / (self.max_reta_demand * 2 - self.demand)

    def render(self):
        if self.render_mode == "human":
            print(f"Time: {self.t}, Stocks: {np.round(self.node_stocks, 2)}")
            print(f"Demand: {np.round(self.demand, 2)}, Backlog: {np.round(self.lack_demand, 2)}")
        elif self.render_mode == "rgb_array":
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def close(self):
        pass

    def get_future_demands(self, start_t: int, H: int) -> np.ndarray:
        demands = np.zeros((H, self.n_retailers), dtype=np.float32)
        



        
        return demands