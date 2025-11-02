from gymnasium.envs.registration import register

register(
    id="SupplyChainG-v0",
    entry_point="supply_chain_env.envs:SupplyChainGV0",
)
