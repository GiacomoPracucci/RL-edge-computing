import gymnasium as gym
from env import TrafficManagementEnv
from gymnasium.envs.registration import register

register(
    id='TrafficManagement-v0',
    entry_point='env.env:TrafficManagementEnv',
)

from sheeprl.utils.env import make_env

def create_custom_env(id, render_mode):
    env = gym.make('TrafficManagement-v0', render_mode="rgb_array")
    wrapped_env = TrafficManagementEnv(env)
    return wrapped_env

make_env("TrafficManagement-v0", create_custom_env, idx=0, capture_video=False)