import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from gymnasium.envs.registration import register

register(
    id='TrafficManagement-v0',
    entry_point='env.env:TrafficManagementEnv',
)