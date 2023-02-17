import os

# from gymnasium.envs.registration import register
from gym.envs.registration import register


with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

register(
    id='kinova-v0',
    entry_point='kinova_gym.envs:KinovaReachEnv',
)
