from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from .humanoid_obstacles import HumanoidWithObstacles

def register_humanoid_obstacles():
    _XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
    _STAND_HEIGHT = 1.4

    WALK_SPEED = 1.0
    RUN_SPEED = 10.0
    
    def _make(cfg):
        # If you exposed config_overrides in your class’s __init__, pass it here as needed.
        return HumanoidWithObstacles(move_speed=WALK_SPEED, run_speed=RUN_SPEED, config=cfg)

    # Start from the environment’s default config and set obstacle-related defaults if needed.
    _DEFAULT_CFG = HumanoidWithObstacles.default_config()
    _DEFAULT_CFG.obstacles = True  # enable obstacles by default
    # Any other defaults, e.g. _DEFAULT_CFG.obstacle_count = 8, etc.

    # Register under a distinct name
    registry.locomotion.register_environment(
        "HumanoidWalkObstacles",
        _make,
        _DEFAULT_CFG,
    )
