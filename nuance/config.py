import os

PHRASES = (
    "slightly faster", "barely faster", "a little faster", "a bit faster", "a tad faster", "just a touch faster",
    "a little quicker", "speed up a bit", "pick it up a little", "increase speed slightly", "much faster",
    "a lot faster", "much quicker", "way faster", "significantly faster", "speed up a lot", "pick up the pace",
    "hurry up", "step on it", "go as fast as you can", "faster", "go faster", "quicker", "speed up",
    "increase speed", "you're going too slow", "slightly slower", "barely slower", "a little slower",
    "a bit slower", "a tad slower", "slow down a touch", "ease off a bit", "decrease speed slightly", "much slower",
    "a lot slower", "way slower", "significantly slower", "slow way down", "take it really slow", "go very slowly",
    "slower", "go slower", "slow down", "decrease speed", "reduce speed", "take it easy", "you're going too fast",
    "keep this pace", "same speed", "don't change your speed", "this speed is fine",
)
CLIP_MAX_S = 20.0
HOLD_S = 1.5  # Spot stands still at the end of the line this long before the clip ends
T_DECISION_S = 2.0
FPS = 30
RAMP_S = 0.5
SPOT_MAX_SPEED = 1.6
MIN_SPEED = 0.25
START_SPEED_RANGE = (0.3, 1.0)
START_SPEED_SPLIT = 0.65
CANDIDATE_BANDS = {"slower": (0.2, 0.75), "near": (0.85, 1.15), "faster": (1.3, 2.5)}  # multiples of v0
MIN_ADJ_RATIO = 1.25
POST_DECISION_M = 3.0
NAVMESH_AGENT_RADIUS = 0.4
NAVMESH_AGENT_HEIGHT = 0.9
SAME_FLOOR_DY = 0.3
SPOT_HALF_LENGTH = 0.55
BODY_MIN_CLEARANCE = 0.10
BODY_CHECK_STEP_M = 0.1
CONTEXT_RADIUS_M = 1.5
AHEAD_M = 3.0
TIGHT_CLEARANCE_M = 0.7
OPEN_CLEARANCE_M = 1.0
TURN_DEG = 45.0
AHEAD_SAMPLE_M = 0.25
NARROWING_RATIO = 0.7
TOPDOWN_RES = 1024
TOPDOWN_CAM_HEIGHT = 2.0
TOPDOWN_MARGIN_M = 1.0
TOPDOWN_MIN_EXTENT_M = 6.0
FRONT_RES = (640, 480)
# Chase view, left of the top-down in each clip frame; widths are multiples of 16 for H.264.
CHASE_RES = (1376, 1024)
CHASE_HFOV = 80.0
# Chase camera candidates around Spot, in order of preference: distance behind it, height, and azimuth off
# straight behind, searched behind-first (every distance and height at one azimuth before the next azimuth). Each
# frame keeps the current candidate while it sees Spot, else takes the first that does, and every CHASE_RESELECT_FRAMES
# it moves back to a better candidate that sees Spot.
CHASE_DISTANCES_M = (2.2, 1.6, 1.1)
CHASE_HEIGHTS_M = (1.7, 2.3)
CHASE_AZIMUTHS_DEG = (0, 25, -25, 50, -50, 80, -80)
CHASE_LOOK_AT_M = 0.45
CHASE_ALPHA = 0.08
CHASE_RESELECT_FRAMES = 15
CHASE_HEADING_AHEAD_M = 1.0
# Low-res depth view from the chase camera that checks a candidate: Spot's outline points are unoccluded by
# anything rendered (objects without collision too), and walls are not pressed against the lens.
CHASE_DEPTH_RES = (172, 128)
CHASE_NEAR_M = 0.6
CHASE_NEAR_FRAC = 0.03  # max fraction of the view nearer than CHASE_NEAR_M
CHASE_VOID_FRAC = 0.25  # max fraction of the view showing nothing (inside a wall)
CHASE_OCCLUDE_TOL_M = 0.15
# Walls and furniture between the camera and Spot: at most CHASE_FORE_FRAC of the view nearer than CHASE_FORE_RATIO
# of the distance to Spot.
CHASE_FORE_RATIO = 0.6
CHASE_FORE_FRAC = 0.12
CHASE_FORE_MAX_AZ_DEG = 0.0
CHASE_ANCHOR_ALPHA = 0.3  # the camera and its aim follow Spot through this, so Spot stays centred without the sway
CHASE_MAX_PITCH_DEG = 50.0  # steeper candidates look down on Spot and flatten its legs
CHASE_MOVING_MPS = 0.1  # below this Spot is standing (the end hold) and the camera stays where it is
# Global lights: one straight down and four low ones around for the walls. Habitat's default four lights sit 45 deg
# up, which a glossy floor mirrors as a bright spot just below Spot in the chase view.
LIGHT_TOP = 1.0
LIGHT_SIDE = 0.45
LIGHT_SIDE_ELEV_DEG = 10.0
FRONT_HFOV = 90.0
FRONT_CAM_HEIGHT = 0.5
FRONT_CAM_FORWARD = 0.45
SPOT_BASE_HEIGHT = 0.54
# Python of the spot-loco env (mujoco, jax, playground) that runs nuance.spot_walk.
LOCO_PYTHON = os.path.expanduser(os.environ.get("NUANCE_LOCO_PYTHON", "~/miniconda3/envs/spot-loco/bin/python"))
SPOT_POLICY_DIR = "data/policies/spot_rough"  # scripts/train_spot_policy.py --terrain rough
SCENE_DATASET = "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
