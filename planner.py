"""NavRep E2E1D (end-to-end PPO over a 1-D lidar encoder) wrapper for the
arena_planners edge bridge.

Reconstructs, outside the upstream NavRep ``E2E1DNavRepEnv`` simulator, the exact
1085-d observation that ``navrep/envs/e2eenv.py::FlatLidarAndStateEncoder`` feeds
the policy:

    obs = concat( lidar(1080 raw ranges, meters), robotstate(5) )
    robotstate = [ goal_x, goal_y, vx, vy, vth ]  -- all in the robot/baselink frame

and runs the conv1d-encoder + MLP controller (``net.NavRepE2E1DNet``, weights
extracted from the TF1 stable-baselines checkpoint). The policy emits a holonomic
``[vx, vy]`` velocity in m/s, clipped to the trained action space [-1, 1]
(robot max speed 1.0). The bridge owns the holonomic->diff-drive projection, so
``step`` returns the raw ``[vx, vy]`` and never touches omega.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch
from arena_planners.geometry import world_to_robot_frame
from arena_planners.sdk import load_manifest, main_loop

from net import LIDAR_SIZE, OBS_SIZE, NavRepE2E1DNet

_WEIGHTS = os.path.join(os.path.dirname(__file__), "model", "navrep_e2e1d.npz")

# NavRep lidar geometry (navreptrainenv.py): 1080-ray 360-deg scan, background
# fill 25.0 m. Unseen/invalid returns are clamped to this maximum range.
_RANGE_MAX: float = 25.0
# Action space Box(-1, 1, shape=(2,)); robot_max_speed = 1.0 m/s (lib_dwa).
_ACTION_CLIP: float = 1.0

_net: NavRepE2E1DNet | None = None
_device = torch.device("cpu")


def _get_net() -> NavRepE2E1DNet:
    global _net
    if _net is None:
        _net = NavRepE2E1DNet().load_npz(_WEIGHTS).to(_device)
    return _net


def _resample_lidar(scan: np.ndarray) -> np.ndarray:
    """Map an arbitrary-length LaserScan to NavRep's 1080 rays.

    NavRep spans a full 360 deg with 1080 evenly-spaced rays. An incoming scan
    is linearly resampled onto that grid (a no-op when it already has 1080 rays),
    then clamped to [0, range_max] like the upstream contour renderer.
    """
    scan = np.asarray(scan, dtype=np.float32).ravel()
    n = scan.shape[0]
    if n == 0:
        return np.full(LIDAR_SIZE, _RANGE_MAX, dtype=np.float32)
    if n != LIDAR_SIZE:
        src = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dst = np.linspace(0.0, 1.0, LIDAR_SIZE, dtype=np.float32)
        scan = np.interp(dst, src, scan).astype(np.float32)
    scan = np.nan_to_num(scan, nan=_RANGE_MAX, posinf=_RANGE_MAX, neginf=_RANGE_MAX)
    return np.clip(scan, 0.0, _RANGE_MAX)


def _state_features(features: dict) -> np.ndarray:
    """Build NavRep's 5 robotstate features in the baselink frame.

    [goal_x_bl, goal_y_bl, vx_bl, vy_bl, vth] -- goal transformed into robot frame;
    world-frame robot velocity rotated into baselink (matching apply_tf_to_vel);
    angular velocity is frame-invariant in 2D.
    """
    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")
    robot_state = features.get("robot_state")

    theta = float(robot_pose[2]) if robot_pose is not None and len(robot_pose) >= 3 else 0.0

    goal_x_bl, goal_y_bl = 0.0, 0.0
    if robot_pose is not None and goal_pose is not None and len(robot_pose) >= 3 and len(goal_pose) >= 2:
        goal_x_bl, goal_y_bl = world_to_robot_frame((float(goal_pose[0]), float(goal_pose[1])), robot_pose)

    # robot_state from OdometryCollector: [x, y, vx, vy, theta, vth] (world-frame velocity).
    vx_w = vy_w = vth = 0.0
    if robot_state is not None and len(robot_state) >= 6:
        vx_w, vy_w, vth = float(robot_state[2]), float(robot_state[3]), float(robot_state[5])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vx_bl = cos_t * vx_w + sin_t * vy_w
    vy_bl = -sin_t * vx_w + cos_t * vy_w

    return np.array([goal_x_bl, goal_y_bl, vx_bl, vy_bl, vth], dtype=np.float32)


def step(features: dict) -> list[float]:
    """Compute a holonomic [vx, vy] twist from raw obs."""
    net = _get_net()

    robot_pose = features.get("robot_pose")
    lidar = _resample_lidar(features.get("laser_scan"))
    state = _state_features(features)
    obs = np.concatenate([lidar, state]).astype(np.float32)
    if obs.shape[0] != OBS_SIZE:  # defensive; both halves are fixed-size
        obs = np.resize(obs, OBS_SIZE).astype(np.float32)

    with torch.no_grad():
        mean = net(torch.from_numpy(obs).unsqueeze(0).to(_device))
    action = mean.squeeze(0).cpu().numpy().astype(np.float64)
    # Deterministic action = gaussian mean. SB2 clips per-axis to Box(-1, 1),
    # which admits ||v||=sqrt(2); norm-clip to v_pref instead to respect the
    # robot's max speed without distorting direction.
    norm = float(np.linalg.norm(action))
    if norm > _ACTION_CLIP:
        action = action / norm * _ACTION_CLIP
    # Policy output is baselink-frame (upstream applies the action as vx_baselink);
    # the omni bridge expects world-frame and rotates by R(-theta), so pre-rotate
    # body->world by R(theta) to cancel it.
    theta = float(robot_pose[2]) if robot_pose is not None and len(robot_pose) >= 3 else 0.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return [float(action[0] * cos_t - action[1] * sin_t),
            float(action[0] * sin_t + action[1] * cos_t)]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    # Feed-forward policy with no recurrent state; nothing to reset.
    pass


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
