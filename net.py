"""Self-contained torch reimplementation of NavRep's E2E1D (VCARCH, C64) policy.

The upstream checkpoint (``models/gym/*PPO_E2E1D_VCARCH_C64_ckpt.zip``) is a
**stable-baselines 2.10.0 / TensorFlow 1.13** save (``data``/``parameters``/
``parameter_list`` zip layout, ``policy = stable_baselines.common.policies``).
That framework caps at Python 3.7 and will not install on Python >= 3.10, so the
TF1 graph is *not* loaded here. Instead the conv + MLP weights were extracted
from the checkpoint into a plain ``.npz`` of named numpy arrays (see
``model/navrep_e2e1d.npz`` / ``weights.yaml``) and this module reimplements the
exact ``Custom1DPolicy`` forward pass in torch:

    obs (1085) = [ lidar(1080) , robotstate(5) ]
      lidar -> Conv1d(1->32,k8,s4) relu
            -> Conv1d(32->64,k9,s4) relu
            -> Conv1d(64->128,k6,s4) relu
            -> Conv1d(128->256,k4,s4) relu        # length 1080 -> 269 -> 66 -> 16 -> 4
            -> flatten (4*256=1024) -> Linear(1024->32)   (enc_fc_mu, no activation)
      feats = concat(enc(32), robotstate(5)) = 37
      feats -> Linear(37->64) relu -> Linear(64->64) relu -> Linear(64->2)  (pi mean)

Action = the Gaussian mean (deterministic), then clipped to the env action space
[-1, 1] exactly as ``ActorCriticRLModel.predict`` does. The action is a holonomic
``[vx, vy]`` velocity in m/s (robot max speed 1.0). No observation normalization
is applied: the env observation_space is ``Box(-inf, inf)`` so stable-baselines'
``scale=True`` path is skipped (it requires finite bounds), and the raw lidar
ranges (meters) and raw state features go straight in.

TF1 ``tf.layers.conv1d`` stores kernels as ``(kw, Cin, Cout)`` channels-last and
operates on NWC tensors; ``tf.layers.dense`` stores ``(in, out)``. We permute the
conv kernels to torch's ``(Cout, Cin, kw)`` and transpose the dense weights to
``(out, in)`` at load time, and flatten the conv output channels-last to match the
TF ``reshape`` before ``enc_fc_mu``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LIDAR_SIZE: int = 1080  # NavRep n_angles
STATE_SIZE: int = 5     # [goal_x, goal_y, vx, vy, vth] in robot/baselink frame
OBS_SIZE: int = LIDAR_SIZE + STATE_SIZE  # 1085


class NavRepE2E1DNet(nn.Module):
    """Conv1d lidar encoder + MLP controller, weights loaded from the extracted npz."""

    def __init__(self) -> None:
        super().__init__()
        # valid (no padding) conv1d stack, strides 4, mirroring custom_policy.Custom1DPolicy
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=4)
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=6, stride=4)
        self.enc_conv4 = nn.Conv1d(128, 256, kernel_size=4, stride=4)
        self.enc_fc_mu = nn.Linear(1024, 32)
        self.pi_fc0 = nn.Linear(STATE_SIZE + 32, 64)
        self.pi_fc1 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 1085) raw observation -> (B, 2) action mean."""
        lidar = obs[:, :LIDAR_SIZE].unsqueeze(1)  # (B, 1, 1080)  NCW
        state = obs[:, LIDAR_SIZE:]               # (B, 5)
        h = F.relu(self.enc_conv1(lidar))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))             # (B, 256, 4)
        # TF flattened the NWC tensor (B, 4, 256) channels-last; torch h is (B, 256, 4),
        # so transpose to (B, 4, 256) before flattening to match enc_fc_mu's input order.
        h = h.transpose(1, 2).reshape(h.shape[0], -1)  # (B, 1024)
        enc = self.enc_fc_mu(h)                    # (B, 32), no activation (enc_fc_mu)
        feats = torch.cat([enc, state], dim=1)     # (B, 37)
        x = F.relu(self.pi_fc0(feats))
        x = F.relu(self.pi_fc1(x))
        return self.pi(x)                          # (B, 2) gaussian mean

    @torch.no_grad()
    def load_npz(self, path: str) -> "NavRepE2E1DNet":
        """Load extracted TF1 weights, permuting conv/dense layouts to torch."""
        w = np.load(path)

        def conv(layer: nn.Conv1d, name: str) -> None:
            # TF (kw, Cin, Cout) -> torch (Cout, Cin, kw)
            k = torch.from_numpy(np.ascontiguousarray(w[f"{name}.kernel"].transpose(2, 1, 0)))
            b = torch.from_numpy(np.ascontiguousarray(w[f"{name}.bias"]))
            layer.weight.copy_(k)
            layer.bias.copy_(b)

        def dense(layer: nn.Linear, name: str, wkey: str = "kernel", bkey: str = "bias") -> None:
            # TF (in, out) -> torch (out, in)
            k = torch.from_numpy(np.ascontiguousarray(w[f"{name}.{wkey}"].T))
            b = torch.from_numpy(np.ascontiguousarray(w[f"{name}.{bkey}"]))
            layer.weight.copy_(k)
            layer.bias.copy_(b)

        conv(self.enc_conv1, "enc_conv1")
        conv(self.enc_conv2, "enc_conv2")
        conv(self.enc_conv3, "enc_conv3")
        conv(self.enc_conv4, "enc_conv4")
        dense(self.enc_fc_mu, "enc_fc_mu")
        dense(self.pi_fc0, "pi_fc0")
        dense(self.pi_fc1, "pi_fc1")
        dense(self.pi, "pi", wkey="w", bkey="b")  # pi/w:0, pi/b:0 (final linear -> mean)
        self.eval()
        return self
