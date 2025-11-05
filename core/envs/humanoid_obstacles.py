# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Humanoid environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from xml.etree import ElementTree as ET
from etils import epath
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common



class HumanoidWithObstacles(mjx_env.MjxEnv):
    """Humanoid environment with optional tripping obstacles and robust locomotion shaping."""

    def __init__(
        self,
        move_speed: float,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        xml_path: str | None = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )

        self._move_speed = move_speed
        if self._move_speed == 0.0:
            self._stand_or_move_reward = self._stand_reward
        else:
            self._stand_or_move_reward = self._move_reward

        # Load and optionally patch MJCF to inject obstacles.
        if xml_path is None:
            xml_path = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
        else:
            xml_path = epath.Path(xml_path)
        xml_text = xml_path.read_text()
        if bool(self._config.get("obstacles", False)):
            xml_text = self._inject_obstacles(xml_text)

        self._xml_path = xml_path.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(xml_text, self._model_assets)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    @classmethod
    def default_config(cls) -> config_dict.ConfigDict:
        return config_dict.create(
            ctrl_dt=0.025,
            sim_dt=0.005,
            episode_length=1000,
            action_repeat=1,
            vision=False,
            impl="jax",
            nconmax=200_000,
            njmax=250,
            # Obstacles
            obstacles=False,
            obstacle_count=8,
            obstacle_size_xy=[0.03, 0.03],     # 3 cm square
            obstacle_height=0.015,             # 1.5 cm tall
            obstacle_spacing=0.35,             # 35 cm between blocks
            obstacle_start_x=0.8,              # first block center x
            obstacle_y=0.0,                    # aligned to center line
            obstacle_group=3,                  # geom group tag for obstacles
            # Reward weights
            w_clearance=0.75,                  # scales clearance factor
            w_trip=0.5,                        # scales trip-avoid penalty
            clearance_margin=0.02,             # extra margin over obstacle height
        )

    def _inject_obstacles(self, xml_text: str) -> str:
        """Insert a row of small box geoms as tripping obstacles."""
        root = ET.fromstring(xml_text)

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF missing <worldbody> while injecting obstacles.")

        count = int(self._config.get("obstacle_count", 8))
        sx, sy = map(float, self._config.get("obstacle_size_xy", [0.03, 0.03]))
        h = float(self._config.get("obstacle_height", 0.015))
        spacing = float(self._config.get("obstacle_spacing", 0.35))
        start_x = float(self._config.get("obstacle_start_x", 0.8))
        y = float(self._config.get("obstacle_y", 0.0))
        group = int(self._config.get("obstacle_group", 3))

        # Create static box geoms at z = height/2 so top sits at height.
        for i in range(count):
            x = start_x + i * spacing
            geom = ET.Element("geom")
            geom.set("name", f"trip_block_{i}")
            geom.set("type", "box")
            geom.set("size", f"{sx} {sy} {h/2.0}")
            geom.set("pos", f"{x} {y} {h/2.0}")
            geom.set("rgba", "0.8 0.2 0.2 1")
            geom.set("group", str(group))
            # Ensure collisions with robot and ground using default contype/conaffinity=1
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            # Moderate friction to prevent sliding over tiny blocks.
            geom.set("friction", "1.0 0.01 0.0001")
            worldbody.append(geom)

        return ET.tostring(root, encoding="unicode")

    def _post_init(self) -> None:
        # Body ids used by existing observations.
        self._head_body_id = self.mj_model.body("head").id
        self._torso_body_id = self.mj_model.body("torso").id

        extremities_ids = []
        for side in ("left_", "right_"):
            for limb in ("hand", "foot"):
                extremities_ids.append(self.mj_model.body(side + limb).id)
        self._extremities_ids = jp.array(extremities_ids)

        # Precompute foot body and geom masks for contact-based rewards.
        self._left_foot_body_id = self.mj_model.body("left_foot").id
        self._right_foot_body_id = self.mj_model.body("right_foot").id

        # geom -> body mapping from CPU model; convert to device constants.
        geom_bodyid = jp.array(self.mj_model.geom_bodyid)
        geom_group = jp.array(self.mj_model.geom_group)
        ngeom = geom_bodyid.shape[0]

        foot_body_ids = jp.array([self._left_foot_body_id, self._right_foot_body_id])
        foot_mask = (geom_bodyid[None, :] == foot_body_ids[:, None]).any(axis=0)
        self._foot_geom_mask = foot_mask  # (ngeom,)

        # Obstacles are tagged by group.
        obstacle_group = int(self._config.get("obstacle_group", 3))
        self._obstacle_geom_mask = (geom_group == obstacle_group)

        # Clearance target = obstacle height + extra margin.
        self._obstacle_clearance_target = float(
            self._config.get("obstacle_height", 0.015)
        ) + float(self._config.get("clearance_margin", 0.02))

        # Reward weights.
        self._w_clearance = float(self._config.get("w_clearance", 0.75))
        self._w_trip = float(self._config.get("w_trip", 0.5))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.make_data(
            self.mj_model,
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward/standing": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stand": jp.zeros(()),
            "reward/small_control": jp.zeros(()),
            "reward/move": jp.zeros(()),
            "reward/clearance": jp.zeros(()),
            "reward/trip_avoid": jp.ones(()),
        }
        info = {"rng": rng}

        reward_val, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward_val = self._get_reward(data, action, state.info, state.metrics)
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        return mjx_env.State(data, obs, reward_val, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        return jp.concatenate([
            self._joint_angles(data),
            self._head_height(data).reshape(1),
            self._extremities(data).ravel(),
            self._torso_vertical_orientation(data),
            self._center_of_mass_velocity(data),
            data.qvel,
        ])

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.

        standing = reward.tolerance(
            self._head_height(data),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        metrics["reward/standing"] = standing

        upright = reward.tolerance(
            self._torso_upright(data),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        metrics["reward/upright"] = upright

        stand_reward = standing * upright
        metrics["reward/stand"] = stand_reward

        small_control = reward.tolerance(
            action, margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (4 + small_control) / 5
        metrics["reward/small_control"] = small_control

        move_reward = self._stand_or_move_reward(data)
        metrics["reward/move"] = move_reward

        # New: clearance factor (encourage swing-foot height above obstacles).
        clearance_factor = self._clearance_factor(data)
        metrics["reward/clearance"] = clearance_factor

        # New: trip-avoid factor (penalize any foot–obstacle contact).
        trip_avoid = self._trip_avoid_factor(data)
        metrics["reward/trip_avoid"] = trip_avoid

        return stand_reward * move_reward * small_control * clearance_factor * trip_avoid

    def _clearance_factor(self, data: mjx.Data) -> jax.Array:
        # Determine which foot is in stance using any contact excluding obstacles.
        foot_on_non_obstacle = self._foot_in_contact_non_obstacle(data)  # (2,)
        # Swing mask: foot not in non-obstacle contact.
        swing_mask = 1.0 - foot_on_non_obstacle.astype(float)           # (2,)

        # Foot body z positions.
        left_z = data.xpos[self._left_foot_body_id, -1]
        right_z = data.xpos[self._right_foot_body_id, -1]
        foot_z = jp.stack([left_z, right_z])                            # (2,)

        # Reward if swing foot(s) exceed target clearance; steady if both swing/stance ambiguous.
        target = self._obstacle_clearance_target
        # Tolerance yields 1 when >= target; lower when below with given margin.
        per_foot = reward.tolerance(
            foot_z, bounds=(target, float("inf")), margin=target, value_at_margin=0, sigmoid="linear"
        )
        # Use swing_mask to emphasize swing foot clearance; if both feet swing_mask ~ 0, fall back to mean.
        weighted = (per_foot * (0.5 + 0.5 * swing_mask)).mean()
        # Blend with weight into [0, 1] factor to multiply base reward.
        return (1.0 + self._w_clearance * weighted) / (1.0 + self._w_clearance)

    def _trip_avoid_factor(self, data: mjx.Data) -> jax.Array:
        # Identify any contact where one geom is a foot geom and the other is an obstacle geom.
        # contact.geom has shape (ncon, 2)
        g = data.contact.geom  # int32 [ncon, 2]
        if g.shape[0] == 0:
            return 0.0
        else:
            # Masks for geoms on device.
            foot_mask = self._foot_geom_mask  # [ngeom]
            obs_mask = self._obstacle_geom_mask  # [ngeom]

            g1 = g[:, 0]
            g2 = g[:, 1]
            foot_vs_obs = (foot_mask[g1] & obs_mask[g2]) | (foot_mask[g2] & obs_mask[g1])
            collided = foot_vs_obs.any().astype(float)

        # Downweight reward when a trip contact occurs.
        return 1.0 - self._w_trip * jp.minimum(1.0, collided)

    def _foot_in_contact_non_obstacle(self, data: mjx.Data) -> jax.Array:
        """Returns per-foot boolean: any contact against non-obstacle geoms."""
        g = data.contact.geom
        if g.shape[0] == 0:
            return jp.array([0, 0])

        foot_mask = self._foot_geom_mask
        obs_mask = self._obstacle_geom_mask
        non_obs_mask = ~obs_mask

        g1 = g[:, 0]
        g2 = g[:, 1]

        # Contacts where foot touches anything non-obstacle.
        foot_g1 = foot_mask[g1]
        foot_g2 = foot_mask[g2]
        nonobs_g1 = non_obs_mask[g1]
        nonobs_g2 = non_obs_mask[g2]

        # Map contacts to left/right using body ids of geom’s parent bodies.
        geom_bodyid = jp.array(self.mj_model.geom_bodyid)
        b1 = geom_bodyid[g1]
        b2 = geom_bodyid[g2]

        left_id = self._left_foot_body_id
        right_id = self._right_foot_body_id

        # Any contact involving left foot geom and non-obstacle counterpart.
        left_contact = ((foot_g1 & (b1 == left_id) & nonobs_g2) |
                        (foot_g2 & (b2 == left_id) & nonobs_g1)).any()
        right_contact = ((foot_g1 & (b1 == right_id) & nonobs_g2) |
                        (foot_g2 & (b2 == right_id) & nonobs_g1)).any()

        return jp.array([left_contact, right_contact]).astype(int)

    def _stand_reward(self, data: mjx.Data) -> jax.Array:
        horizontal_velocity = self._center_of_mass_velocity(data)[:2]
        dont_move = reward.tolerance(horizontal_velocity, margin=2).mean()
        return dont_move

    def _move_reward(self, data: mjx.Data) -> jax.Array:
        move = reward.tolerance(
            jp.linalg.norm(self._center_of_mass_velocity(data)[:2]),
            bounds=(self._move_speed, float("inf")),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return move

    def _joint_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[7:]

    def _torso_vertical_orientation(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._torso_body_id, 2]

    def _center_of_mass_velocity(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")

    def _center_of_mass_position(self, data: mjx.Data) -> jax.Array:
        return data.subtree_com[self._torso_body_id]

    def _head_height(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._head_body_id, -1]

    def _torso_upright(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._torso_body_id, 2, 2]

    def _extremities(self, data: mjx.Data) -> jax.Array:
        torso_frame = data.xmat[self._torso_body_id]
        torso_pos = data.xpos[self._torso_body_id]
        torso_to_limb = data.xpos[self._extremities_ids] - torso_pos
        return torso_to_limb @ torso_frame

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model