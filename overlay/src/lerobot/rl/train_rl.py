# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Top-level pipeline config for distributed RL training (actor / learner)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

from .algorithms.configs import RLAlgorithmConfig
from .algorithms.factory import make_algorithm_config
from .algorithms.sac import SACAlgorithmConfig  # noqa: F401


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

    # Algorithm config.
    algorithm: RLAlgorithmConfig | None = None

    # Data mixer strategy name. Currently supports "online_offline".
    mixer: str = "online_offline"
    # Fraction sampled from online replay when using OnlineOfflineMixer.
    online_ratio: float = 0.5

    # Actor-side live view of cameras (processed and raw), reward and executed action: "rerun",
    # "foxglove" or None
    display_mode: str | None = None
    # Compress images to JPEG before streaming them (less bandwidth and memory, more CPU)
    display_compressed_images: bool = False

    # Probability, per actor step, of trying to close the gripper while it is open. The policy never
    # explores the gripper otherwise (it takes the argmax of the discrete critic); random opening is
    # not used since it would drop held objects. 0 disables it.
    discrete_action_epsilon: float = 0.0

    # When set (and `output_dir` is not), the learner and the actor each write to their own run
    # directory in here: `<timestamp>_learner_<name>` / `<timestamp>_actor_<name>`, where `<name>` is
    # `job_name` without its `hilserl_` prefix.
    runs_dir: str | None = None

    def set_run_dir(self, role: str) -> None:
        """Pick `output_dir` for the learner or the actor process (call before `validate`)."""
        if self.output_dir is not None or self.runs_dir is None:
            return
        name = (self.job_name or "hilserl").removeprefix("hilserl_")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(self.runs_dir) / f"{timestamp}_{role}_{name}"

    def validate(self) -> None:
        super().validate()

        if self.algorithm is None:
            self.algorithm = make_algorithm_config("sac")

        if getattr(self.algorithm, "policy_config", None) is None:
            self.algorithm.policy_config = self.policy

        # Potential-based shaping only preserves the optimal policy with the RL discount
        staged = getattr(getattr(self.env, "processor", None), "staged_reward", None)
        discount = getattr(self.algorithm, "discount", None)
        if staged is not None and discount is not None and abs(staged.discount - discount) > 1e-9:
            raise ValueError(
                f"env.processor.staged_reward.discount ({staged.discount}) must equal algorithm.discount ({discount})"
            )
