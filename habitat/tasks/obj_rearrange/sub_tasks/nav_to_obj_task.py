#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv
import numpy as np


@registry.register_task(name="LangNavToObjTask-v0")
class LangDynNavRLEnv(DynNavRLEnv):
    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):
        if start_hold_obj_idx is None:
            # starting positions of candidate objects
            all_pos = np.stack(
                [goal.position for goal in episode.candidate_objects],
                axis=0,
            )
            if force_idx is not None:
                raise NotImplementedError
        else:
            # positions of candidate goal receptacles
            all_pos = np.stack(
                [goal.position for goal in episode.candidate_goal_receps],
                axis=0,
            )

        return all_pos
