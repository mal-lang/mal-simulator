# -*- encoding: utf-8 -*-
# MAL Petting Zoo Simulator v0.0.12
# Copyright 2024, Andrei Buhaiu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from malpzsim.wrappers.wrapper import LazyWrapper
from malpzsim.wrappers.gym_wrapper import AttackerEnv, DefenderEnv, register_envs

"""
MAL Petting Zoo Simulator
"""

__title__ = "malpzsim"
__version__ = "0.0.12"
__authors__ = ["Andrei Buhaiu", "Jakob Nyberg"]
__license__ = "Apache 2.0"
__docformat__ = "restructuredtext en"

__all__ = ("LazyWrapper", "AttackerEnv", "DefenderEnv", "register_envs")
