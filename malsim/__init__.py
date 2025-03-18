# -*- encoding: utf-8 -*-
# MAL Simulator v0.2.7
# Copyright 2025, Andrei Buhaiu.
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

import logging

from malsim.mal_simulator import MalSimulator

"""
MAL Simulator
"""

__title__ = "malsim"
__version__ = "0.2.7"
__authors__ = ["Andrei Buhaiu", "Joakim Loxdal", "Jakob Nyberg", "Nikolaos Kakouros"]
__license__ = "Apache 2.0"
__docformat__ = "restructuredtext en"

__all__ = ["MalSimulator"]


# TODO: Make sure logging dir exists and make it configurable (or use same as maltoolbox)
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
file_handler = logging.FileHandler('logs/malsim_log.txt', mode='w')
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
