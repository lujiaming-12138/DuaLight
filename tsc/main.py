#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: main
"""
import os
import sys

import torch

from config import config
from a3c_trainer import train_A3C_PPO


def run():
    train_A3C_PPO(config, device)


if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    run()
