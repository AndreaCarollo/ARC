#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:09:47 2020

@author: student
"""
import sys
from os.path import dirname, exists, join

import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper
from example_robot_data.robots_loader import getModelPath, readParamsFromSrdf


def loadUR(robot=5, limited=False, gripper=False):
    assert (not (gripper and (robot == 10 or limited)))
    URDF_FILENAME = "ur%i%s_%s.urdf" % (robot, "_joint_limited" if limited else '', 'gripper' if gripper else 'robot')
    URDF_SUBPATH = "/ur_description/urdf/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)
    model = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, ['/opt/openrobots/share/'])
    if robot == 5 or robot == 3 and gripper:
        SRDF_FILENAME = "ur%i%s.srdf" % (robot, '_gripper' if gripper else '')
        SRDF_SUBPATH = "/ur_description/srdf/" + SRDF_FILENAME
        readParamsFromSrdf(model, modelPath + SRDF_SUBPATH, False, False, None)
    return model
