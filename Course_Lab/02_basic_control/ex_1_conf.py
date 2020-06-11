# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ])  # initial configuration

x_des = -np.array([0.7, 0.1, 0.2])  # test 01
#x_des = -np.array([1.2, 0.0, 0.2]) # test 02
#x_des = -np.array([0.7, 0.1, 0.2])  # test 03

frame_name = 'tool0'
MAX_ITER = 100
gradient_threshold = 1e-6   # absolute tolerance on gradient's norm
hessian_regu = 1e-1         # Hessian regularization
beta = 0.5                  # backtracking line search parameter
gamma = 0.1                 # line search convergence parameter
line_search = 0             # flag to enable/disable line search

randomize_robot_model = 0
model_variation = 30.0
simulate_coulomb_friction = 1
simulation_type = 'timestepping' #either 'timestepping' or 'euler'
tau_coulomb_max = 10*np.ones(6) # expressed as percentage of torque max

use_viewer = True
show_floor = False
PRINT_N = 1                   # print every PRINT_N time steps
DISPLAY_N = 1              # update robot configuration in viwewer every DISPLAY_N time steps
DISPLAY_T = 0.1
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
REF_SPHERE_RADIUS = 0.05
REF_SPHERE_COLOR = (1., 0., 0., 1.)
