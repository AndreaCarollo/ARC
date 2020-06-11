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


# INTEGRATOR PARAMETERS
T = 4                           # OCP horizon
dt = 0.02                       # OCP time step
integration_scheme = 'RK-4'
use_finite_difference = False


# SYSTEM PARAMETERS
system = 'ur'
nq      = 6
q0      = np.array([ -0.3 , -1.6,  1.6 ,  0. ,  0. ,  0. ])       # initial configuration
q_post  = np.array([ 0.   , -0.5,  0.75,  0. ,  0. ,  0. ])       # postural configuration
q_des   = np.array([ 0.   , -0.6,  0.4 ,  0. ,  0. ,  0. ])       # initial configuration
x0      = np.concatenate((q0, np.zeros(6)))                      # initial state
p_des   = np.array([0.2, 0.5, 0.419])


# COST FUNCTION WEIGHTS
weight_pos  = 1e-0      # cost function weight for final position - task or joint space -
weight_vel  = 5e-1      # cost function weight for final velocity
weight_u    = 5e-9      # cost function weight for control
weight_traj = 1e-0      # cost function weight for follow a trajectory during the time interval [0 T]
weight_post = 2e-3      # cost function weight for the postural task


# TRAJECTORIES DEFINITIONS
# Task space trajectory
class Trajectory:
    ''' Desired trajectory definition
    '''
    def __init__(self, x_start, x_fin, T):
        self.A = (x_fin-x_start)/T
        self.pos_0 = x_start

    def compute(self, t):
        self.pos = self.pos_0 + self.A*t

# Joint space trajectory
class TrajectoryJS:
    ''' Desired trajectory definition
    '''
    def __init__(self, q0, q_fin, T):
        self.A = 1/T*(q_fin-q0)
        self.q = q0

    def compute(self, t):
        self.q = q0 + self.A*t


# ADD CONTACT POINTS & SURFACES
# contact points
frame_name = 'tool0'    # name of the frame to control (end-effector)
# frame_name = 'ee_link'  # name of the frame to control (end-effector)
contact_frames = [frame_name]

# data of the contact surface wall
# contact_surface_name = 'wall'
# contact_surface_pos = np.array([0.65, 0.2, 0.4])
# contact_normal = np.array([-1., 0., 0.])
# K = 3e4*np.diagflat([1., 1., 1.])
# B = 2e2*np.diagflat([1., 1., 1.])
# mu = 0.3

# data of the contact surface floor
contact_surface_name = 'floor'
contact_surface_pos = np.array([0., 0., 0.])
contact_normal = np.array([0., 0., 1.])
K = 3e7*np.diagflat([1., 1., 1.])
B = 2e5*np.diagflat([1., 1., 1.])
mu = 0.3


# SIMULATION PARAMETERS
simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(nq)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 0        # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.0568891763687134, 0.7100808024406433, 0.39807042479515076, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
