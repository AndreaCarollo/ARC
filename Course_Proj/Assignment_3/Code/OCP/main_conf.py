# -*- coding: utf-8 -*-
"""
Created on Jun 2020

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60
SHOW_ITERATION_OCP_1    = False
SHOW_THE_FIRST_MOVEMENT = True
SHOW_ITERATION_OCP_2    = True

# Flags to select the desired plots
PLOT_TRACK_OCP      = True
PLOT_CONTACT_FORCE  = True
PLOT_EE_POS         = True
PLOT_JOINT_POS      = False
PLOT_JOINT_VEL      = False
PLOT_TORQUES        = True

T = 2                            # OCP horizon
dt = 0.025                       # OCP time step
integration_scheme = 'RK-4'
use_finite_difference = False
sanity_check = False


# SYSTEM PARAMETERs
system = 'ur'
nq      = 6
q0      = np.array([ 0., -0.9,  1.5 ,  -0.2 ,  -1.0 - np.pi * 1/2  ,  0.])          # init conf trajectory lienar
x0      = np.concatenate((q0, np.zeros(6)))                               # initial state



# Costs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# COST WEIGHTS FOR THE FIRST OCP - WITHOUT CONTACTS
# Final costs weights
weight_vel_1    = 1e-1      # cost function weight for final velocity

# Running cost weights
weight_traj_1   = 8e+0      # cost function weight for follow a trajectory during the time interval [0 T]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# COST WEIGHTS FOR THE SECOND OCP - WITH CONTACTS
# Final costs weights
weight_pos_2    = 10         # cost function weight for final position - task or joint space -
weight_vel_2    = 1e-6      # cost function weight for final velocity

# Running cost weights
weight_vel_r_2  = 1e-12     # cost function weight for velocity during the trajectory
weight_traj_2   = 25        # cost function weight for follow a trajectory during the time interval [0 T]
weight_post_2   = 1e-10     # cost function weight for the postural task
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# CONTACT POINTs & SURFACEs
# contact points
frame_name = 'tool0'    # name of the frame to control (end-effector)
contact_frames = [frame_name]

# data of the contact surface floor
contact_surface_name = 'floor'
contact_surface_pos = np.array([0., 0., 0.])
contact_normal = np.array([0., 0., 1.])

K  = 100 * np.diagflat([1.0 ,1.0, 1.0])
B  = 0.0 * np.diagflat([ 1,  1, 1])
mu = 0.3

# REACTIVE CONTROL
ndt_rc = 25
dt_RC  = dt/ndt_rc       # RC time step

kp   = 10                # proportional gain of end effector task - force control
kp_j = 500               # proportional gain of joint task
kd_j = 2*sqrt(kp_j)      # derivative gain of joint task

# data of the contact surface floor - Reactive control
K_RC = 3e3*np.diagflat([1., 1., 1.])
B_RC = 5e1*np.diagflat([1., 1., 1.])


# SIMULATION PARAMETERS
simulate_coulomb_friction = 0      # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping'   # either 'timestepping' or 'euler'
tau_coulomb_max = 20*np.ones(nq)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 1        # flag specifying whether simulation should be real time or as fast as possible
show_floor = True
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.0568891763687134, 0.7100808024406433, 0.39807042479515076, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
