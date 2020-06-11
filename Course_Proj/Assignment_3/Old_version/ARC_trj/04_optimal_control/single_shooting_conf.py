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

T = 2                           # OCP horizon
dt = 0.025                       # OCP time step
integration_scheme = 'RK-4'
use_finite_difference = False
sanity_check = False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FINAL COSTS FLAGs
FIN_FIXED_POINT = True
weight_pos      = 10         # cost function weight for final position - task or joint space -

FIN_VEL         = True
weight_vel      = 1e-6      # cost function weight for final velocity

# RUNNING COSTS FLAGs
RUN_QUAD        = False     # enable running quadratic cost
weight_u        = 1e-11     # cost function weight for control

RUN_JOINT_VEL   = True      # enalble running joint velocity
weight_vel_r    = 1e-12     # cost function weight for velocity during the trajectory

RUN_TRAJ_TS     = True      # enable running trajectory on task space
weight_traj     = 25        # cost function weight for follow a trajectory during the time interval [0 T]

RUN_POSTURAL    = True      # enable running postural task
weight_post     = 1e-10     # cost function weight for the postural task

RUN_ORTH_EE     = False     # enable running orthogonality with the floor
weight_orth     = 1e-8      # cost function weigth to keep the end effector orthogonal to the ground

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOT USED
RUN_FIXED_POINT = False     # enable running fixed point
RUN_TRAJ_JS     = False     # enable running trajectory on joint space
weight_post_JS  = 0.01


#system = 'ur' or 'pendulum' or 'double-pendulum'
system = 'ur'

if(system=='ur'):
    nq      = 6
    p0      = np.array([ 0. , 0.4 , 0. ])  # desired starting point to evaluate inverse kinematic
    # q0      = np.array([ 0. , 0. , 0. , np.pi*0.5 , 0. , 0. ])
    # q0      = np.array([ -0.3 , -0.9,  2.1 , -1.2-0.5*np.pi,  1.5*np.pi ,  0. ])       # initial configuration
    # q0      = np.array([ -0.3 , -1.0,  1.2 ,  0. ,  0. ,  0. ])                        # init conf trajectory lienar
    q0      = np.array([ 0., -0.9,  1.5 ,  -0.2 ,  -1.0 - np.pi * 1/2  ,  0.])          # init conf trajectory lienar

    q_post  = np.array([ - np.pi * 1/2, -1.05,  1.2 ,  -0.2 ,  -1.1 - np.pi * 1/2  ,  0.])   # postural configuration
    q_des   = np.array([ 0.     , -0.6,  0.4  ,  0.   ,  0.    ,  0. ])       # initial configuration
    x0      = np.concatenate((q0, np.zeros(6)))                               # initial state
    # p_des   = np.array([ 0.  , 0.5,  0.419])
    p_des   = np.array([ 0.4 , 0.4, -0.05 ])
    
elif(system=='pendulum'):
    nq = 1
    q0 = np.array([np.pi/2])
    x0 = np.concatenate((q0, np.zeros(1)))  # initial state
    p_des = np.array([0.6, 0.2, 0.4]).T     # desired end-effector final position
    frame_name = 'joint1'
elif(system=='double-pendulum'):
    nq = 2
    q0 = np.array([np.pi+0.3, 0.0])
    x0 = np.concatenate((q0, np.zeros(nq)))  # initial state
    p_des = np.array([0.0290872, 0, 0.135])  # upper position
    q_des = np.array([0.0, 0.0])
    frame_name = 'joint2'


# ADD CONTACT POINT & SURFACE
# contact points
frame_name = 'tool0'    # name of the frame to control (end-effector)
#frame_name = 'ee_link'   # name of the frame to control (end-effector)
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
# K = 3e7*np.diagflat([1., 1., 1.])   # troppo stiff
# B = 2e5*np.diagflat([1., 1., 1.])   # troppo stiff
# mu = 0.3

K = np.diagflat([100.0, 100.0, 100.0])
B = 0.0* np.diagflat([ 1,  1, 1])
mu = 0.3


# SIMULATION PARAMETERS
simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(nq)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 0        # flag specifying whether simulation should be real time or as fast as possible
show_floor = True
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.0568891763687134, 0.7100808024406433, 0.39807042479515076, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
