# -*- coding: utf-8 -*-
"""
Created on Jun 2020

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
import os
from math import sqrt


class Empty:
    def __init__(self):
        pass


# ~~~~~~~~~~~~~~~~~~~ JOINT SPACE ~~~~~~~~~~~~~~~~~~~ #

class TrajectoryJS:
    ''' Desired trajectory definition
    '''
    def __init__(self, q0, q_fin, T):
        self.A = 1/T*(q_fin-q0)
        self.q = q0

    def compute(self, t):
        self.q = t*self.A + self.q


# ~~~~~~~~~~~~~~~~~~~ TASK  SPACE ~~~~~~~~~~~~~~~~~~~ #

class TrajectoryLin:
    ''' Desired straight trajectory definition with ramp velocity at the beginning and at the end (trapezoidal profile)
    '''
    def __init__(self, x_start, x_fin, T):
        self.DX = (x_fin-x_start)
        self.T = T
        # Profile parameters
        self.a = 0.1
        self.b = - self.a
        self.v1 = 0.5
        self.Dt = self.v1/self.a
        # if the ramp time is too high
        if self.Dt > 0.5*self.T:
            self.Dt = 0.5*self.T
            self.v1 = 0.5*self.a*self.T
        
        self.p0 = x_start                           # Starting point
        self.lamb_max = self.a*self.Dt**2 + self.v1*(self.T - 2*self.Dt)    # Maximum value of curvilinear coordinate
        
        self.p = 0.5*self.a*self.Dt**2              # Position reeched after the first velocity ramp

    def compute(self, t):
        # Computation of the curvilinear coordinate as function of time
        if (t >= 0 and t < self.Dt):
            self.lamb = 0.5*self.a*t**2
        elif (t >= self.Dt and t <= (self.T - self.Dt)):
            self.lamb = self.p + self.v1*(t - self.Dt)
        else:
            self.lamb = self.p + self.v1*(t - self.Dt) + 0.5*self.b*(t - self.T + self.Dt)**2
        
        # Computation of the 3D position as a function of the curvilinear coordinate
        self.pos = self.p0 + (self.lamb/self.lamb_max)*self.DX


class TrajectoryCirc:
    ''' Desired circular trajectory definition with ramp velocity at the beginning and at the end (trapezoidal profile)
        The z coordinate of the trajectory is zero and the center is defined by the class variables xc and yc while
        the radious is defined by the class variable R. We runs over this circular trajectory by using a curvilinear
        coordinate s.
        If the robot end-effector is not in the trajectory place, the starting time is not zero to allows the robot to reach
        the trajectory.
    '''
    def __init__(self, T, T_start = 0):
        self.T = T
        self.T_start = T_start
        # Profile Parameters
        self.a = 0.1
        self.b = - self.a
        self.v1 = 0.1
        self.Dt = self.v1/self.a
        # if the ramp time is too high
        if self.Dt > 0.5*self.T:
            self.Dt = 0.5*self.T
            self.v1 = 0.5*self.a*self.T
        self.p = 0.5*self.a*self.Dt**2              # Position reeched after the first velocity ramp

        # Trajectory Parameters
        self.xc, self.yc = 0.0, 0.  # [m]
        self.R = 0.5                # [m]
        self.f = 0.25               # [Hz]
        self.pos = np.zeros(3)
        self.lamb_max = self.a*self.Dt**2 + self.v1*(self.T - 2*self.Dt)    # Maximum value of curvilinear coordinate


    def compute(self, t):
        t = t - self.T_start
        # Computation of the curvilinear coordinate as function of time
        if (t <= 0):
            self.lamb = 0
        elif (t > 0 and t < self.Dt):
            self.lamb = 0.5*self.a*t**2
        elif (t >= self.Dt and t <= (self.T - self.Dt)):
            self.lamb = self.p + self.v1*(t - self.Dt)
        else:
            self.lamb = self.p + self.v1*(t - self.Dt) + 0.5*self.b*(t - self.T + self.Dt)**2
        
        # Computation of the 3D position as a function of the curvilinear coordinate
        self.pos[0] = self.xc + self.R*np.cos(2*np.pi*self.f*self.lamb/self.lamb_max)
        self.pos[1] = self.yc + self.R*np.sin(2*np.pi*self.f*self.lamb/self.lamb_max)
