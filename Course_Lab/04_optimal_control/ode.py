# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Classes representing different kind of Ordinary Differential Equations (ODEs).

@author: student
"""

import numpy as np
import pinocchio as pin
        
        
class ODESin:
    ''' ODE defining a sinusoidal trajectory '''
    def __init__(self, name, A, f, phi):
        self.name = name
        self.A = A
        self.two_pi_f = 2*np.pi*f
        self.phi = phi
        
    def f(self, x, u, t):
        return self.two_pi_f*self.A*np.cos(self.two_pi_f*t + self.phi)
       
       
class ODELinear:
    ''' A linear ODE: dx = A*x + b
    '''
    def __init__(self, name, A, b):
        self.name = name
        self.A = A
        self.b = b
        
    def f(self, x, u, t):
        return self.A.dot(x) + self.b
        
        
class ODEStiffDiehl:
    def f(self, x, u, t):
        return -50.0*(x - np.cos(t))
        
        
class ODEPendulum:
    def __init__(self):
        self.g = -9.81
        
    def f(self, x, u, t):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = self.g*np.sin(x[0])
        return dx
        
        
class ODERobot:
    ''' An ordinary differential equation representing a robotic system
    '''
    
    def __init__(self, name, robot):
        ''' robot: instance of RobotWrapper
        '''
        self.name = name
        self.robot = robot
        self.nu = robot.na
        nq, nv = self.robot.nq, self.robot.nv
        self.nx = nq+nv
        self.nu = self.robot.na
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)
        
        
    ''' System dynamics '''
    def f(self, x, u, t, jacobian=False):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        
        if(nv==1):
            # for 1 DoF systems pin.aba does not work (I don't know why)
            pin.computeAllTerms(model, data, q, v)
            ddq = (self.u-data.nle) / data.M[0]
        else:
            ddq = pin.aba(model, data, q, v, u)

        self.dx[:nv] = v
        self.dx[nv:] = ddq
        
        if(jacobian):
            pin.computeABADerivatives(model, data, q, v, u)
            self.Fx[:nv, :nv] = 0.0
            self.Fx[:nv, nv:] = np.identity(nv)
            self.Fx[nv:, :nv] = data.ddq_dq
            self.Fx[nv:, nv:] = data.ddq_dv
            self.Fu[nv:, :] = data.Minv
            
            return (np.copy(self.dx), np.copy(self.Fx), np.copy(self.Fu))
        
        return np.copy(self.dx)
        
        
    def f_x_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. x computed via finite differences '''
        f0 = self.f(x, t)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, t)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
        
    def f_u_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed via finite differences '''
        f0 = self.f(x, u)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up)
            Fu[:,i] = (fp-f0)/delta
        return Fu
