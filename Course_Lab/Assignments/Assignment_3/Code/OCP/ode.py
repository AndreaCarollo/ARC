# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Classes representing different kind of Ordinary Differential Equations (ODEs).

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
import pinocchio as pin
from numpy.linalg import norm


class ContactPoint_ODE:
    ''' A point on the robot surface that can make contact with surfaces.
    '''
    def __init__(self, model, data, frame_name):
        self.model = model      # robot model
        self.data = data        # robot data
        self.frame_name = frame_name    # name of reference frame associated to this contact point
        self.frame_id = model.getFrameId(frame_name)    # id of the reference frame
        self.active = False         # True if this contact point is in contact
        self.p0 = np.zeros(3)
        
    def get_position(self):
        ''' Get the current position of this contact point 
        '''
        M = self.data.oMf[self.frame_id]
        return M.translation
        
    def get_velocity(self):
        M = self.data.oMf[self.frame_id]
        R = pin.SE3(M.rotation, 0*M.translation)    # same as M but with translation set to zero
        v_local = pin.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear   # convert velocity from local frame to world frame
        return v_world
        
    def get_jacobian(self):
        J6 = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J6[:3,:]
        
        
class ContactSurface_ODE:
    ''' A visco-elastic planar surface
    '''
    def __init__(self, name, pos, normal, K, B, mu):
        self.name = name        # name of this contact surface
        self.x0 = pos           # position of a point of the surface
        self.normal = normal    # direction of the normal to the surface
        self.K = K              # stiffness of the surface material
        self.B = B              # damping of the surface material
        self.mu = mu            # friction coefficient of the surface
        self.bias = self.x0.dot(self.normal)
        
    def check_collision(self, p):
        ''' Check the collision of the given point
            with this contact surface. If the point is not
            inside this surface, then return False.
        '''
        normal_penetration = self.bias - p.dot(self.normal)
        if(normal_penetration < 0.0):
            return False # no penetration
        return True

    def compute_force(self, contact_point, q, vq, robot):
        cp = contact_point
        # cp.p0 anchor_point
        
        # get position of the contact point
        H = robot.framePlacement(q, cp.frame_id, False)
        p = H.translation

        # get velocity of the contact point
        R = pin.SE3(H.rotation, 0*H.translation)    # same as M but with translation set to zero
        v_local = robot.frameVelocity(q, vq, cp.frame_id, False)
        v = (R.act(v_local)).linear 

        # compute contact force using spring-damper law
        f = self.K.dot(cp.p0 - p) - self.B.dot(v)
        
        # check whether contact force is outside friction cone
        f_N = f.dot(self.normal)        # norm of normal force
        f_T = f - f_N*self.normal       # tangential force (3d)
        f_T_norm = norm(f_T)            # norm of tangential force
        
        if(f_T_norm > self.mu*f_N):
            # contact is slipping 
            # f_T_norm = self.mu*f_N
            t_dir = f_T/np.amax(np.absolute(f_T))
            t_dir = t_dir/norm(t_dir)
            
            # saturate force at the friction cone boundary
            f = f_N*self.normal + self.mu*f_N*t_dir
            # update anchor point so that f is inside friction cone
            if (f_T_norm >= 8.9e999):
                f_T_norm = 8.9e999
            delta_p0 = (f_T_norm - self.mu*f_N) / self.K[0,0]
            # print( "Delta_p0 : ", delta_p0, " - f_t_norm : ", f_T_norm.T, " - f_N_norm : ", norm(f_N))
            # print( "Delta_p0 : ", delta_p0, " - t_dir : ", t_dir.T)
            cp.p0 -= delta_p0*t_dir

            
        return f
        

class ODE:
    def __init__(self, name):
        self.name = name

    def f(self, x, u, t):
        return np.zeros(x.shape)


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

    def __init__(self, name, A, B, b):
        self.name = name
        self.A = A
        self.B = B
        self.b = b
        self.nx = A.shape[0]
        self.nu = B.shape[1]

    def f(self, x, u, t, jacobian=False):
        dx = self.A.dot(x) + self.b + self.B.dot(u)
        if(jacobian):
            return (np.copy(dx), np.copy(self.A), np.copy(self.B))
        return np.copy(dx)


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

        if(nv == 1):
            # for 1 DoF systems pin.aba does not work (I don't know why)
            pin.computeAllTerms(model, data, q, v)
            ddq = (u-data.nle) / data.M[0]
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
        f0 = self.f(x, u, t)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u, t)
            Fx[:, i] = (fp-f0)/delta
        return Fx

    def f_u_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed via finite differences '''
        f0 = self.f(x, u, t)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up, t)
            Fu[:, i] = (fp-f0)/delta
        return Fu


class ODERobot_wc:
    ''' An ordinary differential equation representing a robotic system
    '''

    def __init__(self, name, robot, contact_points, contact_surfaces):
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
        self.contact_points = contact_points
        self.contact_surfaces = contact_surfaces

        nk = 3*len(self.contact_points)*len(self.contact_surfaces)  # size of contact force vector
        self.fc = np.zeros(nk)                                      # contact forces
        self.Jc = np.zeros((nk, self.robot.model.nv))               # contact Jacobian

    ''' System dynamics '''

    def f(self, x, u, t, jacobian=False):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]

        i = 0
        self.robot.computeAllTerms(q,v)
        for cs in self.contact_surfaces:         # for each candidate contact surface
            for cp in self.contact_points:       # for each candidate contact point
                # Contact point placement
                H = self.robot.framePlacement(q, cp.frame_id, False)
                p_c = H.translation

                if(cs.check_collision(p_c)):      # check whether the point is colliding with the surface
                    if(cp.active == False):          # if the contact was not already active
                        cp.active = True
                        cp.p0 = np.copy(p_c)         # anchor point

                    # Compute the contact force
                    self.fc[i:i+3] = cs.compute_force(cp, q, v, self.robot)
                    # compute the jacobian
                    self.Jc[i:i+3, :] = cp.get_jacobian()
                    i += 3

                else:                           # if the point is not colliding more
                    if(cp.active):              # if the contact was already active
                        cp.active = False

                    # Contact force equal to 0
                    self.fc[i:i+3] = np.zeros(3)
                    # jacobian equl to zero
                    self.Jc[i:i+3, :] = np.zeros((3, self.robot.model.nv))
                    i += 3
        # compute JT*force from contact point
        u_con = u + self.Jc.T.dot(self.fc)

        if(nv == 1):
            # for 1 DoF systems pin.aba does not work (I don't know why)
            ddq = (u_con-data.nle) / data.M[0]
        else:
            ddq = pin.aba(model, data, q, v, u_con)

        self.dx[:nv] = v
        self.dx[nv:] = ddq

        if(jacobian):
            pin.computeABADerivatives(model, data, q, v, u_con)
            self.Fx[:nv, :nv] = 0.0
            self.Fx[:nv, nv:] = np.identity(nv)
            self.Fx[nv:, :nv] = data.ddq_dq
            self.Fx[nv:, nv:] = data.ddq_dv
            self.Fu[nv:, :] = data.Minv

            return (np.copy(self.dx), np.copy(self.Fx), np.copy(self.Fu))

        return np.copy(self.dx)

    def f_x_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. x computed via finite differences '''
        f0 = self.f(x, u, t)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u, t)
            Fx[:, i] = (fp-f0)/delta
        return Fx

    def f_u_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed via finite differences '''
        f0 = self.f(x, u, t)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up, t)
            Fu[:, i] = (fp-f0)/delta
        return Fu