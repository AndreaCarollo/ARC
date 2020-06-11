# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Classes representing different kind of Ordinary Differential Equations (ODEs).

@author: student
"""

import numpy as np
import pinocchio as pin
from utils.robot_simulator import ContactPoint, ContactSurface



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

    # def f(self, x, u, t, jacobian=False):
    #     nq = self.robot.nq
    #     nv = self.robot.nv
    #     model = self.robot.model
    #     data = self.robot.data
    #     q = x[:nq]
    #     v = x[nq:]

    #     i = 0
    #     for s in self.contact_surfaces:         # for each candidate contact surface
    #         for cp in self.contact_points:      # for each candidate contact point
    #             # Contact point placement
    #             H = self.robot.framePlacement(q, cp.frame_id, True)
    #             p = H.translation
    #             if(s.check_collision(p)):       # check whether the point is colliding with the surface
    #                 if(not cp.active):          # if the contact was not already active
    #                     # print("Collision detected between point", cp.frame_name, " at ", p)
    #                     cp.active = True
    #                 self.fc[i:i+3], p = s.compute_force(cp, p)  # Compute the contact force
    #                 self.Jc[i:i+3, :] = cp.get_jacobian()       # compute the jacobian
    #                 i += 3
    #                 # print('contact force --> ', np.copy(self.Jc[i:i+3, 1]).T)

    #             else:                           # if the point is not colliding more
    #                 if(cp.active):              # if the contact was already active
    #                     # print("Contact lost between point", cp.frame_name, " at ", p)
    #                     cp.active = False
    #                 self.fc[i:i+3] = np.zeros(3)                # Contact force equal to 0
    #                 self.Jc[i:i+3,:] = np.zeros((3, self.robot.model.nv)) # jacobian equl to zero
    #                 i += 3
                
    #     # compute JT*force from contact point
    #     u_con = u + self.Jc.T.dot(self.fc)
            

    #     if(nv == 1):
    #         # for 1 DoF systems pin.aba does not work (I don't know why)
    #         # pin.computeAllTerms(model, data, q, v)
    #         ddq = (u_con-data.nle) / data.M[0]
    #     else:
    #         ddq = pin.aba(model, data, q, v, u_con)

    #     self.dx[:nv] = v
    #     self.dx[nv:] = ddq

    #     if(jacobian):
    #         pin.computeABADerivatives(model, data, q, v, u_con)
    #         self.Fx[:nv, :nv] = 0.0
    #         self.Fx[:nv, nv:] = np.identity(nv)
    #         self.Fx[nv:, :nv] = data.ddq_dq
    #         self.Fx[nv:, nv:] = data.ddq_dv
    #         self.Fu[nv:, :] = data.Minv

    #         return (np.copy(self.dx), np.copy(self.Fx), np.copy(self.Fu))

    #     return np.copy(self.dx)

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
                        # print("Collision detected between point", cp.frame_name)
                        cp.active = True
                        cp.p0 = np.copy(p_c)         # anchor point

                    # Compute the contact force
                    self.fc[i:i+3] = cs.compute_force2(cp, q, v, self.robot)
                    # compute the jacobian
                    self.Jc[i:i+3, :] = cp.get_jacobian()
                    i += 3
                    #print('contact force --> ', np.copy(self.Jc[i:i+3, :]))

                else:                           # if the point is not colliding more
                    if(cp.active):              # if the contact was already active
                        # print("Contact lost between point", cp.frame_name)
                        cp.active = False
                        # cp.p0 = np.zeros(3)
                    # Contact force equal to 0
                    self.fc[i:i+3] = np.zeros(3)
                    # jacobian equl to zero
                    self.Jc[i:i+3, :] = np.zeros((3, self.robot.model.nv))
                    i += 3
        # compute JT*force from contact point
        u_con = u + self.Jc.T.dot(self.fc)

        if(nv == 1):
            # for 1 DoF systems pin.aba does not work (I don't know why)
            # pin.computeAllTerms(model, data, q, v)
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


if __name__ == '__main__':
    from utils.robot_loaders import loadUR, loadPendulum, loadUR_urdf
    from example_robot_data.robots_loader import loadDoublePendulum
    from utils.robot_wrapper import RobotWrapper
    import single_shooting_conf as conf
    np.set_printoptions(precision=3, linewidth=200, suppress=True)

    dt = conf.dt                 # time step
    system = conf.system
    N_TESTS = 10

    if(system == 'ur'):
        r = loadUR_urdf()
    elif(system == 'double-pendulum'):
        r = loadDoublePendulum()
    elif(system == 'pendulum'):
        r = loadPendulum()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    n = nq+nv                       # state size
    m = robot.na                    # control size
    ode = ODERobot('ode', robot)
    t = 0.0

    for i in range(N_TESTS):
        x = np.random.rand(n)
        u = np.random.rand(m)
        (dx, Fx, Fu) = ode.f(x, u, t, jacobian=True)
        Fx_fd = ode.f_x_fin_diff(x, u, t)
        Fu_fd = ode.f_u_fin_diff(x, u, t)

        Fx_err = Fx-Fx_fd
        Fu_err = Fu-Fu_fd
        if(np.max(np.abs(Fx_err)) > 1e-4):
            print('Fx:   ', Fx)
            print('Fx FD:', Fx_fd)
        else:
            print('Fx is fine', np.max(np.abs(Fx_err)))

        if(np.max(np.abs(Fu_err)) > 1e-4):
            print('Fu:   ', Fu)
            print('Fu FD:', Fu_fd)
        else:
            print('Fu is fine', np.max(np.abs(Fu_err)))
