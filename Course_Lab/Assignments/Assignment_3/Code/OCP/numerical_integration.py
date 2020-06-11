# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
import pinocchio as pin


class Integrator:
    ''' A class implementing different numerical integrator schemes '''

    def __init__(self, name):
        self.name = name

    def integrate(self, ode, x_init, U, t_init, dt, ndt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory:
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - ndt: the number of inner time steps for each time step
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        n = x_init.shape[0]
        t = np.zeros((N*ndt+1))*np.nan
        x = np.zeros((N*ndt+1, n))*np.nan
        dx = np.zeros((N*ndt, n))*np.nan
        h = dt/ndt  # inner time step
        x[0, :] = x_init
        t[0] = t_init

        if(scheme == 'RK-1'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                f = ode.f(x[i, :], U[ii, :], t[i])
                dx[i, :] = f
                x[i+1, :] = x[i, :] + h*f
                t[i+1] = t[i] + h
        elif(scheme == 'RK-2'):   # explicit midpoint method
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,            U[ii, :], t[i])
                k2 = ode.f(x0 + 0.5*h*k1, U[ii, :], t[i]+0.5*h)
                dx[i, :] = k2
                x[i+1, :] = x0 + h*k2
                t[i+1] = t[i] + h
        elif(scheme == 'RK-2-Heun'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,        U[ii, :], t[i])
                k2 = ode.f(x0 + h*k1, U[ii, :], t[i]+h)
                dx[i, :] = 0.5*(k1+k2)
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        elif(scheme == 'RK-3'):  # Kutta's third-order method
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,                  U[ii, :], t[i])
                k2 = ode.f(x0 + h*0.5*k1,       U[ii, :], t[i]+0.5*h)
                k3 = ode.f(x0 + h*(-k1 + 2*k2), U[ii, :], t[i]+h)
                dx[i, :] = (k1 + 4*k2 + k3)/6.0
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        elif(scheme == 'RK-4'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,            U[ii, :], t[i])
                k2 = ode.f(x0 + 0.5*h*k1, U[ii, :], t[i]+0.5*h)
                k3 = ode.f(x0 + 0.5*h*k2, U[ii, :], t[i]+0.5*h)
                k4 = ode.f(x0 + h * k3,   U[ii, :], t[i]+h)
                dx[i, :] = (k1 + 2*k2 + 2*k3 + k4)/6.0
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        self.dx = dx
        self.t = t
        self.x = x
        return x[::ndt, :]

    def integrate_w_sensitivities_u(self, ode, x_init, U, t_init, dt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory.
            Compute also the derivative of the x trajectory w.r.t. U.
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        nx = x_init.shape[0]
        nu = ode.nu
        t = np.zeros((N+1))*np.nan
        x = np.zeros((N+1, nx))*np.nan
        dx = np.zeros((N+1, nx))*np.nan
        dXdU = np.zeros(((N+1)*nx, N*nu))
        h = dt
        x[0, :] = x_init
        t[0] = t_init

        I = np.identity(nx)
        if(scheme == 'RK-1'):
            for i in range(N):
                (f, f_x, f_u) = ode.f(x[i, :], U[i, :], t[i], jacobian=True)
                dx[i, :] = f
                x[i+1, :] = x[i, :] + h*f
                t[i+1] = t[i] + h

                phi_x = I + h*f_x
                phi_u = h * f_u
                ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
                iu, iu1 = i*nu, (i+1)*nu
                dXdU[ix1:ix2, :] = phi_x.dot(dXdU[ix:ix1, :])
                dXdU[ix1:ix2, iu:iu1] += phi_u
        elif(scheme == 'RK-4'):
            for i in range(x.shape[0]-1):
                x1 = x[i, :]
                t1 = t[i]
                (k1, f1_x, f1_u) = ode.f(x1, U[i, :], t1, jacobian=True)
                k1_x = f1_x
                k1_u = f1_u

                x2 = x1 + 0.5*h*k1
                t2 = t[i]+0.5*h
                (k2, f2_x, f2_u) = ode.f(x2, U[i, :], t2, jacobian=True)
                k2_x = f2_x.dot(I + 0.5*h*k1_x)
                k2_u = f2_u + 0.5*h*f2_x @ k1_u

                x3 = x1 + 0.5*h*k2
                t3 = t[i]+0.5*h
                (k3, f3_x, f3_u) = ode.f(x3, U[i, :], t3, jacobian=True)
                k3_x = f3_x.dot(I + 0.5*h*k2_x)
                k3_u = f3_u + 0.5*h*f3_x @ k2_u

                x4 = x1 + h * k3
                t4 = t[i]+h
                (k4, f4_x, f4_u) = ode.f(x4, U[i, :], t4, jacobian=True)
                k4_x = f4_x.dot(I + h*k3_x)
                k4_u = f4_u + h*f4_x @ k3_u

                dx[i, :] = (k1 + 2*k2 + 2*k3 + k4)/6.0
                x[i+1, :] = x1 + h*dx[i, :]
                t[i+1] = t[i] + h

                phi_x = I + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
                phi_u = h*(k1_u + 2*k2_u + 2*k3_u + k4_u)/6.0
                ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
                iu, iu1 = i*nu, (i+1)*nu
                dXdU[ix1:ix2, :] = phi_x.dot(dXdU[ix:ix1, :])
                dXdU[ix1:ix2, iu:iu1] += phi_u
        else:
            return None
        self.dx = dx
        self.t = t
        self.x = x
        return (x, dXdU)

    def check_sensitivities_u(self, ode, x_init, t_init, dt, N, scheme, N_TESTS=10):
        eps = 1e-8
        nx = x_init.shape[0]
        nu = ode.nu
        for iii in range(N_TESTS):
            U = np.random.rand(N, nu)
            (X, dXdU) = self.integrate_w_sensitivities_u(
                ode, x_init, U, t_init, dt, N, scheme)
            X = X.reshape(X.shape[0]*X.shape[1])
            dXdU_fd = np.zeros(((N+1)*nx, N*nu))
            for i in range(N):
                for j in range(nu):
                    U_bar = np.copy(U)
                    U_bar[i, j] += eps
                    X_bar = self.integrate(
                        ode, x_init, U_bar, t_init, dt, 1, N, scheme)
                    X_bar = X_bar.reshape(X_bar.shape[0]*X_bar.shape[1])
                    dXdU_fd[:, i*nu+j] = (X_bar-X)/eps
            dXdU_err = dXdU - dXdU_fd

            print("Error in sensitivities", np.max(np.abs(dXdU_err)))
            if(np.max(np.abs(dXdU_err)) > np.sqrt(eps)):
                print("dXdU", dXdU)
                print("dXdU_fd", dXdU_fd)

