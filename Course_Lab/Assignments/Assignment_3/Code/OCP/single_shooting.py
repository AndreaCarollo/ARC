# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from ode import ODERobot
from numerical_integration import Integrator
import time

class Empty:
    def __init__(self):
        pass
    

class SingleShootingProblem:
    ''' A simple solver for a single shooting OCP.
        In the current version, the solver considers only a cost function (no path constraints).
    '''
    
    def __init__(self, name, ode, x0, dt, N, integration_scheme, simu):
        self.name = name
        self.ode = ode
        self.integrator = Integrator('integrator')
        self.x0 = x0
        self.dt = dt
        self.N = N
        self.integration_scheme = integration_scheme
        self.simu = simu

        self.frame_id = self.simu.robot.model.getFrameId('tool0')
        
        self.nq = int(x0.shape[0]/2)
        self.nx = x0.shape[0]
        self.nu = self.ode.nu
        self.X = np.zeros((N, self.x0.shape[0]))
        self.U = np.zeros((N, self.nu))
        self.last_cost = 0.0
        self.running_costs = []
        self.final_costs = []

        self.visu = True
        
    # Add the running cost to the problem with its weight
    def add_running_cost(self, c, weight=1):
        self.running_costs += [(weight,c)]
    
    # Add the final cost to the problem with its weight
    def add_final_cost(self, c):
        self.final_costs += [c]
        
    # Compute the running cost of one step (with compute) and integrate it over the horizon
    def running_cost(self, X, U):
        ''' Compute the running cost integral '''
        cost = 0.0
        t = 0.0
        for i in range(U.shape[0]):         # Integration over the horizon
            for (w,c) in self.running_costs:
                cost += w * dt * c.compute(X[i,:], U[i,:], t, recompute=True)   # Computation
                t += self.dt
        return cost
    
    # Compute the running cost and its gradient of one step (with compute) and integrate it over the horizon
    def running_cost_w_gradient(self, X, U, dXdU):
        ''' Compute the running cost integral and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        t = 0.0
        nx, nu = self.nx, self.nu
        for i in range(U.shape[0]):         # Integration over the horizon
            for (w,c) in self.running_costs:
                
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)   # Computation
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[i*nu:(i+1)*nu] += ci_u
                
                cost += w * self.dt * ci
                grad += w * self.dt * dci
                t += self.dt

        return (cost, grad)
        
    # Compute the final cost that depends only by the final states x_N  
    def final_cost(self, x_N):
        ''' Compute the final cost '''
        cost = 0.0

        for c in self.final_costs:
            cost += c.compute(x_N, recompute=True)
        return cost
        
    # Compute the final cost and its gradient that depends only by the final states x_N
    def final_cost_w_gradient(self, x_N, dxN_dU):
        ''' Compute the final cost and its gradient w.r.t. U'''
        cost = 0.0

        grad = np.zeros(self.N*self.nu)
        for c in self.final_costs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU)
            cost += ci
            grad += dci
        return (cost, grad)
        
    # Compute the overall cost for a given input sequence y
    def compute_cost(self, y):
        ''' Compute cost function '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0, ndt = 0.0, 1

        # Integrate the dynamics finding the sequence of X = [x_1 x_2 x_3 ... x_N]
        X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                      self.N, self.integration_scheme)
        
        # compute cost
        run_cost = self.running_cost(X, U)
        fin_cost = self.final_cost(X[-1,:])
        cost = run_cost + fin_cost
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_cost = cost        
        return cost

    # Compute the overall cost and its gradient for a given input sequence y
            # The gradient is computed using finite diffrence
            #       grad = ( cost(y + eps) - cost(y) )/eps
    def compute_cost_w_gradient_fd(self, y):
        ''' Compute both the cost function and its gradient using finite differences '''
        eps = 1e-8
        y_eps = np.copy(y)
        grad = np.zeros_like(y)
        cost = self.compute_cost(y)     # Use the previous copute_cost funfion
        for i in range(y.shape[0]):
            y_eps[i] += eps
            cost_eps = self.compute_cost(y_eps)
            y_eps[i] = y[i]
            grad[i] = (cost_eps - cost) / eps
        return (cost, grad)
        
    # Compute the overall cost and its gradient for a given input sequence y
    def compute_cost_w_gradient(self, y):
        ''' Compute cost function and its gradient '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0 = 0.0

        # Integrate the dynamics finding the sequence of X = [x_1 x_2 x_3 ... x_N]
        # In the integration, it computes also the sensitivities
        X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                              self.dt, self.N, 
                                                              self.integration_scheme)
        
        # compute cost
        (run_cost, grad_run) = self.running_cost_w_gradient(X, U, dXdU)
        (fin_cost, grad_fin) = self.final_cost_w_gradient(X[-1,:], dXdU[-self.nx:,:])
        cost = run_cost + fin_cost
        grad = grad_run + grad_fin
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_cost = cost        
        return (cost, grad)
        
    # Solve the problem:
    def solve(self, y0=None, method='BFGS', use_finite_difference=False, max_iter_grad = 100, max_iter_fd = 100):
        ''' Solve the optimal control problem '''
        # Given an initial guess for the input...
        if(y0 is None):
            y0 = np.zeros(self.N*self.nu)
            
        self.iter = 0
        print('Start optimizing')
        # ... start to iterate using minimize function:
        #     minimize("compute cost function", 
        #              "initial guess for the input", 
        #              jac (?), 
        #              method (?),
        #              "callback fnc to show the motion at each iteration", 
        #              "options")
        if(use_finite_difference):
            r = minimize(self.compute_cost_w_gradient_fd, y0, jac=True, method=method, 
                     callback=self.clbk, options={'maxiter': max_iter_fd, 'disp': True})
        else:
            r = minimize(self.compute_cost_w_gradient, y0, jac=True, method=method, 
                     callback=self.clbk, options={'maxiter': max_iter_grad, 'disp': True})
        return r
        

    def sanity_check_cost_gradient(self, N_TESTS=10):
        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        for i in range(N_TESTS):
            y = np.random.rand(self.N*self.nu)
            (cost, grad_fd) = self.compute_cost_w_gradient_fd(y)
            (cost, grad) = self.compute_cost_w_gradient(y)
            grad_err = grad-grad_fd
            if(np.max(np.abs(grad_err))>1):
                print('Grad:   ', grad)
                print('Grad FD:', grad_fd)
            else:
                print('Everything is fine', np.max(np.abs(grad_err)))
        
    # Callback function to show the motion during the iteration
    def clbk(self, xk):
        print('Iter %3d, cost %5f'%(self.iter, self.last_cost))
        self.iter += 1
        if (self.iter%20 == 0 and self.visu ):
            self.display_motion()
        return False
        
    # Function that display the motion
    def display_motion(self, slow_down_factor=1):
        for i in range(0, self.N):
            time_start = time.time()
            q = self.X[i,:self.nq]
            self.simu.display(q)        
            time_spent = time.time() - time_start
            if(time_spent < slow_down_factor*self.dt):
                time.sleep(slow_down_factor*self.dt-time_spent)
        