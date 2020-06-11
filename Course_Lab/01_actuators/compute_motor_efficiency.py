from __future__ import print_function
from dc_motor_w_coulomb_friction_complete import MotorCoulomb
from dc_motor_complete import get_motor_parameters
import utils.plot_utils
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=200, suppress=True)


class Empty:
    def __init__(self):
        pass


def compute_efficiency(V, motor, dq_max):
    state = np.zeros(2)             # motor state (pos, vel)
    res = Empty()
    # create a vector with values ranging from 0 to dq_max with a step of 0.01
    res.dq = np.arange(0.0, dq_max, 0.01)
    N = res.dq.shape[0]
    res.tau_f = np.zeros(N)         # friction torque (Coulomb+viscous)
    res.tau = np.zeros(N)           # output torque (motor+friction)
    res.P_m = np.zeros(N)           # mechanical output power
    res.P_e = np.zeros(N)           # electrical input power
    res.efficiency = np.zeros(N)    # motor efficiency (out/in power)

    for i in range(N):
        # set velocity
        state[1] = res.dq[i]
        motor.set_state(state)

        # apply constant voltage
        motor.simulate_voltage(V, method='time-stepping')

        # compute motor efficiency
        res.tau_f[i] = motor.tau_f + motor.b * res.dq[i]
        res.tau[i]   = motor.tau()
        res.P_m[i]   = (res.tau[i] - res.tau_f[i] )* res.dq[i]
        # res.P_e[i] = motor.i() * motor.V()
        res.P_e[i]   = (motor.tau())*V/motor.K_b

        res.efficiency[i] = res.P_m[i] / res.P_e[i]

    i = np.argmax(res.efficiency)
    print("Max efficiency", res.efficiency[i])
    print("reached at velocity", res.dq[i], "and torque", res.tau[i])
    return res


V = 48                      # input voltage
dt = 1e-3                   # time step
params = get_motor_parameters('Maxon148877')

motor = MotorCoulomb(dt, params)

# compute maximum motor vel for given voltage
# V = R i + K_b dq --> dq_max = V / K_b  , considering i_min = 0
# dq_max = (V - motor.R * params.i_0) / motor.K_b
dq_max = V / motor.K_b
print("Max velocity",dq_max)


res = compute_efficiency(V, motor, dq_max)

def plot_stuff(res):
    f, ax = plt.subplots(1, 1, sharex=True)
    alpha = 0.8
    ax.plot(res.tau, res.dq, label='dq-tau', alpha=alpha)
    ax.plot(res.tau, res.P_m, label='P_m', alpha=alpha)
    # ax.plot(res.tau, res.P_e, label ='P_e', alpha=alpha)
    dq_max = np.max(res.dq)
    ax.plot(res.tau, res.efficiency * dq_max,
            label='efficiency (scaled)', alpha=alpha)
    ax.legend()
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Velocity [rad/s]')
    plt.ylim([0, dq_max])


plot_stuff(res)

f, ax = plt.subplots(1, 1, sharex=True)
alpha = 0.8
ax.plot(res.tau, res.efficiency, label='efficiency', alpha=alpha)
ax.legend()
plt.xlabel('Torque [Nm]')
plt.ylabel('Efficiency')
plt.ylim([0, 1])
plt.show()
