import gymnasium as gym
import numpy as np
import control
import time

## Constants
lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
# state matrix
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, -mp*lp/mt*a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mk)))
B = np.array([[0], [(1-mp*lp*b/mt)/(mt)], [0], [b/mt]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])

dt = 0.02

x_hat = np.array([[0.011],
                  [0.045],
                  [-0.035584],
                  [0.4486]])
u = np.array([[0]])


# find  K and L based on targeted poles
# best 1
P1 = np.array([-0.7+0.68j, -0.7-0.68j, -1.18, -8.2])
P2 = np.array([-0.55+0.55j, -0.55-0.55j, -2.7815, -18.258
              ])

K = control.place(A, B, P2)
K = np.array(K)

L = control.place(np.transpose(A), np.transpose(C), 10*P1)
L = np.array(L)
L = np.transpose(L)
# K = control.place(A, B, [-0.72+0.74j, -0.72-0.74j, -1.5, -8.8])
# K = np.array(K)
# K = 2*K
# L = control.place(np.transpose(A), np.transpose(C), [-7.2+7.4j, -7.2-7.4j, -15, -88])
# L = np.array(L)
# L = np.transpose(L)



## Initialize environment
env = gym.make('CartPole-v1', render_mode='human')
(state, info) = env.reset(seed=1)
reward_total = 0


## Add something here
def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    print('K = ', K)
    print('x = ', x)
    print('u = ', u)
    # print(u)
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left


def apply_compensator(A, B, C, u, L, x_hat, y, delta_t):
    x_hat_dot = A @ x_hat + B @ u + L @ (y - C @ x_hat)
    x_hat = x_hat + x_hat_dot * delta_t
    print('delta_t :', delta_t)
    return x_hat


## Calculate performance
def perform_calc(y_array, t_array):
    y_max = np.amax(y_array)
    y_min = np.amin(y_array)
    y_ss = np.average(y_array[-101:-1] - y_min)

    # Overshoot (in Percentage)
    OV = ((y_max - y_min) / y_ss - 1) * 100
    if (OV < 1e-6): OV = 0

    # Peak Time
    peak_idx = np.argmax(y_array)
    peak_time = t_array[peak_idx]

    # Settling Time
    TH = 0.02  # 2 % Criterion
    y_set = 0
    n_set = 0
    n_prev = 0
    datapoints = len(t_array)
    for n in range(0, datapoints):
        if (np.abs(y_array[n, :] - y_min - y_ss) <= TH * y_ss) and (n_prev == 0):
            y_set = y_array[n, :]
            n_set = n
            n_prev = 1
        elif (np.abs(y_array[n, :] - y_min - y_ss) > TH * y_ss):
            n_prev = 0
    settling_time = t_array[n_set]

    return OV, peak_time, settling_time, y_min, y_max, y_ss

# Copy this for scoring
t_array = []
theta_array = []
force_array = []
T = 0
u = np.array([[0]])

for i in range(1000):
    env.render()

    # get the output only:
    y = np.array([[state[0]], [state[2]]])
    x_hat = apply_compensator(A, B, C, u, L, x_hat, y, dt)
    # get force direction (action) and force value (force)

    action, force = apply_state_controller(K, x_hat)
    u = np.resize(force, (1,1))

    print("sudah berada pada iterasi ke-", i)
    # print(state, '<-state')
    # print(x_hat, "<-x_hat")

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N

    force = np.float32(np.clip(force, -10, 10))

    # change magnitude of the applied force in CartPole
    env.env.force_mag = force
    # apply action
    state, reward, done, _, _ = env.step(action)
    # env.force_mag = abs_force
    reward_total = reward_total + reward

    # Copy this for scoring
    t_array.append(T)
    theta_array.append(state[2])
    T = T + dt

    time.sleep(dt)
    if i > 600:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break
    if done:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break

env.close()


theta_array = np.array(theta_array).reshape((len(theta_array), 1))
OV, peak_time, settling_time, y_min, y_max, y_ss = perform_calc(theta_array, np.array(t_array))
print('result is.....')
print('K = ', K)
print('OV =', OV)
print('peak time = ', peak_time)
print('settling_time = ', settling_time)
print('y_min = ', y_min)
print('y_max = ', y_max)
print('y_ss =', y_ss)
