import gymnasium as gym
import numpy as np
import scipy as sc
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

# K = np.array([-10.0000,  -24.8801, -265.0991, -119.4685])
# R = np.eye(1, dtype=int)          # choose R (weight for input)
# Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
R = [[1]]
Q = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])


# solve ricatti equation
P = sc.linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),np.dot(B.T, P))


# get environment
env = gym.make('CartPole-v1', render_mode='human')
(obs, info) = env.reset(seed=1)
reward_total = 0


# ADD SOMETHING HERE


def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -np.dot(K, x)  # u = -Kx
    u = u.item()
    print('U=', u)
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left

# Copy this for scoring
t_array = []
theta_array = []
T = 0
dt = 0.02

for i in range(1000):
    env.render()

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)
    print(force, "----", action)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _, _ = env.step(action)

    reward_total = reward_total + reward
    # Copy this for scoring
    t_array.append(T)
    theta_array.append(obs[2])
    T = T + dt
    if reward_total > 998:
        print(reward_total)
        u = np.array([force])
        print(u)
        break
    if done:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break

env.close()

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


theta_array = np.array(theta_array).reshape((len(theta_array), 1))
OV, peak_time, settling_time, y_min, y_max, y_ss = perform_calc(theta_array, np.array(t_array))
print('result is.....')
print('OV =', OV)
print('K = ', K)
print('R = ' , R)
print(' Q = ' ,  Q)
print('y_ss =', y_ss)
print('peak time = ', peak_time)
print('settling_time = ', settling_time)
print('y_min = ', y_min)
print('y_max = ', y_max)


