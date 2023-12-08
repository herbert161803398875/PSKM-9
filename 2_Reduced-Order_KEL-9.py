import gymnasium as gym
import numpy as np
import control
import time

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
# state matrix

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs
a = g / (lp * (4.0 / 3 - mp / (mp + mk)))
# System State Space Equation
# state matrix
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


D = np.array([[0],
              [0]])

dt = 0.02

x_hat = np.array([[0],
                  [0],
                  [0],
                  [0]])

print(A)
print('B = ', B)
# System State Space Equation
A = np.array([[0, 1, 0, 0],
              [0, 0, -mp * (mp * (g - l) + mc * g) / ((mc + mp) * ((4 / 3) * mc + (1 / 3) * mp)), 0],
              [0, 0, 0, 1],
              [0, 0, (mp * (g - l) + mc * g) / (l * ((4 / 3) * mc + (1 / 3) * mp)), 0]])
print(' A :', A)
B = np.array([[0],
              [(1 / (mc + mp) - mp / ((mc + mp) * ((4 / 3) * mc + (1 / 3) * mp)))],
              [0],
              [(-1 / (l * ((4 / 3) * mc + (1 / 3) * mp)))]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# BUAT SISTEM STATE SPACE BARU
Ar = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, -0.71707317, 0, 0],
               [0, 15.77560976, 0, 0]])

Br = np.array([[0],
               [0],
               [0.97560976],
               [-1.46341463]])
Aaa = np.array([[0, 0],
                [0, 0]])
Aab = np.array([[1, 0],
                [0, 1]])
Aba = np.array([[0, -0.71374723],
                [0, 15.70243902]])
Abb = np.array([[0, 0],
                [0, 0]])
Ba = np.array([[0],
               [0]])
Bb = np.array([[0.84257206],
               [-1.46341463]])
#
P2 = np.array([-1.15+1.1j, -1.15-1.1j, -3, -18
              ])

K = control.place(A, B, P2)
K = np.array(K)
# K = control.place(A, B, [-0.68+0.72j, -0.68-0.72j, -1.4, -8.2])
# K = np.array(K)
# K =K*2


L = control.place(np.transpose(Abb), np.transpose(Aab), [-22-20j, -22+20j])
L = np.array(L)
L = np.transpose(L)
# # desired poles
# P = np.array([-0.25+0.25j, -0.25-0.25j, -21+0.25j, -21-0.25j])
# Pt = 4 * P[:2]
#
# # compute regulator and observer gains
# K = control.place(A, B, P)
#
# L = control.place(np.transpose(Abb), np.transpose(Aab), Pt)
# L = np.transpose(L)
# print('K = ', K)
# print('L = ', L)

# misal
xb_hat = np.array([[0], [0]])
u_array = []
theta_array = []
t_array = []
T = 0


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


def estimator_reduced_order(state, x_hat, y, xb_hat, u):
    statepy = np.array([state[0], state[2]])
    statepy = statepy.reshape(-1, 1)

    x_hatpy = np.array([x_hat[1], x_hat[3]])

    xb_hat_dot = (Abb - L @ Aab) @ x_hatpy + (Aba - L @ Aaa) @ y + (Bb - L @ Ba) @ u
    print('xcc awal', xb_hat)
    print('xb_hat_dot * dt = ', xb_hat_dot * dt)
    xb_hat = xb_hat + xb_hat_dot * dt
    print('nilai dt = ', dt)

    xc = xb_hat + L @ y

    x_hat_new = np.array([[0], [0], [0], [0]])

    # x_hat_new[:2] = statepy
    # x_hat_new[2:] = xc
    # x_hat_new[[2, 1]] = x_hat_new[[1, 2]]
    x_hat_new = np.concatenate((statepy, xc))
    x_hat_new[[2, 1]] = x_hat_new[[1, 2]]
    print('statepy : ', statepy)
    print('x_hatpy : ', x_hatpy)
    print('xb_hat_dot : ', xb_hat_dot)
    print('xb_hat : ', xb_hat)
    print('xc : ', xc)
    print('x_hat_new : ', x_hat_new)
    return xb_hat, x_hat_new


def apply_state_controller(K, x):
    # feedback controller
    print('K = ', K)
    print('x = ', x)
    u = -np.dot(K, x)  # u = -Kx

    print('u = ', u)
    # print(u)
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left


## Initialize environment
env = gym.make('CartPole-v1', render_mode='human')
(state, info) = env.reset(seed=1)
print('state:', state)
reward_total = 0

u = np.array([[0]])
force = u
y = np.array([[state[0]], [state[2]]])
print('y sblm render = ', y)

# Copy this for scoring

force_array = []

u = np.array([[0]])

for i in range(1000):
    env.render()
    print("sudah berada pada iterasi ke-", i)
    theta_array.append(state[2])
    # get the output only:

    action, force = apply_state_controller(K, x_hat)
    print("u:", force)
    print('force = ', force)
    force = np.clip(force, -10, 10)
    u = force
    force = np.abs(float(force))
    env.force_mag = force

    state, reward, done, _, _ = env.step(action)
    y = np.array([[state[0]], [state[2]]])
    print('y = ', y)
    xb_hat, x_hat = estimator_reduced_order(state, x_hat, y, xb_hat, u)
    print('x_hat = ', x_hat)
    print('xb_hat = ', xb_hat)

    # force = np.resize(force, (1,1))
    print('u sini = ', u)
    print('force = ', force)

    # print(state, '<-state')
    # print(x_hat, "<-x_hat")

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N

    # force = np.abs(float(force))
    print('absforce = ', force)


    # change magnitude of the applied force in CartPole
    # env.env.force_mag = abs_force
    # apply action
    # env.force_mag = abs_force
    reward_total = reward_total + reward

    # Copy this for scoring
    t_array.append(T)
    theta_array.append(state[2])
    T = i*dt

    time.sleep(dt)
    if i > 998:
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
print('L = ', L)
print('OV =', OV)
print('peak time = ', peak_time)
print('settling_time = ', settling_time)
print('y_min = ', y_min)
print('y_max = ', y_max)
print('y_ss =', y_ss)
