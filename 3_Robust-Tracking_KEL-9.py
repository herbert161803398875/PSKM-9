import gymnasium as gym
import numpy as np
import control
import time
from scipy import integrate

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, -mp*lp/mt*a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

b = -1/(lp*(4.0/3 - mp/(mp+mk)))

B = np.array([[0], [(1-mp*lp*b/mt)/(mt)], [0], [b/mt]])
C = np.array([[1, 0, 0, 0]])
D = np.array([[0],
              [0]])
dt = 0.02
Arobust =  np.block([[np.zeros((C.shape[0], 1)), C],
                            [np.zeros((A.shape[0], 1)), A]])
Brobust = np.block([[np.zeros([C.shape[0],1])],
                  [B]])
Bl = Brobust
w = np.array([1])#GANGGUANNYA
w = np.reshape(w,1)

P = np.array([-0.7+0.68j, -0.7-0.68j, -1.18, -8.2])
Probust =np.array([-0.7+0.68j, -0.7-0.68j, -1.18, -8.2,-121])

K = control.place(A,B,P)
Krobust = control.place(Arobust, Brobust, Probust)

## Initialize environment
env = gym.make('CartPole-v1', render_mode='human')
(state, info) = env.reset(seed=1)
reward_total = 0

def fungsirobust(x, u):
    Arobust_dot = Arobust@x + Brobust@u + Bl@w
    return Arobust_dot
    
def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    # print(u)
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left
r=0
for i in range(1000):
    env.render()

    # get the output only:
    y = np.array([[state[0]], [state[2]]])
    x = np.transpose(np.array([state]))
    xl_dot=C@x-r
    augmentedx=np.concatenate((xl_dot,x))
    # get force direction (action) and force value (force)

    action, force = apply_state_controller(Krobust, augmentedx)
    u = np.resize(force, (1,1))

    print("sudah berada pada iterasi ke-", i)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force =abs(float(np.clip(force, -10, 10)))

    # change magnitude of the applied force in CartPole
    env.env.force_mag = abs_force
    # apply action
    state, reward, done, _, _ = env.step(action)
    # env.force_mag = abs_force
    reward_total = reward_total + reward
    augmentedx = augmentedx + fungsirobust(augmentedx,force) * dt
    time.sleep(dt)
    if i > 300:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break
    if done:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break

env.close()