from __future__ import division
from sympy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x, y, z = symbols('x y z')
#k, m, n = symbols('k m n', integer=True)
#f, g, h = symbols('f g h', cls=Function)

trace = pd.read_csv("0.csv", names = ['x', 'y', 'z'])

lon = [None]*3
lon[0] = x
lon[1] = x+np.sqrt(3)*y
lon[2] = x-np.sqrt(3)*y

lat = [None]*2
lat[0] = z+0.5
lat[1] = z-0.5

sphere = x**2+y**2+z**2-1

#vp = [0,1,0]
#vp = [1, 0, 0]
#vp = [0.5, sqrt(0.5), 0]
#vp = [np.cos(np.pi/10000), np.sin(np.pi/10000), 0]
#vp = [np.cos(np.pi/10), np.sin(np.pi/10), 0]
#vp = [np.cos(np.pi/6), np.sin(np.pi/6), 0]
#vp = [np.cos(np.pi/5), np.sin(np.pi/5), 0]
#vp = [np.cos(np.pi/4), np.sin(np.pi/4), 0]
#vp = [np.cos(0)*np.cos(np.pi/2.), np.sin(0)*np.cos(np.pi/2.), np.sin(np.pi/2.)]
#vp = [np.cos(np.pi/6)*np.cos(np.pi/3), np.sin(np.pi/6)*np.cos(np.pi/3), np.sin(np.pi/3)]
#vp = [np.cos(np.pi/5)*np.cos(np.pi/5), np.sin(np.pi/5)*np.cos(np.pi/5), np.sin(np.pi/5)]

output = [[[0 for _ in range(3)] for _ in range(6)] for _ in range(trace.shape[0])]

for i in range(len(trace)):
    vp = [trace.iloc[i, 0], trace.iloc[i, 1], trace.iloc[i, 2]]
    vp = np.around(vp, 6)
    print(vp)
    full = (60-0.01)/180*np.pi
    near = (25-0.01)/180*np.pi
    norm_full = np.cos(full)
    norm_near = np.cos(near)
    full_fov = vp[0]*(x-vp[0]*norm_full) + \
               vp[1]*(y-vp[1]*norm_full) + \
               vp[2]*(z-vp[2]*norm_full)
    near_fov = vp[0]*(x-vp[0]*norm_near) + \
               vp[1]*(y-vp[1]*norm_near) + \
               vp[2]*(z-vp[2]*norm_near)
    print(full_fov)


    fov = [full_fov, near_fov]
    for w in range(1,3):
        for l1 in lon:
            sol = solve([sphere, l1, fov[w-1]])
            for s in sol:
                if not sum(s.values()).is_real:
                    continue
                if s[x] == 0:
                    if s[y] < 0:
                        if s[z] < -0.5:
                            output[i][5][0] = w
                            output[i][0][0] = w
                        elif s[z] > 0.5:
                            output[i][5][2] = w
                            output[i][0][2] = w
                        else:
                            output[i][5][1] = w
                            output[i][0][1] = w
                    if s[y] > 0:
                        if s[z] < -0.5:
                            output[i][2][0] = w
                            output[i][3][0] = w
                        elif s[z] > 0.5:
                            output[i][2][2] = w
                            output[i][3][2] = w
                        else:
                            output[i][2][1] = w
                            output[i][3][1] = w

                if s[x] > 0:
                    if s[y] < 0:
                        if s[z] < -0.5:
                            output[i][0][0] = w
                            output[i][1][0] = w
                        elif s[z] > 0.5:
                            output[i][0][2] = w
                            output[i][1][2] = w
                        else:
                            output[i][0][1] = w
                            output[i][1][1] = w
                    if s[y] > 0:
                        if s[z] < -0.5:
                            output[i][1][0] = w
                            output[i][2][0] = w
                        elif s[z] > 0.5:
                            output[i][1][2] = w
                            output[i][2][2] = w
                        else:
                            output[i][1][1] = w
                            output[i][2][1] = w
                if s[x] < 0:
                    if s[y] > 0:
                        if s[z] < -0.5:
                            output[i][3][0] = w
                            output[i][4][0] = w
                        elif s[z] > 0.5:
                            output[i][3][2] = w
                            output[i][4][2] = w
                        else:
                            output[i][3][1] = w
                            output[i][4][1] = w
                    if s[y] < 0:
                        if s[z] < -0.5:
                            output[i][4][0] = w
                            output[i][5][0] = w
                        elif s[z] > 0.5:
                            output[i][4][2] = w
                            output[i][5][2] = w
                        else:
                            output[i][4][1] = w
                            output[i][5][1] = w

        #print(f"1output[{i}] = ", output[i])
        
        for l2 in lat:
            sol = solve([sphere, l2, fov[w-1]])
            for s in sol:
                if not sum(s.values()).is_real:
                    continue
                if s[z] < 0:
                    if s[x] > 0:
                        if s[y] < -0.5:
                            output[i][0][0] = w
                            output[i][0][1] = w
                        elif s[y] > 0.5:
                            output[i][2][0] = w
                            output[i][2][1] = w
                        else:
                            output[i][1][0] = w
                            output[i][1][1] = w
                    if s[x] < 0:
                        if s[y] < -0.5:
                            output[i][5][0] = w
                            output[i][5][1] = w
                        elif s[y] > 0.5:
                            output[i][3][0] = w
                            output[i][3][1] = w
                        else:
                            output[i][4][0] = w
                            output[i][4][1] = w
                if s[z] > 0:
                    if s[x] > 0:
                        if s[y] < -0.5:
                            output[i][0][1] = w
                            output[i][0][2] = w
                        elif s[y] > 0.5:
                            output[i][2][1] = w
                            output[i][2][2] = w
                        else:
                            output[i][1][1] = w
                            output[i][1][2] = w
                    if s[x] < 0:
                        if s[y] < -0.5:
                            output[i][5][1] = w
                            output[i][5][2] = w
                        elif s[y] > 0.5:
                            output[i][3][1] = w
                            output[i][3][2] = w
                        else:
                            output[i][4][1] = w
                            output[i][4][2] = w
        #print(f"2output[{i}] = ", output[i])

    if vp[0] > 0:
        if vp[1] < -0.5:
            if vp[2] < -0.5:
                output[i][0][0] = 2
            elif vp[2] > 0.5:
                output[i][0][2] = 2
            else:
                output[i][0][1] = 2
        elif vp[1] > 0.5:
            if vp[2] < -0.5:
                output[i][2][0] = 2
            elif vp[2] > 0.5:
                output[i][2][2] = 2
            else:
                output[i][2][1] = 2
        else:
            if vp[2] < -0.5:
                output[i][1][0] = 2
            elif vp[2] > 0.5:
                output[i][1][2] = 2
            else:
                output[i][1][1] = 2
    elif vp[0] < 0:
        if vp[1] < -0.5:
            if vp[2] < -0.5:
                output[i][5][0] = 2
            elif vp[2] > 0.5:
                output[i][5][2] = 2
            else:
                output[i][5][1] = 2
        elif vp[1] > 0.5:
            if vp[2] < -0.5:
                output[i][3][0] = 2
            elif vp[2] > 0.5:
                output[i][3][2] = 2
            else:
                output[i][3][1] = 2
        else:
            if vp[2] < -0.5:
                output[i][4][0] = 2
            elif vp[2] > 0.5:
                output[i][4][2] = 2
            else:
                output[i][4][1] = 2
    
    print(f"output[{i}] = ", output[i])

weights = [0 for _ in range(len(trace))]

for i in range(len(trace)):
    weights[i] = sum(map(lambda x: x**2, np.concatenate(output[i])))

print(weights)

plt.hist(weights)
plt.show()
