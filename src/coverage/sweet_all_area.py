import numpy as np
import matplotlib.pyplot as plt

def xyz_to_erp(xyz):
    if xyz[1] > 0:
        if xyz[0] == 0:
            lon = np.pi / 2
        elif xyz[0] > 0:
            lon = np.arctan(xyz[1]/xyz[0])
        elif xyz[0] < 0:
            lon = np.pi + np.arctan(xyz[1]/xyz[0])
    elif xyz[1] < 0:
        if xyz[0] == 0:
            lon = np.pi + np.pi / 2
        elif xyz[0] > 0:
            lon = np.arctan(xyz[1]/xyz[0]) + 2*np.pi
        elif xyz[0] < 0:
            lon = np.arctan(xyz[1]/xyz[0]) + np.pi
    else:
        if xyz[0] > 0:
            lon = 0
        elif xyz[0] < 0:
            lon  = np.pi
        else:
            lon = 0
    
    xy = np.sqrt(xyz[0]**2 + xyz[1]**2)
    if xy == 0:
        if xyz[2] > 0:
            lat = np.pi / 2
        else:
            lat = -np.pi / 2
    else:
        lat = np.arctan(xyz[2] / xy)

    return lon, lat

def erp_to_xyz(erp):
    x = np.cos(erp[0]) * np.cos(erp[1])
    y = np.sin(erp[0]) * np.cos(erp[1])
    z = np.sin(erp[1])
    return x, y, z

'''
xyz = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.),\
        (0.5, np.sqrt(3)/2, 0.),\
        (0.5, 0., np.sqrt(3)/2),\
        (0., 0.5, np.sqrt(3)/2),\
        (0., np.sqrt(3)/2, 0.5),\
        (np.sqrt(3)/2, 0., 0.5),\
        (np.sqrt(3)/2, 0.5, 0.),\
        (0.5, 0.5, 1/np.sqrt(2))]

_xyz = [(-1., -0., -0.), (-0., -1., -0.), (-0., -0., -1.),\
        (-0.5, -np.sqrt(3)/2, -0.),\
        (-0.5, -0., -np.sqrt(3)/2),\
        (-0., -0.5, -np.sqrt(3)/2),\
        (-0., -np.sqrt(3)/2, -0.5),\
        (-np.sqrt(3)/2, -0., -0.5),\
        (-np.sqrt(3)/2, -0.5, -0.),\
        (-0.5, -0.5, -1/np.sqrt(2))]

__xyz = [(-1., 0., 0.), (0., 1., 0.), (0., 0., 1.),\
        (-0.5, np.sqrt(3)/2, 0.),\
        (-0.5, 0., np.sqrt(3)/2),\
        (0., 0.5, np.sqrt(3)/2),\
        (0., np.sqrt(3)/2, 0.5),\
        (-np.sqrt(3)/2, 0., 0.5),\
        (-np.sqrt(3)/2, 0.5, 0.),\
        (-0.5, 0.5, 1/np.sqrt(2))]

___xyz = [(1., 0., 0.), (0., -1., 0.), (0., 0., 1.),\
        (0.5, -np.sqrt(3)/2, 0.),\
        (0.5, 0., np.sqrt(3)/2),\
        (0., -0.5, np.sqrt(3)/2),\
        (0., -np.sqrt(3)/2, 0.5),\
        (np.sqrt(3)/2, 0., 0.5),\
        (np.sqrt(3)/2, -0.5, 0.),\
        (0.5, -0.5, 1/np.sqrt(2))]

erp = [(0., 0.), (np.pi/2, 0.), (0., np.pi/2),\
        (np.pi/3, 0.),\
        (0, np.pi/3),\
        (np.pi/2, np.pi/3),\
        (np.pi/2, np.pi/6),\
        (0, np.pi/6),\
        (np.pi/6, 0.),\
        (np.pi/4,np.pi/4)]

_erp = [(np.pi, 0.), (3*np.pi/2, 0.), (0, -np.pi/2),\
        (np.pi/3+np.pi, 0.),\
        (np.pi, -np.pi/3),\
        (np.pi/2+np.pi, -np.pi/3),\
        (np.pi/2+np.pi, -np.pi/6),\
        (np.pi, -np.pi/6),\
        (np.pi/6+np.pi, 0.),\
        (np.pi+np.pi/4,-np.pi/4)]

__erp = [(np.pi, 0.), (np.pi/2, 0.), (0., np.pi/2),\
        (np.pi-np.pi/3, 0.),\
        (np.pi, np.pi/3),\
        (np.pi/2, np.pi/3),\
        (np.pi/2, np.pi/6),\
        (np.pi, np.pi/6),\
        (np.pi-np.pi/6, 0.),\
        (np.pi-np.pi/4,np.pi/4)]

___erp = [(0., 0.), (2*np.pi-np.pi/2, 0.), (0., np.pi/2),\
        (2*np.pi-np.pi/3, 0.),\
        (0, np.pi/3),\
        (2*np.pi-np.pi/2, np.pi/3),\
        (2*np.pi-np.pi/2, np.pi/6),\
        (0, np.pi/6),\
        (2*np.pi-np.pi/6, 0.),\
        (2*np.pi-np.pi/4,np.pi/4)]

print(f"Test xyz:")
for xx, ll in zip(xyz, erp):
    if np.linalg.norm(np.array(ll)-np.array(xyz_to_erp(xx))) > 0.1:
        print(f"{ll} != {xyz_to_erp(xx)}")
    if np.linalg.norm(np.array(xx)-np.array(erp_to_xyz(ll))) > 0.1:
        print(f"{xx} != {erp_to_xyz(xx)}")
print(f"Test _xyz:")
for xx, ll in zip(_xyz, _erp):
    if np.linalg.norm(np.array(ll)-np.array(xyz_to_erp(xx))) > 0.1:
        print(f"{ll} != {xyz_to_erp(xx)}")
    if np.linalg.norm(np.array(xx)-np.array(erp_to_xyz(ll))) > 0.1:
        print(f"{xx} != {erp_to_xyz(xx)}")
print(f"Test __xyz:")
for xx, ll in zip(__xyz, __erp):
    if np.linalg.norm(np.array(ll)-np.array(xyz_to_erp(xx))) > 0.1:
        print(f"{ll} != {xyz_to_erp(xx)}")
    if np.linalg.norm(np.array(xx)-np.array(erp_to_xyz(ll))) > 0.1:
        print(f"{xx} != {erp_to_xyz(xx)}")
print(f"Test ___xyz:")
for xx, ll in zip(___xyz, ___erp):
    if np.linalg.norm(np.array(ll)-np.array(xyz_to_erp(xx))) > 0.1:
        print(f"{ll} != {xyz_to_erp(xx)}")
    if np.linalg.norm(np.array(xx)-np.array(erp_to_xyz(ll))) > 0.1:
        print(f"{xx} != {erp_to_xyz(xx)}")

'''

def R_x(theta):
    return np.array(\
            [[1, 0, 0],\
            [0, np.cos(theta), -np.sin(theta)],\
            [0, np.sin(theta), np.cos(theta)]])

def R_y(theta):
    return np.array(\
            [[np.cos(theta), 0, np.sin(theta)],\
            [0, 1, 0],\
            [-np.sin(theta), 0, np.cos(theta)]])

def R_z(theta):
    return np.array(\
            [[np.cos(theta), -np.sin(theta), 0],\
            [np.sin(theta), np.cos(theta), 0],\
            [0, 0, 1]])

V = [\
        R_x(-np.pi/2),\
        R_x(-np.pi/2) @ R_y(-np.pi/9),\
        R_x(-np.pi/2) @ R_y(-np.pi/9*2),\
        R_x(-np.pi/2) @ R_z(-np.pi/3),\
        R_x(-np.pi/2) @ R_y(-np.pi/9) @ R_z(-np.pi/3),\
        R_x(-np.pi/2) @ R_y(-np.pi/9*2) @ R_z(-np.pi/3),\
        R_x(-np.pi/2) @ R_z(-np.pi/3*2),\
        R_x(-np.pi/2) @ R_y(-np.pi/9) @ R_z(-np.pi/3*2),\
        R_x(-np.pi/2) @ R_y(-np.pi/9*2) @ R_z(-np.pi/3*2)
    ]

H = [\
        R_z(-np.pi/2),\
        R_z(-np.pi/180*105),\
        R_z(-np.pi/180*75)
    ]

E = [\
        H[0] @ R_x(-np.pi/5),\
        H[0] @ R_x(-np.pi/5*2),\
        H[0] @ R_x(-np.pi/5*3),\
        H[0] @ R_x(-np.pi/5*4),\
        H[0] @ R_x(-np.pi/5) @ R_z(-np.pi/3),\
        H[0] @ R_x(-np.pi/5*2) @ R_z(-np.pi/3),\
        H[0] @ R_x(-np.pi/5*3) @ R_z(-np.pi/3),\
        H[0] @ R_x(-np.pi/5*4) @ R_z(-np.pi/3),\
        H[0] @ R_x(-np.pi/5) @ R_z(-np.pi/3*2),\
        H[0] @ R_x(-np.pi/5*2) @ R_z(-np.pi/3*2),\
        H[0] @ R_x(-np.pi/5*3) @ R_z(-np.pi/3*2),\
        H[0] @ R_x(-np.pi/5*4) @ R_z(-np.pi/3*2)
    ]

def coverage(alpha, step):
    count = 0
    hit = 0
    lat_step = step
    #alpha = 10 / 180 * np.pi

    for i in range(lat_step):
        sigma = (i / lat_step) * 0.5 * np.pi
        lon_step = int(lat_step * np.cos(sigma))
        for j in range(lon_step):
            theta = (j / lon_step) * 2. * np.pi

            xyz = np.array(erp_to_xyz((theta, sigma)))
            #print(f"xyz = {xyz}")

            for k, P in enumerate(V + H + E):
                erp = xyz_to_erp(P @ xyz)
                #print(f"erp={erp}")
                if abs(erp[1]) <= alpha and (erp[0] % (np.pi/3) <= alpha or erp[0] % (np.pi/3) >= np.pi/3-alpha):
                    hit += 1
                    break
            count += 1

            #if abs(sigma) < alpha and abs(theta % (np.pi/3)) < alpha:
            #    hit += 1
            #count += 1
    return hit / count


def main():
    alpha_param = np.arange(0,16,1) / 180 * np.pi
    results = [0 for _ in alpha_param]
    for i, alpha in enumerate(alpha_param):
        results[i] = coverage(alpha, 1000)
    
        print(results)

    plt.plot(60 - 2 * alpha_param * 180 / np.pi, results)
    plt.xticks(fontsize=14)
    plt.xlabel(r"$\alpha$ $(^\circ)$", fontsize=16)
    plt.ylabel(r"Coverage of sweet spots $(\%)$", fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig("coverage.pdf", bbox_inches='tight')
    

main()
