import numpy as np

count = 0
hit = 0
lat_step = 200 
alpha = 5 / 180 * np.pi

for i in range(lat_step):
    sigma = ((i / lat_step) - 0.5) * np.pi
    lon_step = int(lat_step * np.cos(sigma))
    for j in range(lon_step):
        theta = ((j / lon_step) - 0.5) * np.pi * 2
        
        if abs(sigma) < alpha and (theta % (np.pi/3) < alpha or theta % (np.pi/3) > np.pi/3 - alpha):
            hit += 1
        count += 1
print(count)
print(hit)
print(hit / count)
        
