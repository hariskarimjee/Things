import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
from random import *
import time

R = 6e7    #radius of torus centroid
r = 1e7    #radius of torus tube
limits = R + r
G = 6.674e-11 #graviational constantt = 0

dt = 5e3
n_particles = 10000

M = 5e23 #mass of body | kg
M_p = M/n_particles #mass of each torus particle
m = 1000 #mass of orbiter | kg
 
torus_particles = np.zeros((n_particles, 3)) #array holding torus particles

object = np.zeros((2,3)) #tensor holding orbiter position, velocity
object[0,:] = (0,0,0)
object[1,:] = [500,500,800]

def in_space(location):
    x,y,z = location[0], location[1], location[2]
    # if (((x**2 + y**2)**0.5 - R)**2 + z**2) <= r**2: return True
    if (x**2 + y**2 + z**2) <= R**2: return True
    else: return False

def gravity(orbiter, particle):
    r_vec = np.array([(orbiter[0,0]-particle[0]),(orbiter[0,1]-particle[1]),(orbiter[0,2]-particle[2])])
    mod_r =  np.linalg.norm(r_vec)
    r_hat = r_vec / mod_r
    F = -G*M_p*m / (mod_r**2)
    F_vec = r_hat * F
    return F_vec

def get_force(orbiter, body_particles):
    force = np.zeros(3)
    for num,particle in enumerate(body_particles):
        force += gravity(orbiter, particle)
        # print(num)
    # print(force)
    return force
       

def orbit(orbiter, body_particles):
    acceleration = get_force(orbiter, body_particles) / m
    # print(acceleration)
    dv = acceleration * dt
    orbiter[1,:] += dv[:]
    dr = orbiter[1,:] * dt
    orbiter[0,:] += dr

    

for loc in torus_particles:
    location = loc
    location[0], location[1], location[2] = randrange(-limits, limits), randrange(-limits,limits), randrange(-limits,limits)
    while not in_space(location):
        location[0], location[1], location[2] = randrange(-limits, limits), randrange(-limits,limits), randrange(-limits,limits)

# plt.ion()
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(-limits*3, limits*3)
ax.set_ylim3d(-limits*3, limits*3)
ax.set_zlim3d(-limits*3, limits*3)
   


orbit_x = [object[0,0]]
orbit_y = [object[0,1]]
orbit_z = [object[0,2]]

ax.plot(torus_particles[:,0],torus_particles[:,1],torus_particles[:,2],marker="o",linestyle="",markersize=3)
# sc =     

for i in range(1000):
    orbit(object, torus_particles)
    orbit_x.append(object[0,0])
    orbit_y.append(object[0,1])
    orbit_z.append(object[0,2])
    print(i)
    if in_space(object[0,:]): break

# def update_graph(num):
#     orbit(object, torus_particles)
#     orbit_x.append(object[0,0])
#     orbit_y.append(object[0,1])
#     orbit_z.append(object[0,2])
#     graph.set_data (orbit_x, orbit_y)
#     graph.set_3d_properties(orbit_z)
#     if in_space(object[0,:]): sys.exit()
#     return graph, 


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(-limits*2, limits*2)
# ax.set_ylim3d(-limits*2, limits*2)
# ax.set_zlim3d(-limits*2, limits*2)
# ax.plot(torus_particles[:,0],torus_particles[:,1],torus_particles[:,2], linestyle="", marker="o",markersize = 3)
# graph, = ax.plot(orbit_x,orbit_y,orbit_z, linestyle="", marker="o", markersize=3)

# ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=1, blit=True)
ax.plot(orbit_x,orbit_y,orbit_z, linestyle="-", marker="o", markersize=3)
plt.show()




