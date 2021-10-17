

import matplotlib.pyplot as plt
import numpy as np

#%%

nx, ny = 100,100
x = np.linspace(-1,3,nx)
y = np.linspace(-1,3,ny)

xv, yv = np.meshgrid(x, y)

def gauss(x,y,mx,my,sigma):
    return 1/sigma * np.exp( -1 / (2*sigma**2) * ((x - mx)**2+(y-my)**2))

def gauss_x(x,y,mx,my,sigma):
    return 1/sigma * np.exp( -1 / (2*sigma**2) * ((x - mx)**2))

fig, ax = plt.subplots(1,1,figsize=(6,6))
plt.contour(xv,yv,gauss(xv,yv,0,0,1))
plt.contour(xv,yv,gauss(xv,yv,2,2,0.5))

#%%

def wx(x, y, g0, g1):
    p0 = gauss(x,y,*g0)
    p1 = gauss(x,y,*g1)
    return p0 / (p0 + p1)

def w1(x, y, g0, g1):
    p0 = gauss_x(x,y,*g0)
    p1 = gauss_x(x,y,*g1)
    return p0 / (p0 + p1)

def orth(wx_0, w1_0):
    p0 = wx_0 / w1_0
    p1 = (1 - wx_0) / (1 - w1_0)
    return p0 / (p0 + p1)

wx_0 = wx(xv,yv, (0,0,1),(2,2,0.5))
w1_0 = w1(xv,yv, (0,0,1),(2,2,0.5))
print(wx_0.shape, w1_0.shape)
w2_0 = orth(wx_0,w1_0)
fig, ax = plt.subplots(1,1,figsize=(6,6))
plt.contour(xv,yv,wx_0)
plt.contour(xv,yv,w1_0)
plt.contour(xv,yv,w2_0)
plt.show()