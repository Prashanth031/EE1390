import numpy as np
import matplotlib.pyplot as plt

C = np.array([2,-3])

R = 5
n=100
theta = np.linspace(0,2*np.pi,n)
S = np.array([-3,2])
M = np.zeros((2,n))

for i in range (n):
	M[:,i] = (np.array([2+R*np.cos(theta[i]),-3+R*np.sin(theta[i])])).T 

# a = 2+(5/2**0.5)
# print(a)

plt.plot(M[0,:],M[1,:])
plt.plot(C[0],C[1],'o')
plt.text(C[0]*(1.1),C[1]*(1.1),'C')
plt.text(S[0]*(1.1),S[1]*(1.1),'S')
plt.plot(S[0],S[1],'o')

len=10
l = np.linspace(0,1,len)
W = np.zeros((2,len))

for i in range(len):
	W[:,i] = C + l[i]*(S-C)

t1 = np.pi/4
t2 = 5*np.pi/4
# A = np.array([2+(5/2**0.5),-3+(5/2**0.5)])

# B = np.array([2-(5/2**0.5),-3-(5/2**0.5)])
Theta1 = np.array([np.cos(t1),np.sin(t1)])
Theta2 = np.array([np.cos(t2),np.sin(t2)])

A = C + R*(Theta1.T)
B = C + R*(Theta2.T)

plt.plot(A[0],A[1],'o')
plt.plot(B[0],B[1],'o')

plt.text(A[0]*(1.1),A[1]*(1.1),'A')
plt.text(B[0]*(1.1),B[1]*(1.1),'B')

x_AB = np.zeros((2,len))

for i in range(len):
	x_AB[:,i] = (A + l[i]*(B-A)).T

plt.plot(x_AB[0,:],x_AB[1,:])

plt.plot(W[0,:],W[1,:])
plt.grid()

plt.show()