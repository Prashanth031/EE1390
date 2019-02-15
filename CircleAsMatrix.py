import numpy as np
import matplotlib.pyplot as plt

# equation given X^T*X + 2*[-2,3].X-12 = 0  
# (X-C)^T . (X-C) = R^2 ... Equation of a circle
#  C = [2,-3] , R^2 - C^T.C = 12 

C = np.array([2,-3])

R = np.sqrt(12 + np.matmul(C.T,C))  # Square of the Radius of the circle

# print(R)
n=100

theta = np.linspace(0,2*np.pi,n)

M1 = np.zeros((2,n))

for i in range (n):
	M1[:,i] = (np.array([2+R*np.cos(theta[i]),-3+R*np.sin(theta[i])])).T 

plt.plot(M1[0,:],M1[1,:])

# Centre of other circle - [-3,2]
S = np.array([-3,2])

D = C-S

d2 = np.matmul(D.T,D) #Square of distance between the two centres

# The lines CS and the radii of the both the circles form a right angled triangle

# Using pythagoras theorem 

# R_s = np.sqrt(d2 + R**2)

# print(R_s)

len = 10

lam = np.linspace(0,1,len)

N = np.zeros((2,len))

for i in range (len):
	N[:,i] = (S + lam[i]*(C-S)).T

plt.plot(N[0,:],N[1,:])

plt.plot(C[0],C[1],'o')
plt.plot(S[0],S[1],'o')

plt.text(C[0]*(1.1),C[1]*(0.9),'C')
plt.text(S[0]*(1.1),S[1]*(0.9),'S')

# to find points of intersections of the circles 

G = (C-S)/5

# Equation of line - N^T*X = P = N^T*C 

P = np.matmul(G.T,C)

# Using Parametric equation (theta) of circle to substitute in line equation and find points of intersection

# theta = [cos(t),sin(t)]
# R*G^T*theta = P - G^T*C
# we get tan(t) = 1 
# t = pi/4 , -pi/4 

t1 = np.pi/4
t2 = 5*np.pi/4
# A = np.array([2+(5/2**0.5),-3+(5/2**0.5)])

# B = np.array([2-(5/2**0.5),-3-(5/2**0.5)])
Theta1 = np.array([np.cos(t1),np.sin(t1)])
Theta2 = np.array([np.cos(t2),np.sin(t2)])

A = C + R*(Theta1.T)
B = C + R*(Theta2.T)
Q = S-A
R_s = np.sqrt(np.matmul(Q.T,Q))

print("The radius of the circle with centre S :",R_s)

M2 = np.zeros((2,n))
for i in range (n):
	M2[:,i] = (np.array([-3+R_s*np.cos(theta[i]),2+R_s*np.sin(theta[i])])).T 

plt.plot(M2[0,:],M2[1,:])

plt.plot(A[0],A[1],'o')
plt.plot(B[0],B[1],'o')

plt.text(A[0]*(1.1),A[1]*(1.1),'A')
plt.text(B[0]*(1.1),B[1]*(1.1),'B')


x_AB = np.zeros((2,len))
x_SB = np.zeros((2,len))
x_AS = np.zeros((2,len))

for i in range(len):
	x_AB[:,i] = (A + lam[i]*(B-A)).T
	x_AS[:,i] = (A + lam[i]*(S-A)).T
	x_SB[:,i] = (S + lam[i]*(B-S)).T

plt.plot(x_AB[0,:],x_AB[1,:],label='$2R$')
plt.plot(x_AS[0,:],x_AS[1,:],label='$R_s$')
plt.plot(x_SB[0,:],x_SB[1,:],label='$R_s$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')


plt.grid()
plt.show()


