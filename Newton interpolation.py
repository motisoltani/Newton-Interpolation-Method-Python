'''
Interpolation and Curve Fitting
Newton Interpolation Method 
language: Python

Motahare Soltani
soltani.wse@gmail.com

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

#Entering X and Y values
x =np.array([0, 2, 4, 6, 8, 10], float)
y = np.array([0, 1.5, 2, 2.5, 3, 3], float) 
n = len(x)
p = np.zeros([n, n+1])


def newton_interpol(z):
  f_list = []
  for m in z:
   
    for i in range(n):
      p[i, 0] = x[i]
      p[i, 1] = y[i]

    for i in range(2, n+1): 
      for j in range(n+1-i):
        p[j,i] = ( p[j+1, i-1] - p[j, i-1] ) / ( x[j+i-1] - x[j] )
        
    np.set_printoptions(suppress=True)

    #Coefficients
    b = p[0][1:]  

    lst = [] 

    t = 1
    for i in range(len(x)):
      t *= (m-x[i]) 
      lst.append(t)

    f = b[0]
    for k in range(1,len(b)):
      f += b[k] * lst[k-1] 
    f_list.append(f)
  return f_list

#Enter the point at which you want to calculate	
z = [2.25]
print("The value of polynomial: ", newton_interpol(z))


#Plot 
step = 0.1
x_list = np.arange(x[0], x[-1]+0.1, step).tolist()
y_list = newton_interpol(x_list)
fig = figure(figsize=(8, 8), dpi=75)
font1 = {'color':'blue','size':15}
plt.scatter(x,y, marker='*', c='purple', s=250)
plt.plot(x_list, y_list, 'b-')
plt.xlabel('X', fontsize=12)
plt.ylabel('Function', fontsize=12)
plt.title('Newton Interpolation Method', fontdict=font1, loc='left')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend(['Interpolation', 'Data'], loc ="lower right")
plt.show()




