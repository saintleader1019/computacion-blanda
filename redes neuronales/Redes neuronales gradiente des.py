import numpy as np
import matplotlib.pyplot as plt

def prediction(x):
    return w*x

w = 0.1
learningRate = 0.01
iterations = 20
x = np.array([1,2,3,4,5,6,7,8])
target = np.array([2,4,6,8,10,12,14,16])+10
errors = np.zeros(iterations)

for iter in range(iterations):
    mse =   -2*(target - prediction(x))*x
    print(mse)  
    mse = np.mean(mse)
    errors[iter] = mse
    plt.scatter(iter, mse)
    w -= learningRate * mse

plt.plot(range(iterations), errors,"--")
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()


#test
xtest = np.array([2,2.5,3,7,7.2,100])
ytest = prediction(xtest)
print(x)
print(target)
print(xtest)
print(ytest)