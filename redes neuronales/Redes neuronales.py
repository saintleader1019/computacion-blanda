import numpy as np
import matplotlib.pyplot as plt

def prediction(x):
    return w*x

#Valor de peso
w = 0.1
learningRate = 0.1
iterations = 50
x = np.array([1,2,3,4,5])
target = np.array([2,4,6,8,10])
errors = np.zeros(iterations)

for iter in range(iterations):
    predictions = prediction(x)
    averageCost = target - predictions
    averageCost = np.mean(averageCost)
    errors[iter] = averageCost
    plt.scatter(iter, averageCost)
    w += learningRate * averageCost

plt.plot(range(iterations), errors,"--")
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()


#test

xtest = np.array([2,2.5,3,7])
ytest = prediction(xtest)
print(x)
print(target)
print(xtest)
print(ytest)