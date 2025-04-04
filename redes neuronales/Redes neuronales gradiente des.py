import numpy as np
import matplotlib.pyplot as plt

def prediction(x):
    return w*x + b

w = 0.1
b = 0.3
learningRate = 0.01
iterations = 100
# x = np.array([1,2,3,4,5,6,7,8])
x = np.array([0.2,1.0,1.4,1.6,2.0,2.2,2.7,2.8,3.2,3.3,3.5,3.7,4.0,4.4,5.0,5.2])
# target = np.array([12,14,16,18,20,22,24,26])
# target = np.array([2,4,6,8,10,12,14,16])
target = np.array([230,555,815,860,1140,1085,1200,1330,1290,870,1545,1480,1750,1845,1790,1955])

# xnormalitation = x/np.linalg.norm(x)
xnormalitation = (x - min(x))/(max(x)-min(x))
targetsNormalitation = (target - min(target))/(max(target)-min(target))



errors = np.zeros(iterations)

for iter in range(iterations):
    avg_cost = np.mean((targetsNormalitation - prediction(xnormalitation))**2)  
    errors[iter] = avg_cost
    plt.scatter(iter,avg_cost , c='red')

    #derivada parcial respecto a w
    w_d =   -2*(targetsNormalitation - prediction(xnormalitation))*xnormalitation
    avg_w_d = np.mean(w_d)
    
    #derivada parcial respecto a b
    b_d =   -2*(targetsNormalitation - prediction(xnormalitation))
    avg_b_d = np.mean(b_d)

    #actualizar w y b
    w -= learningRate * avg_w_d
    b -= learningRate * avg_b_d

    # print("Iter: ",iter," Error: ",avg_cost," w: ",w," b: ",b)

print("w: ",w," b: ",b)

plt.plot(range(iterations), errors,"--")
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()


#test
print("data test")
xtest = np.array([2,2.5,3,7,7.2,100])
xtestNormalitation = (xtest - min(x))/(max(x)-min(x))
ytest = prediction(xtest)
ytestNormalitation = ytest*(max(target)-min(target))+min(target)
print(xnormalitation)
print(targetsNormalitation)
print(xtestNormalitation)
print(ytestNormalitation)