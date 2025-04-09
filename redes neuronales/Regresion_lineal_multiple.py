import numpy as np
import matplotlib.pyplot as plt

def prediction(x):
    return x.dot(w) + b

w=np.array([0.1,0.1])
b=1
lr = 0.1
x = np.array([(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800,0.3709),
              (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
              (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
              (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)])
targets =np.array([230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955])
predictions = prediction(x)

#mse = (targets - predictions)**2
#print(predictions)
#print("Mean Squared Error:", mse)

iterations = 3000

for i in range (iterations):

    mse = (targets - prediction(x))**2
    avg_mse = np.mean(mse)
    plt.scatter(i,avg_mse , c='red', alpha = 0.7)

    #derivada parcial respecto a w
    w_d = -2*(x.T).dot(targets - prediction(x))
    avg_w_d = w_d/np.size(w_d,0)
    #print("w_d: ",w_d)
    #print("avg_w_d: ",avg_w_d)
    
    #derivada parcial respecto a b
    b_d = -2*(targets - prediction(x))
    b_d = np.ones([1,np.size(targets)]).dot(b_d)
    avg_b_d = b_d/np.size(targets)
    #print("b_d: ",b_d)
    #print("avg_b_d: ",avg_b_d)

    #actualizar w y b
    w -= lr * avg_w_d
    b -= lr * avg_b_d
    print("Iter: ",i," Error: ",avg_mse," w: ",w," b: ",b)


xt = x[[0,4,9,10,14],:]
target = targets[0],targets[4],targets[9],targets[10],targets[14]

yt = prediction(xt)

print(f'xt: {xt}',f'yt: {yt}' ,f'targets: {target}')