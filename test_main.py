import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def sigmoid(x):
    return 1/(1+np.exp(-x))

def mlp_foward(xe, W):
    activations = [xe.copy()]
    zs = []
    for i in range(len(W)-1):
        zs.append(W[i].dot(activations[-1]))
        a = sigmoid(zs[-1])
        activations.append(a)
    z = W[-1].dot(activations[-1] )
    zs.append(z)
    activations.append(zs[-1])
    return activations, zs


def mlp_train(xe, ye, w):
    act, z_s = mlp_foward(xe, w)
    E = ye - act[-1]
    costs = []
    costs.append(np.square(E).mean()) # mse
    costs.append(np.sqrt(costs[0])) # rmse
    costs.append(r2_score(ye, act[-1][0]))

    compare = list(zip(ye, act[-1][0]))
    
    return costs, compare



if __name__ == '__main__':
    #load data and do stuff
    df_test = pd.read_csv("test_data.csv", header=None)
    weights = np.load("pesos.npy", allow_pickle=True)
    #save the train data
    
    X = df_test.iloc[:,:-1].T.values
    Y = df_test.iloc[:,-1].T.values
    #feed foward 

    costs, compare = mlp_train(X, Y, weights)
    with open("errores.csv", 'w') as f:
        f.write(",".join([str(x) for x in costs]))

    with open("estimados.csv", 'w') as f:
        for real, predicted in compare:
            f.write("{},{}\n".format(real,predicted))



    

