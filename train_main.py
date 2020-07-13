import pandas as pd
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def mlp_foward(xe, W):
    activations = [xe.copy()]
    for i in range(len(W)-1):
        z = W[i].dot(activations[-1])
        a = sigmoid(z)
        activations.append(a)
    z = W[-1].dot(activations[-1] )
    activations.append(z)
    return activations



def mlp_train(xe, ye, arglist):
    C, epochs, mu, b1, b2, hidden_sizes = arglist
    W = []
    input_size = xe.shape[0]
    for size in hidden_sizes:
        r = np.sqrt(6/(size+input_size))
        W.append(np.random.uniform(
                low=-r, high=r, size=(size, input_size)
                ))
        input_size = size
    r = np.sqrt(6/(1+input_size))
    W.append(np.random.uniform(
                low=-r, high=r, size=(1,input_size
        ))
    )

    for epoch in range(epochs):
        act = mlp_foward(xe, W)
        E = ye - act[-1]
        cost = np.square(E).mean()
        print(cost)
    pass


if __name__ == '__main__':
    #load params
    with open('param_mlp.csv',"r") as f:
        params = [l.strip() for l in f.readlines()]
    train_size = float(params[0])
    C = float(params[1])
    epochs = int(params[2])
    mu = float(params[3])
    b1, b2 = [float(x) for x in params[4].split()]
    hidden_sizes = [int(x) for x in params[5].split()]
    #load data and do stuff
    df = pd.read_csv("scaled_data.csv", header=None)
    df = df.sample(frac=1).reset_index(drop=True) #inplace
    train_samples = int(df.shape[0]*train_size)
    print(df.shape[0])
    print(train_samples)
    df_train = df.iloc[0:train_samples,:]
    df_test = df.iloc[train_samples:, :]
    #save the train data
    df_test.to_csv("test_data.csv", header=None, index=None)
    
    X = df_test.iloc[:,:-1].T.values
    Y = df_test.iloc[:,-1].T.values
    #feed foward epochs and shit
    args = (C, epochs, mu, b1, b2, hidden_sizes)
    mlp_train(X, Y, args)


    

