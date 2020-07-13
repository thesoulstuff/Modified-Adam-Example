import pandas as pd
import numpy as np



if __name__ == '__main__':
    #open the data file
    df = pd.read_csv("data.csv", header = None)
    #drop nan columns
    df = df.dropna()
    #apply min max to everything
    for c in df.columns:
        xmin = np.min(df[c])
        xmax = np.max(df[c])
        df[c] = (df[c] - xmin)/(xmax-xmin)
        df[c] = df[c]*(0.99-0.01) + 0.01
    df.to_csv("scaled_data.csv", index=None, header=None)


