from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
import connectivity.constants as const
import connectivity.model as mod
import quadprog as qp
import timeit
import

def simulate_IID_Data(N=8, P1=6, P2=5):
    """
        Make some artificial data with positive connectivity weights
    """
    # Make cortical data (iid)
    X = np.random.normal(0, 1, (N, P1))
    X = X - X.mean(axis=0)
    X = X / np.sqrt(np.sum(X ** 2, 0) / X.shape[0])

    # Make non-negative connectivity weights
    W = np.random.normal(0, 1, (P1, P2))
    W[W < 0] = 0.0

    # Generate the cerebellar data
    Y = X @ W + np.random.normal(0, 1, (N, P2))

    return X,Y

def compare_OLS_NNLS():
    X, Y = simulate_IID_Data()
    W1 = np.linalg.solve(X.T @ X, X.T @ Y)  # Normal OLS solution

    # Non-negative solution without regularisation
    nn1 = mod.NNLS(alpha=0, gamma=0)
    nn1.fit(X, Y)
    Yp = nn1.predict(X)
    R2 = 1 - ((Y - Yp) ** 2).sum() / (Y ** 2).sum()
    print(f"model1: {R2.round(2)}")
    # Now
    nn2 = mod.NNLS(alpha=0, gamma=1)
    nn2.fit(X, Y)
    Yp = nn2.predict(X)
    R2 = 1 - ((Y - Yp) ** 2).sum() / (Y ** 2).sum()
    print(f"model2: {R2.round(2)}")

def NNLS_speed_test():
    P1 = [10,20,30,40,50,70,100,200,300,400,500,600]
    time = []
    for p1 in P1:
        X, Y = simulate_IID_Data(N=42,P1=p1,P2=10)

        # Non-negative solution without regularisation
        nn1 = mod.NNLS(alpha=0, gamma=0)
        tic = timeit.default_timer()
        nn1.fit(X,Y)
        toc = timeit.default_timer()
        time.append(tic-toc)



    T = pd.DataFrame({'P':P1,'time':time})
    pass

if __name__ == "__main__":
    NNLS_speed_test()
