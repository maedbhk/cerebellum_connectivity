from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
import connectivity.constants as const
import connectivity.model as mod
import quadprog as qp
import timeit

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

    return X,Y,W

def simulate_real_Data(corticalParc="tessels0162", subj_id = "s02",P2 = 100):
    """ 
        Make some artifical data from the real problem size
    """
    Xdata = Dataset(experiment="sc1", glm="glm7", roi=corticalParc, subj_id=subj_id)
    Xdata.load_mat()  # Import the data from Matlab
    X, S = Xdata.get_data(averaging="sess",weighting=2)
    X = X - X.mean(axis=0)
    X = X / np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
    N, P1 = X.shape 

    # Make non-negative connectivity weights
    W = np.random.normal(0, 1, (P1, P2))
    W[W < 0] = 0.0

    # Generate the cerebellar data
    Y = X @ W + np.random.normal(0, 1, (N, P2))

    return X,Y,W 


def compare_OLS_NNLS():
    X, Y, W = simulate_IID_Data()
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
    P1 = [10,20,30,40,50,70,100,200,300,400,500,600,1000]
    time1=[]
    time2=[]
    for i,p1 in enumerate(P1):
        X, Y, W  = simulate_IID_Data(N=42,P1=p1,P2=10)

        # Non-negative solution 
        nn1 = mod.NNLS(alpha=0.1, gamma=0, solver="quadprog")
        tic = timeit.default_timer()
        nn1.fit(X,Y)
        toc = timeit.default_timer()
        time1.append(toc-tic)

        # Non-negative solution 
        nn2 = mod.NNLS(alpha=0.1, gamma=0, solver="cvxopt")
        tic = timeit.default_timer()
        nn2.fit(X,Y)
        toc = timeit.default_timer()
        time2.append(toc-tic)

    T = pd.DataFrame({'P':P1,'time1':time1,'time2':time2})
    pass

def NNLS_speed_real():
    X, Y, W  = simulate_real_Data(P2=100)

    # Non-negative solution 
    nn1 = mod.NNLS(alpha=0.1, gamma=0, solver="quadprog")
    tic = timeit.default_timer()
    nn1.fit(X,Y)
    toc = timeit.default_timer()
    time1 = toc-tic 

    # Non-negative solution 
    nn2 = mod.NNLS(alpha=0.1, gamma=0, solver="cvxopt")
    tic = timeit.default_timer()
    nn2.fit(X,Y)
    toc = timeit.default_timer()
    time2 = toc-tic

    pass

if __name__ == "__main__":
    NNLS_speed_real()