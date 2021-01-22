from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
import connectivity.constants as const
import connectivity.model as mod
import quadprog as qp

# Make some artificial data with positive connectivity weights
N = 8 # number of conditions
P1 = 6 # Number of cortical voxels
P2 = 5 # number of cerebellar voxels

X = np.random.normal(0,1,(N,P1))
X = X - X.mean(axis = 0)
X = X / np.sqrt(np.sum(X**2,0)/X.shape[0])
W = np.random.normal(0,1,(P1,P2))
W[W<0]=0.0
Y = X @ W + np.random.normal(0,1,(N,P2))

W1 = np.linalg.solve(X.T @ X,X.T @ Y) # Normal OLS solution

# Non-negative solution without regularisation
nn1 = mod.NNLS(alpha=0, gamma=0)
nn1.fit(X,Y)
Yp = nn1.predict(X)
R2 = 1-((Y-Yp)**2).sum()/(Y**2).sum()
print(f'model1: {R2.round(2)}')
# Now fot
nn2 = mod.NNLS(alpha=0, gamma=1)
nn2.fit(X,Y)
Yp = nn2.predict(X)
R2 = 1-((Y-Yp)**2).sum()/(Y**2).sum()
print(f'model2: {R2.round(2)}')
