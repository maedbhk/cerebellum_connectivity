from connectivity.data import Dataset
import connectivity.model as model
Ydata=Dataset(glm =7, sn = 2, roi='cerebellum_grey')
Ydata.load_mat()
Y,T = Ydata.get_data(averaging='sess')
Xdata = Dataset(glm=7, sn= 2, roi='tesselsWB162') 
Xdata.load_mat()
X,T = Xdata.get_data(averaging='sess')
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
R = model.L2regression(alpha=1)
R.fit(X,Y)

pass
