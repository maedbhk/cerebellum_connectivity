from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run

def test_single_fit():
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

def test_run_fit(): 
    config = run.get_default_train_config()
    config['subjects']=[2,3,4,5]
    Model = run.train_models(config, save=True)
    pass

def test_run_eval():
    config = run,get_default_eval_config()
    config['subjects']=[3,4,5]
    T = run.eval_models(config)
    pass

if __name__ == '__main__':
    test_run_fit()
