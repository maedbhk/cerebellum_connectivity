from collections import defaultdict
import click
import os
import pandas as pd
import nibabel as nib
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as ss
from statsmodels.stats.anova import AnovaRM

from connectivity import weights as cweights
from connectivity import visualize as summary
from connectivity import data as cdata
import connectivity.constants as const

# @click.command()
# @click.option("--roi")
# @click.option("--weights")
# @click.option("--data_type")

def plot_dispersion(method='ridge',cortex='tessels0042'):
    dirs = const.Dirs(exp_name='sc1')
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_dispersion_stats.csv')
    df = pd.read_csv(fpath)
    df['w_var']=df.Variance*df.sum_w
    df = df[df.cortex==cortex]
    T = pd.pivot_table(df,values=['sum_w','w_var','Variance'],index=['sn','roi'],aggfunc='mean')
    T = T.reset_index()
    T['var_w'] = T.w_var/T.sum_w
    fig, axes = plt.subplots(2, 2,figsize=(12,10))
    sns.lineplot(data=df,x='roi',y='Variance',hue='hem',ax=axes[0,0])
    axes[0,0].set_title('Spherical Var by Hemisphere')
    sns.lineplot(data=df,x='roi',y='sum_w',hue='hem',ax=axes[0,1])
    axes[0,1].set_title('Sum of weights for hemisphere')
    sns.lineplot(data=T,x='roi',y='Variance',ax=axes[1,0])
    axes[1,0].set_title('Raw averaged Spherical Var')
    sns.lineplot(data=T,x='roi',y='var_w',ax=axes[1,1])
    axes[1,1].set_title('weighted spherical Var')
    return T,df

def dispersion_summary(
    atlas='MDTB10',
    method='ridge', # L2regression
    exp='sc1',
    ):

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)
    subjs = const.return_subjs

    # models, cortex_names = summary.get_best_models(method=method)
    # cortex = 'tessels1002';
    models = ['ridge_tessels0042_alpha_4','ridge_tessels0162_alpha_6','ridge_tessels0362_alpha_6','ridge_tessels0642_alpha_8','ridge_tessels1002_alpha_8']
    cortex_names = ['tessels0042','tessels0162','tessels0362','tessels0642','tessels1002']

    data_dict_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        for subj in subjs:
                roi_betas, reg_names, colors = cweights.average_region_data(subj,
                                        exp=exp, cortex=cortex,
                                        atlas=atlas, method=method, alpha=alpha,
                                        weights='nonzero', average_subjs=False)

                # save out cortical distances
                df_res = dispersion_cortex(roi_betas, reg_names, colors,cortex=cortex)
                N=df_res.shape[0]
                df_res['sn']=[subj]*N
                df_res['cortex']=[cortex]*N
                df_res['method']=[method]*N
                df = pd.concat([df,df_res])
        pass
    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_dispersion_stats.csv')
    df.to_csv(fpath)

def dispersion_cortex(roi_betas,
    reg_names,
    colors,
    cortex):
    """Caluclate spherical dispersion for the connectivity weights

    Args:
        roi_betas (np array):
        reg_names (list of str):
        colors (np array):
        cortex (str):

    Returns:
        dataframe (pd dataframe)
    """
    # get data
    num_roi, num_parcel = roi_betas.shape

    hem_names = ['L', 'R']

    # optionally threshold weights based on `threshold`
    data = {}; roi_dist_all = []
    dist,coord = cdata.get_distance_matrix(cortex)

    df = pd.DataFrame()

    for h,hem in enumerate(hem_names):

        labels = cweights.get_labels_hemisphere(roi=cortex, hemisphere=hem)
        # weights

        # Calculate spherical STD as measure
        # Get coordinates and move back to 0,0,0 center
        coord_hem = coord[labels,:]
        coord_hem[:,0]=coord_hem[:,0]-(h*2-1)*500

        # Now compute a weoghted spherical mean, variance, and STD
        # For each tessel, the weigth w_i is the connectivity weights with negative weights set to zero
        # also set the sum of weights to 1
        w = roi_betas[:,labels]
        w[w<0]=0
        sum_w = w.sum(axis=1)
        w = w /sum_w.reshape(-1,1)

        # We then define a unit vector for each tessel, v_i:
        v = coord_hem.copy().T
        v=v / np.sqrt(np.sum(v**2,axis=0))

        # Weighted average vector =sum(w_i*v_i)
        # R is the length of this average vector

        R = np.zeros((num_roi,))
        for i in range(num_roi):

            mean_v = np.sum(w[i,:] * v,axis=1)
            R[i] = np.sqrt(np.sum(mean_v**2))

            # Check with plot
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(w[i,:]*v[0,:],w[i,:]*v[1,:],w[i,:]*v[2,:])
            # ax.scatter(mean_v[0],mean_v[1],mean_v[2])
            # pass

        V = 1-R # This is the Spherical variance
        Std = np.sqrt(-2*np.log(R)) # This is the spherical standard deviation
        df1 = pd.DataFrame({'Variance':V,'Std':Std,'hem':h*np.ones((num_roi,)),'roi':np.arange(num_roi),'sum_w':sum_w})
        df = pd.concat([df,df1])
    return df

def distances_map(
    atlas='MDTB10',
    method='ridge',
    weights='nonzero',
    threshold=100
    ):

    exp = 'sc1'
    dirs = const.Dirs(exp_name=exp)

    subjs, _ = cweights.split_subjects(const.return_subjs, test_size=0.3)

    # models, cortex_names = summary.get_best_models(method=method)
    cortex = 'tessels1002'
    models = [f'{method}_{cortex}_alpha_8']
    cortex_names = ['tessels1002']

    for (best_model, cortex) in zip(models, cortex_names):

        # full path to best model
        fpath = os.path.join(dirs.conn_train_dir, best_model)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        roi_betas_all = []
        for subj in subjs:
            roi_betas, reg_names, colors = cweights.average_region_data(subj,
                                    exp=exp, cortex=cortex,
                                    atlas=atlas, method=method, alpha=alpha,
                                    weights=weights, average_subjs=False)

            roi_betas_all.append(roi_betas)

        roi_betas_group = np.nanmean(np.stack(roi_betas_all), axis=0)
        giis = cweights.regions_cortex(roi_betas_group, reg_names, cortex=cortex, threshold=threshold)

        fname = f'group_{atlas}_threshold_{threshold}'
        [nib.save(gii, os.path.join(fpath, f'{fname}.{hem}.func.gii')) for (gii, hem) in zip(giis, ['L', 'R'])]

def run():
    # dispersion_summary()
    T,df=plot_dispersion(cortex='tessels1002')
    res = AnovaRM(data=T, depvar='var_w', subject='sn', within=['roi']).fit()
    print(res)
    pass

if __name__ == "__main__":
     run()