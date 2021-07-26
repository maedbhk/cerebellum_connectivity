#predict on the sc2 02 to get the prediction sum of squared error
X_cortex_sc2_02 = pd.read_csv("X_cortex0162_sc2_02_sess_weightF.csv").iloc[:, 1:] # read the large size data

Y_cere_sc2_02=pd.read_csv("Y_cere_sc2_02_sess_weightF_suit.csv").iloc[:,1:] # read the large size

X_cortex_sc2_02_mean=np.array([X_cortex_sc2_02.mean(axis=0)]*int(X_cortex_sc2_02.shape[0]))
X_test=X_cortex_sc2_02-X_cortex_sc2_02_mean

Y_cere_sc2_02_mean=np.array([Y_cere_sc2_02.mean(axis=0)]*int(Y_cere_sc2_02.shape[0]))
Y_test=Y_cere_sc2_02-Y_cere_sc2_02_mean

for i in range(15): 
    print(f"{i+1}")  
    C_est=pd.read_csv("C_hat_R"+str(i+1)+"_weightF_lambda1.csv").iloc[:, 1:]
    Y_pred_smooth=np.array(X_test)@np.array(C_est)
    #Prediction error int terms of the norm of sum sqr err on test data
    pred_err_smoth=LA.norm(Y_test-Y_pred_smooth)
    print(f"pred_err={pred_err_smoth}")
    R_LRS, R_voxel_LRS=ev.calculate_R(Y_test, Y_pred_smooth)
    print(f"R={R_LRS}")
    R2_LRS, R2_voxel_LRS=ev.calculate_R2(Y_test, Y_pred_smooth)
    print(f"R2={R2_LRS}")

U=pd.read_csv("U_weightF.csv").iloc[:,1:] 

nii_func = data.convert_cerebellum_to_nifti(U.iloc[:,0])
# Map to surface
map_func = suit.flatmap.vol_to_surf(nii_func)
# Plot the flatmap
suit.flatmap.plot(map_func,cscale=[-0.5,0.5])

