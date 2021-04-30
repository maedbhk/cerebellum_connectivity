# import libraries
from connectivity.scripts import script_mk

models = ['ridge_yeo7_alpha_2', 'ridge_yeo17_alpha_2',
         'ridge_tessels0042_alpha_4', 'ridge_tessels0162_alpha_6',
         'NNLS_tessels0042_alpha_0_gamma_0_no_cv',
         'NNLS_tessels0162_alpha_0_gamma_0_no_cv', 
         'NNLS_yeo7_alpha_0_gamma_0_no_cv', 'NNLS_yeo17_alpha_0_gamma_0_no_cv']

for exp in range(2):

    try: 
        for model in models:

            cortex = model.split('_')[1]

            # save voxel/vertex maps for best training weights
            script_mk.save_weight_maps(model_name=model, cortex=cortex, train_exp=f"sc{2-exp}")

            # test best train model
            script_mk.eval_model(model_name=model, cortex=cortex, train_exp=f"sc{2-exp}", eval_exp=f"sc{exp+1}")
    except: 
        pass

if __name__ == "__main__":
    run()