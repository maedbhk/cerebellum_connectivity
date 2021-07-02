import pandas as pd
import os

import connectivity.io as cio
import connectivity.constants as const

def run(summary='train'):

    for exp in ['sc1', 'sc2']:
        dirs = const.Dirs(exp_name=exp)

        if summary=='train':
            fpath = os.path.join(dirs.conn_train_dir, 'train_summary.csv')
        elif summary=='eval':
            fpath = os.path.join(dirs.conn_eval_dir, 'eval_summary.csv')
        
        # read dataframe
        df = pd.read_csv(fpath)

        # housekeeping
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # get unique atlas names
        atlases = df['X_data'].unique()

        df_all = pd.DataFrame()
        # loop over atlases in dataframe
        for atlas in atlases:

            df_atlas = df.query(f'X_data=="{atlas}"')
            
            X = cio.read_mat_as_hdf5(os.path.join(dirs.beta_reg_dir, 's02', f'Y_glm7_{atlas}.mat')) # cortical features should be the same for all subjects
            df_atlas['num_regions'] = X['data'].shape[0]

            df_all = pd.concat([df_all, df_atlas.reset_index(drop=True)])
        
        df_all.to_csv(fpath, index=False)

if __name__ == "__main__":
    run(summary='train')
    run(summary='eval')

