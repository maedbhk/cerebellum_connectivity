import pandas as pd
import os
import time

import connectivity.constants as const

def run():
    for exp in ['sc1', 'sc2']:  

        # initialise 
        dirs = const.Dirs(exp_name=exp)

        df_train = pd.read_csv(os.path.join(dirs.conn_train_dir, 'train_summary.csv'))

        # get model names
        models = df_train['name'].unique()

        df_all = pd.DataFrame()
        # loop over models
        for model in models:

            df = df_train.query(f'name=="{model}"')
            
            # get timestamp when folder was created
            df['timestamp'] = time.ctime(os.path.getctime(dirs.conn_train_dir / model))

            df_all = pd.concat([df_all, df])

        # save to disk
        df_all.to_csv(os.path.join(dirs.conn_train_dir, 'train_summary.csv'))

if __name__ == "__main__":
    run()
