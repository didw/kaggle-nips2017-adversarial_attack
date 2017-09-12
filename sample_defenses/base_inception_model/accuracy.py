"""Accuracy of output file
"""

import os
import pandas as pd
import numpy as np

def main():
    f_true = 'images.csv'
    f_nt = 'nt_output.csv'
    f_t = 't_output.csv'

    df_images = pd.read_csv(f_true)
    df_nt_outputs = pd.read_csv(f_nt, header=None, names=[u'ImageId', u'PredLabel'], index_col=0)
    df_nt_outputs.index = map(lambda x: x.replace('.png', ''), df_nt_outputs.index)
    df_images = df_images.join(df_nt_outputs, on=u'ImageId')
    correct_count = np.sum(df_images['TrueLabel']==df_images['PredLabel'])
    print("Defense accuracy: %.1f%%"%(correct_count/10.0))
    print("NT success: %.1f%%"%(100.0-correct_count/10.0))

    if os.path.exists(f_t):
        df_t_outputs = pd.read_csv(f_t, header=None, names=[u'ImageId', u'PredTargetLabel'], index_col=0)
        df_images = df_images.join(df_t_outputs, on=u'ImageId')
        correct_count = np.sum(df_images['TargetLabel']==df_images['PredTargetLabel'])
        print("T success: %.1f%%"%(correct_count/10.0))

if __name__ == '__main__':
    main()
