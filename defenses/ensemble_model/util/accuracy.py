"""Accuracy of output file
"""

import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='To get accuracy')
parser.add_argument('--input_file', dest='input_file', default='images.csv',
                    help='sum the integers (default: images.csv)')
parser.add_argument('--output_file', dest='output_file', default='output.csv',
                    help='sum the integers (default: images.csv)')


def main():
    args = parser.parse_args()
    f_true = args.input_file
    f_nt = args.output_file

    df_images = pd.read_csv(f_true)
    df_nt_outputs = pd.read_csv(f_nt, header=None, names=[u'ImageId', u'PredLabel'], index_col=0)
    df_nt_outputs.index = map(lambda x: x.replace('.png', ''), df_nt_outputs.index)
    df_images = df_images.join(df_nt_outputs, on=u'ImageId')
    correct_count = np.sum(df_images['TrueLabel']==df_images['PredLabel'])
    print("Defense accuracy: %.1f%%"%(correct_count/10.0))
    print("NT success: %.1f%%"%(100.0-correct_count/10.0))

if __name__ == '__main__':
    main()
