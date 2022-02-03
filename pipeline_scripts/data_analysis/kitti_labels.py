import os
import pandas as pd
import numpy as np
import sys

def main():
    pathtodir = sys.argv[1]
    column_names =['height','width','length']
    df_all = pd.DataFrame(columns=column_names)
    for filename in os.listdir(pathtodir):
        if filename.endswith('.txt'):
            try:
                data = pd.read_csv(os.path.join(pathtodir, filename), delim_whitespace=True, header=None)
                filter = data[data.iloc[:,0]=='Car'].iloc[:,8:11]
                if not filter.empty:
                    filter.columns = column_names
                    df_all = df_all.append(filter)
            except: print(filename, ' is empty and has been skipped.')
    all_scales(df_all)

def all_scales(df):
    df.to_csv(sys.argv[2] + sys.argv[3] + '.csv', header=True, index=False)

    full_scale = df.prod(axis=1)
    height = df.iloc[:,0]
    width = df.iloc[:,1]
    length = df.iloc[:,2]
    n = len(df.index)

    print(  "Full Scale: \n", full_scale.describe(), "\n"
            "Height: \n", height.describe(), "\n"
            "Length: \n", length.describe(), "\n"
            "Width: \n", width.describe(), "\n"
            "Number of instances studied: ", n)

if __name__ == '__main__':
    main()
