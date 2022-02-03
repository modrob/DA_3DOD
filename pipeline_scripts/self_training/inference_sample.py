import pandas as pd
import os
import sys

def read_scores(dir_path):

    df = pd.DataFrame(columns=['filename','line_position','score'])

    for file in os.listdir(dir_path):
        if file.endswith('.txt'):
            with open(os.path.join(dir_path,file)) as f:
                content_list = [x.strip() for x in f.readlines()]
                content_list_split = [x.split() for x in content_list]

                for x in content_list_split:
                    df.loc[len(df)] = [file] + [content_list_split.index(x)] + [x[-1]]

    df.score = pd.to_numeric(df.score)

    return(df)

def filter_inf_samples(df, dir_path, output_file):
    threshold_bottom = df.score.quantile(0.8)

    #filter to have at least a certain quality on all predictions
    filter_bottom = df[df.score < threshold_bottom]
    drop_files = filter_bottom.filename.unique().tolist()

    inf_samples = df[~df.filename.isin(drop_files)].reset_index(drop=True)

    #kept files
    print('Number of files which meet the lower threshold: ', len(inf_samples.filename.unique()))

    #filter to have at least one sample which is above average
    threshold_top = df.score.quantile(0.9)  # as of You et al.
    filter_top = inf_samples[inf_samples.score > threshold_top]

    keep_files = filter_top.filename.unique().tolist()

    inf_samples_final = inf_samples[inf_samples.filename.isin(keep_files)].reset_index(drop=True)

    #kept files
    print('Number of files which meet the lower AND top threshold: ', len(inf_samples_final.filename.unique()))

    #write list
    textfile = open(dir_path+ '/'+output_file+'.txt', "w")
    for element in inf_samples_final.filename.unique().tolist():
        textfile.write(os.path.splitext(element)[0] + "\n")
    textfile.close()

if __name__ == '__main__':
    df_read = read_scores(sys.argv[1])
    filter_inf_samples(df_read, sys.argv[1], sys.argv[2])
