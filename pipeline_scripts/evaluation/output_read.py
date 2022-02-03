import pandas as pd
import os
import sys

def main():
    dir_name = sys.argv[1]
    output_path = sys.argv[2]
    filename = sys.argv[3]

    df = pd.DataFrame(columns=['filename', 'metric', 'easy', 'medium', 'hard'])

    for file in os.listdir(dir_name):
        if file.endswith('.out'):
            with open(os.path.join(dir_name, file), encoding='utf8') as f:
                contents = f.read()
                contents_split = contents.splitlines()
                contents_split_again = [x.split(':')[1] for x in contents_split[20:22]]

                contents_split_again_again = [x.split(',') for x in contents_split_again]
                df.loc[len(df)] = [file] + ['BEV'] + contents_split_again_again[0][0:3]
                df.loc[len(df)] = [file] + ['AP3D'] + contents_split_again_again[1][0:3]

    df.to_csv(output_path + '/' + filename + '.csv')

if __name__ == '__main__':
    main()
