import os
import numpy as np
import pandas as pd
import sys

def main():
    pathtodir = sys.argv[1]
    output_dir = sys.argv[2]
    number_of_points_per_frame = []

    for filename in os.listdir(pathtodir):
        if filename.endswith('.bin'):
    
            cloud = np.fromfile(os.path.join(pathtodir, filename), dtype=np.float32).reshape((-1, 4))
            number_of_points_per_frame.append(len(cloud))
            print("processed file: ", filename)

    df = pd.DataFrame(number_of_points_per_frame, columns=["colummn"])
    df.to_csv(output_dir + 'number_of_points_per_frame_' + sys.argv[3] + '.csv', index=False)

if __name__ == '__main__':
    main()
