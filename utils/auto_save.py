
import os
import torch

# for test outputs .csv
def export_to_csv(data_dir,labels,preds,output_csv_folder):
    ids = sorted(os.listdir(
        os.path.join(data_dir, 'test',
                     'unknown')))
    with open(output_csv_folder, 'w') as f:
        f.write('id,' + ','.join(
            labels) + '\n')
        for i, output in zip(ids, preds):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')
    print('export done.')