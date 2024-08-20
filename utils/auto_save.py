
import os
import torch

# for test outputs .csv
def export_to_csv(valid_dir,csv,pred,TEXT_LABELS):
    ids = sorted(os.listdir(
        os.path.join(valid_dir, 'unknown')))
    with open(csv, 'w') as f:
        f.write('id,' + ','.join(TEXT_LABELS) + '\n')
        for i, output in zip(ids, pred):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')