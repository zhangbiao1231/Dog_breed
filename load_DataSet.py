
import os
import torch
import torchvision
from d2l import torch as d2l
from augmentations import classify_augmentations as augment

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

demo = True
batch_size = 32 if demo else 128
valid_ratio = 0.1

transform_train = augment()
transform_test = augment(is_Train=False)
data_dir = '../dog-breed/data'
#加载数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
for X, y in train_iter:
        print(X.shape,y.shape)
        break
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = '../dog-breed/data'
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))