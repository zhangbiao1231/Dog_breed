import torch
import os

from utils.dataloaders import (dataLoader,make_dataSets)
from utils.auto_save import export_to_csv

def inference(model,test_iter, device):
    preds = []
    net = model.eval()
    for X, _ in test_iter:
        output = torch.nn.functional.softmax(net(X.to(device)), dim=1)
        preds.extend(output.cpu().detach().numpy())
    return preds
def do_evaluation(cfg,args,model):
    device = torch.device(cfg.MODEL.DEVICE)
    test_iter = dataLoader(cfg=cfg,
                           data_dir=args.data_dir,
                           batch_size=cfg.SOLVER.BATCH_SIZE,
                           folder= 'test',
                           is_Train=False,
                           is_Test=True)
    train_valid_ds = make_dataSets(
                           cfg=cfg,
                           data_dir=args.data_dir,
                           folder='train_valid',
                           is_Train=True)
    labels = train_valid_ds.classes
    print(f'num of labels: {len(labels)}')
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    preds = inference(model, test_iter,device)
    #===================inference done=================
    print(' ========== inference done ==========')
    print(f'export {args.export_csv_filename}...')
    export_to_csv(data_dir=args.data_dir,
                  labels = labels,
                  preds = preds,
                  output_csv_folder =os.path.join(output_folder,args.export_csv_filename))