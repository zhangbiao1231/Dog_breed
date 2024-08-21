import torch.nn as nn
import torchvision
#TODO 建模函数，需要增加功能
def get_net(name):
    if name in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
        model = torchvision.models.__dict__[name](weights="DEFAULT")
    # with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
    # if Path(opt.model).is_file() or opt.model.endswith(".pt"): #TODO 加载模型、编辑模型
    #     model = attempt_load(opt.model, device="cpu", fuse=False)
    # elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
    #     model = torchvision.models.__dict__[opt.model](weights="DEFAULT" if pretrained else None)
    # else:
    #     m = hub.list("ultralytics/yolov5")  # + hub.list('pytorch/vision')  # models
    #         raise ModuleNotFoundError(f"--model {opt.model} not found. Available models are: \n" + "\n".join(m))
    model = nn.Sequential(model,nn.Sequential(nn.Linear(1000, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, 120)))
    return model
