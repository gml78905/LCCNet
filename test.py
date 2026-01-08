import torch
if not hasattr(torch.nn, 'SyncBatchNorm'):
    torch.nn.SyncBatchNorm = torch.nn.BatchNorm2d
torch.backends.cudnn.enabled = False

from models.LCCNet import LCCNet

model = LCCNet(
    image_size=(256, 512),
    use_feat_from=1,
    res_num=18
)


print("before cuda")
model.cuda()
print("after cuda")

