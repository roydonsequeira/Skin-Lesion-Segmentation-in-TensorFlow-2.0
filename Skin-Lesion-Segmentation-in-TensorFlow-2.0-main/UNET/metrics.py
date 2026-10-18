import torch

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.type(torch.float32)
        return x
    return torch.from_numpy(f(y_true.numpy(), y_pred.numpy())).float()

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
