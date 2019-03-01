from fastai.vision import *

class PairStats(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.cat([
                        (x[0]+x[1]), 
                        (x[0]*x[1]), 
                        torch.abs(x[0]-x[1]), 
                        torch.pow(x[0]-x[1], 2)
                        ], 0).reshape(x[0].shape[0], 4, 1, -1)

class AbsDiff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.abs(x[0]-x[1])

class SqDiff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.pow(x[0]-x[1], 2)
                                       
class SiameseModel(nn.Module):
    def __init__(self, cnn_head, cnn_body):
        super().__init__()
        self.cnn_body = cnn_body
        self.cnn_head = cnn_head
        
    def forward(self, img1, img2):
        x1 = self.cnn_body(img1)
        x2 = self.cnn_body(img2)
        return self.cnn_head([x1, x2])