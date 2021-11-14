import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=-1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, data, alpha):
        input_data = data.expand(data.data.shape[0], 3, 28, 28)
        
        feature = self.feature(input_data)
        
        feature = feature.view(-1, 50 * 4 * 4)
        
        class_output = self.class_classifier(feature)
        

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        
        domain_output = self.domain_classifier(reverse_feature)
        
        
        return class_output, domain_output


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def main():
    import torch
    img_size = 28
    #train_data = Imgdataset(train_file_img, csv_data_path, img_size, transform=[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    model = DANN()
    img = torch.ones((2, 3, 28, 28))

    class_output, domain_output = model(img, 0.1)
    print(domain_output)
if __name__ == '__main__':
    main()