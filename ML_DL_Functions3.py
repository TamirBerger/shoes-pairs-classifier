import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        n = int(12)
        kernel_size = int(5)
        padding = int((kernel_size - 1) / 2)
        
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=kernel_size, stride=2, padding=padding)
        self.dropout1 = nn.Dropout(0)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        # max pooling
        self.dropout2 = nn.Dropout(0)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, stride=2, padding=padding)
        self.dropout3 = nn.Dropout(0)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        # max pooling
        self.dropout4 = nn.Dropout(0)
        self.fc1 = nn.Linear(in_features=8*n*28*14, out_features=100)
        self.dropout5 = nn.Dropout(0)
        self.fc2 = nn.Linear(in_features=100, out_features=2)

    def forward(self,inp):
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # Move all parameters tensors to the same device as the inp tensor
        for param in self.parameters():
            param.data = param.data.to(inp.device)
        
        n = int(12)

        #inp = self.dropout(inp)
        out = self.conv1(inp)
        out = nn.functional.relu(out)

        #out = self.dropout1(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        #out = self.dropout2(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)

        #out = self.dropout3(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        #out = self.dropout4(out)
        out = out.contiguous().view(-1, 8*n*28*14)
        out = self.fc1(out)
        out = nn.functional.relu(out)
        
        #out = self.dropout5(out)
        out = self.fc2(out)

        return out


class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = int(12)
        kernel_size = int(5)
        padding = int((kernel_size - 1) / 2)
        
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, stride=2, padding=padding)
        self.dropout1 = nn.Dropout(0)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        # max pooling
        self.dropout2 = nn.Dropout(0)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, stride=2, padding=padding)
        self.dropout3 = nn.Dropout(0)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        # max pooling
        self.dropout4 = nn.Dropout(0)
        self.fc1 = nn.Linear(in_features=8*n*14*14, out_features=100)
        self.dropout5 = nn.Dropout(0)
        self.fc2 = nn.Linear(in_features=100, out_features=2)
        

    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        
        # Move all parametes tensors to the same device as the inp tensor
        for param in self.parameters():
            param.data = param.data.to(inp.device)
        
        n = int(12)
        inp = torch.cat((inp[:, :, :224, :], inp[:, :, 224:, :]), dim=1)

        inp = self.dropout(inp)
        out = self.conv1(inp)
        out = nn.functional.relu(out)

        out = self.dropout1(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.dropout2(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)

        out = self.dropout3(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.dropout4(out)
        out = out.contiguous().view(-1, 8*n*14*14)
        out = self.fc1(out)
        out = nn.functional.relu(out)
        
        out = self.dropout5(out)
        out = self.fc2(out)

        return out
        