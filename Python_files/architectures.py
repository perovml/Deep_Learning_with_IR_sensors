import torch
import torch.nn as nn
import torch.nn.functional as F

#1. Social distancing architectures
#for output size calculation: [(W−K+2P)/S]+1.
    
#The best performing version without interpolation
class CNN_base_pad(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)#in 8*8 - out 8*8 #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        self.conv2 = nn.Conv2d(8, 4, 3)# 8*8 - 6*6 
        self.conv3 = nn.Conv2d(4, 2, 2)# 6*6 - 5*5
        self.fc1 = nn.Linear(50, 16) 
        #self.fc2 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc2(x))
        return x



#Fully connected version of CNN
class FF_Net_long(nn.Module):
    def __init__(self):
        super().__init__() 
        #self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 1)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x
    

class CNN_int(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 2)# 16*16 - 14*14
        self.conv3 = nn.Conv2d(8, 4, 6)# 14*14 - 9*9
        self.conv4 = nn.Conv2d(4, 2, 4)# 9*9 - 6*6
        self.fc1 = nn.Linear(36*2, 32) 
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class CNN_int_small(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 2)# 16*16 - 14*14
        self.conv3 = nn.Conv2d(8, 4, 6)# 14*14 - 9*9
        self.conv4 = nn.Conv2d(4, 2, 4)# 9*9 - 6*6
        self.fc1 = nn.Linear(36*2, 16) 
        #self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_int_bn(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 2)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 6)# 14*14 - 9*9
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 2, 4)# 9*9 - 6*6
        self.bn4 = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(36*2, 32) 
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#small version CNN-IR with 3 by 3 kernel for the first layer examined in the paper
class CNN_IR_3by3(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 4, 5)# 10*10 - 6*6
        self.bn4 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(36*4, 16) 
        #self.fc2 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
        
#medium version CNN-IR with 7 by 7 kernel for the first layer examined in the paper
class CNN_IR_7by7(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 4, 5)# 10*10 - 6*6
        self.bn4 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(36*4, 16) 
        #self.fc2 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
#medium version CNN-IR with 3 by 3 kernel for the first layer and extra fc layer examined in the paper
class CNN_IR_3by3_3fc(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        #self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 4, 5)# 10*10 - 6*6
        self.bn4 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(36*4, 32) 
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#big version CNN-IR with 3 by 3 kernel for the first and second layers and extra fc layer examined in the paper
class CNN_IR_7by7_3fc(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        #self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 2)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 4, 5)# 10*10 - 6*6
        self.bn4 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(36*4, 32) 
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





#The version with pooling, performed slightly worse
class CNN_int_bn_pool(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)  # Input: 16*16 - Output: 16*16 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3)  # Input: 16*16 - Output: 15*15
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 7*7
        self.conv3 = nn.Conv2d(16, 4, 3)  # Input: 7*7 - Output: 5*5, feature maps reduced to 4
        self.bn3 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(5*5*4, 16)  # Adjusted for the flattened 7*7*4 feature map
        self.fc2 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




#The version the best suitable for tchecking the 'sanity' of the approach with GRAD_CAM CNN interpretability technique
class CNN_grad_cam_new_big(nn.Module):
    def __init__(self, class_number):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 3, 3)# 10*10 - 8*8
        self.bn4 = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(64*3, 16) 
        #self.fc2 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


#2. Multiple People Localization architectures

class CNN_detection_new(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)#in 16*16 - out 16*16 
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3)# 16*16 - 14*14
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 16, 5)# 14*14 - 10*10
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 1, 3)# 10*10 - 8*8
        self.bn4 = nn.BatchNorm2d(1)  # Batch Norm for 1 feature map


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class CNN_detection_flat(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)#in 8*8 - out 8*8 
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)# 8*8 - 8*8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 16, 3, 1, 1)# 8*8 - 8*8
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 1, 3, 1, 1)# 8*8 - 8*8
        self.bn4 = nn.BatchNorm2d(1)  # Batch Norm for 1 feature map


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class CNN_detection_flat_seq(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(8, 128, 3, 1, 1)#in 8*8 - out 8*8 
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)# 8*8 - 8*8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 16, 3, 1, 1)# 8*8 - 8*8
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 1, 3, 1, 1)# 8*8 - 8*8
        self.bn4 = nn.BatchNorm2d(1)  # Batch Norm for 1 feature map


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x



#UNET RELATED

import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



class UNET_IR(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[16, 32]):
        super(UNET_IR, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for i, feature in enumerate(reversed(features)):
            if i == 0:
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))
            else:
                self.ups.append(DoubleConv(feature*3, feature))   

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        start = 0
        #because of small resolution we make up/down sampling only with neighrors to the bottleneck
        x = self.ups[0](x)
        skip_connection = skip_connections[0]
        if x.shape != skip_connection.shape:
            x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.ups[1](concat_skip)
        start = 2

        for idx in range(start, len(self.ups)):
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx](concat_skip)
            #print(f'upbranch: {x.shape}, skip connection{skip_connection.shape}')


        return self.final_conv(x)






#Convolutional LSTM version of UNET with temporal and spatial awareness 
#The implementation of the ConvLSTM Cell from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        
        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b,
                                         image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


#Adaptation of DoubleCNN class:
class DoubleConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size=(3,3), bias=True):
        super(DoubleConvLSTM, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=in_channels,
                                  hidden_dim=[hidden_dim, hidden_dim],
                                  kernel_size=kernel_size,
                                  num_layers=2,
                                  batch_first=True,
                                  bias=bias,
                                  return_all_layers=False)
    def forward(self, x):
        # Assuming x is of shape (batch, seq_len, channels, height, width)
        #print(f'in double conv: {x.shape}')
        layer_output_list, _ = self.conv_lstm(x)
        # Take the output of the last layer
        x = layer_output_list[-1]  # Shape: (batch, seq_len, channels, height, width)
        # We take the output of the last timestep
        #x = x[:, -1, :, :, :]  # Shape: (batch, channels, height, width)
        #print(f'in double conv after first LSTM: {x.shape}')
        return x



#The LSTM where pooling is performed oly before the bottleneck
class T_UNET_IR(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[16, 32], hidden_dims=[16, 32], incl_botneck = False
    ):
        super(T_UNET_IR, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.incl_botneck = incl_botneck

        # Down part of UNET
        for i, feature in enumerate(features):
            self.downs.append(DoubleConvLSTM(in_channels, feature, hidden_dim=hidden_dims[i]))
            in_channels = feature
        
        # Up part of UNET
        for i, feature in enumerate(reversed(features)):
            if i == 0:
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))
            else:
                self.ups.append(DoubleConv(feature*3, feature))      
        if self.incl_botneck:
            self.bottleneck = DoubleConvLSTM(features[-1], features[-1]*2, hidden_dim=hidden_dims[-1]*2)
        else:
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x[:, -1, :, :, :])
        if self.incl_botneck == False:
            x = x[:, -1, :, :, :]
        x = self.pool(x)
        #print(f'before bottle neck: {x.shape}')
        x = self.bottleneck(x)
        if self.incl_botneck:
            x = x[:, -1, :, :, :]
        skip_connections = skip_connections[::-1]
        #print(f'after bottle neck: {x.shape}')
        x = self.ups[0](x)
        skip_connection = skip_connections[0]
        if x.shape != skip_connection.shape:
            x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.ups[1](concat_skip)
        start = 2

        for idx in range(start, len(self.ups)):
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx](concat_skip)
            #print(f'upbranch: {x.shape}, skip connection{skip_connection.shape}')


        return self.final_conv(x)
        

#3D Convolution CNN UNET

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TransitionTo2D(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(TransitionTo2D, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(depth, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.transition(x)
        # Squeeze the depth dimension
        x = torch.squeeze(x, dim=2)
        return x


class UNET_IR_3D(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, features=[16, 32], shrink=True, incl_botneck = False, depth = 8
    ):
        super(UNET_IR_3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.shrink = shrink
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.incl_botneck = incl_botneck

        # Transition layer from 3D to 2D
        self.transition_to_2d_bbn = TransitionTo2D(features[-1], features[-1], depth)
        self.transition_to_2d_abn = TransitionTo2D(features[-1]*2, features[-1]*2, depth)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            self.transitions.append(TransitionTo2D(feature, feature, depth))  # Transition for each skip connection
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            if self.shrink:
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))
            else:
                self.ups.append(DoubleConv(feature*3, feature))   
        if self.incl_botneck:
            self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        else:
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = x.permute(0, 2, 1, 3, 4)
        for down, transition in zip(self.downs, self.transitions):
            x = down(x)
            skip_connections.append(transition(x))
            if self.shrink:
                x = self.pool(x)

        if self.incl_botneck == False:
            x = self.transition_to_2d_bbn(x)
        #print(f'before bottle neck: {x.shape}')
        x = self.bottleneck(x)
        if self.incl_botneck:
            x = self.transition_to_2d_abn(x)
        skip_connections = skip_connections[::-1]
        if self.shrink:
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]
                if x.shape != skip_connection.shape:
                    # Adjust TF.resize to 3D if needed, or use another method for resizing
                    x = TF.resize(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)
        else:
            for idx in range(0, len(self.ups)):
                skip_connection = skip_connections[idx]
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx](concat_skip)

        return self.final_conv(x)