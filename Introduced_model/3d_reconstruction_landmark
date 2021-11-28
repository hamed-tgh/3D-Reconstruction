from random import seed
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
import os
import numpy as np
from PIL import Image
import torch
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pthflops import count_ops


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

seed(1)

def coef2object(coef, mu, pc, ev , Flag = True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ev = torch.tensor(ev)
    mu = torch.tensor(mu)
    pc = torch.tensor(pc)
    ev = ev.to(device)
    mu = mu.to(device)
    pc = pc.to(device)
    if(Flag == False):
        coef = torch.tensor(coef)
        coef = coef.to(device)
    coef = torch.t(coef)
    b = coef * ev
    a= torch.matmul(pc , b)
    obj = mu + a
    return obj


def ploting(shape,tl , name = "Fig"):
    name = str.split(name , '.')[0]
    shape = shape.cpu().detach().numpy()
    x = shape[:,0]
    y = shape[:,1]
    z = shape[:,2]


    fig = plt.figure()

    ax = Axes3D(fig)


    # Creating plot
    ax.plot_trisurf(x,y,z, triangles=tl,linewidth = 0.2,
                antialiased = True)


    # plt.show()
    plt.savefig('temp'+ name + ".jpg" )



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Test(model , path):
    mu = sio.loadmat('01_MorphableModel.mat')['shapeMU']
    pc = sio.loadmat('01_MorphableModel.mat')['shapePC']
    ev = sio.loadmat('01_MorphableModel.mat')['shapeEV']
    tl = sio.loadmat('01_MorphableModel.mat')['tl']
    tl=tl-1
    list_test = os.listdir(path)
    running_loss = 0
    for i in range(len(list_test)):
        print(i)
        image = sio.loadmat(path + '/' + list_test[i])['landmarks_2d']
        image = np.reshape(image , (144,1))
        name = str.split(list_test[i] , '.')[0]
        image = TF.to_tensor(image)
        # normal = transforms.Normalize((0.5,), (0.5,))
        # image = normal(image)

        image -= image.min()
        image /= (image.max() - image.min())
        image = image.to(device)
        model = model.to(device)
        image = image.squeeze(2)
        label = sio.loadmat(path2 + '/' + list_test[i])['alpha']
        label = TF.to_tensor(label).to(device).float()

        with torch.no_grad():
            model.eval()
            output = model(image)

        output = output.detach().cpu().numpy()
        mu = sio.loadmat('01_MorphableModel.mat')['shapeMU']
        pc = sio.loadmat('01_MorphableModel.mat')['shapePC']
        ev = sio.loadmat('01_MorphableModel.mat')['shapeEV']
        tl = sio.loadmat('01_MorphableModel.mat')['tl']
        tl=tl-1
        print("going for 3d make")
        Shape = coef2object(output, mu, pc, ev)
        shape2 = torch.reshape(Shape,(int(len(Shape)/3),3))
        ploting(shape2 , tl)
        



class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(144, 100)
        self.batch1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.batch2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 30)
        self.batch3 = nn.BatchNorm1d(30)
        self.fc4 = nn.Linear(30,50)
        self.batch4 = nn.BatchNorm1d(50)
        self.fc5 = nn.Linear(100,100)
        self.batch5 = nn.BatchNorm1d(100)
        self.fc6 = nn.Linear(100,199)
        self.batch6 = nn.BatchNorm1d(199)



    def forward(self, x):
        # flatten all dimensions except the batch dimension
        x = x.float()
        temp = self.fc1(x)
        temp2 = F.leaky_relu(temp , negative_slope= 0.3)
        x = self.batch1(temp2)
        x1 = self.batch2(F.leaky_relu(self.fc2(x) , negative_slope= 0.3))
        x = self.batch3(F.leaky_relu(self.fc3(x1) , negative_slope= 0.3))
        x = self.batch4(F.leaky_relu(self.fc4(x) , negative_slope= 0.3))
        x = torch.cat((x1, x), -1)
        x = self.batch5(F.leaky_relu(self.fc5(x) , negative_slope= 0.3))
        x = self.batch6(self.fc6(x))

        return x




def TESTING():
    net = My_model()
    net.load_state_dict(torch.load('weight2.pth'))
    path_x_Test = 'Data/'
    Test(net , path_x_Test)

# Train()
#TESTING()

def calc_flops_fps_parameter():
  inp = torch.rand([1, 144]).to(device)
  net = My_model()
  net.to(device) 
  flops1 = count_ops(net, inp)
  print("parameter is :" , sum(p.numel() for p in net.parameters() if p.requires_grad))
  print("flops is " , flops1[0])
  import time
  start = time.time()


  for i in range(1000):
      with torch.no_grad():
          net.eval()
      #    image = TF.to_tensor(inp)
          put = net(inp)

  print("our FP/sec is:" , 1000 / (time.time() - start))



  start = time.time()
  for i in range(1000):
      with torch.no_grad():
          print(i)
          net.eval()
      #    image = TF.to_tensor(inp)
          put = net(inp)
      #    put = torch.reshape(put , (199,1))
          Shape = coef2object(put, mu, pc, ev)
          shape2 = torch.reshape(Shape,(int(len(Shape)/3),3))

  print("fps for our model with reconstruct :", 1000/ (time.time() - start))



if __name__ == "__main__":
  TESTING()
  calc_flops_fps_parameter()
  
