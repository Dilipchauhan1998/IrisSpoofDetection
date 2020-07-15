# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from scnet import scnet50_v1d
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--train_dataset', default='home/dataset/train/', help='path to train dataset')
ap.add_argument('--test_intra_sensor_dataset', default='home/dataset/test_intra_sensor/', help='path to intra sensor test dataset')
ap.add_argument('--test_inter_sensor_dataset', default='home/dataset/test_inter_sensor/', help='path to inter sensor test dataset')
ap.add_argument('--load', default='home/output/model/model_300.pth', help='path of trained model')
ap.add_argument('--k', type=int, default=5, help='number of nearest neighbour')
ap.add_argument('--image_width', type=int, default=256, help='width of image')
ap.add_argument('--image_height', type=int, default=192, help='height of image')
ap.add_argument('--num_class', type=int, default=2, help='number of class')
ap.add_argument('--latent_feature_dim', type=int, default=128)
ap.add_argument('--batch_size', type=int, default=32)
args = ap.parse_args()


#Finetune model for claassification
class FineTuneModel(nn.Module):
    def __init__(self, original_model, latent_dim, num_classes):
        super(FineTuneModel, self).__init__()
        # Everything except the last linear layer
        self.conv_features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc_features =  nn.Sequential(
                            nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(512,latent_dim),
                            nn.ReLU()
                            #nn.LogSoftmax(dim=1) # For using NLLLoss()
                        )
        self.classifier = nn.Sequential(
                            nn.Dropout(0.4),
                            nn.Linear(latent_dim, num_classes)
                        )

    def forward(self, x):
        f = self.conv_features(x)        
        f = f.view(f.size(0), -1)
        fc_feat = self.fc_features(f)
        y = self.classifier(fc_feat)
        return fc_feat, y


#Transforming and loading training images
train_trans = transforms.Compose([
    transforms.Resize((args.image_height, args.image_width)),
    # This makes it into [0,1]
    transforms.ToTensor(),
    # This makes it into [-1,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(root=args.train_dataset, transform=train_trans)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

#Transforming and loading intra sensor testing images
test_intra_sensor_trans = transforms.Compose([
	transforms.Resize((args.image_height, args.image_width)),
	# This makes it into [0,1]
	transforms.ToTensor(),
	# This makes it into [-1,1]
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

test_intra_sensor_data = datasets.ImageFolder(root=args.test_intra_sensor_dataset, transform=test_intra_sensor_trans)
test_intra_sensor_loader = DataLoader(test_intra_sensor_data, batch_size=args.batch_size, shuffle=True, num_workers=4)


#Transforming and loading inter sensor testing images
test_inter_sensor_trans = transforms.Compose([
    transforms.Resize((args.image_height, args.image_width)),
    # This makes it into [0,1]
    transforms.ToTensor(),
    # This makes it into [-1,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

test_inter_sensor_data = datasets.ImageFolder(root=args.test_inter_sensor_dataset, transform=test_inter_sensor_trans)
test_inter_sensor_loader = DataLoader(test_inter_sensor_data, batch_size=args.batch_size, shuffle=True, num_workers=4)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = scnet50_v1d(pretrained=False)
model= FineTuneModel(model,args.latent_feature_dim, args.num_class)
model = model.to(device)
print("Loading pretrained model......")
model.load_state_dict(torch.load(args.load))
print("pretrained model loaded")


#######################################  Feature Extraction and Training KNN and Testing #####################################

#for storing feature and labels
x_train=[]
y_train=[]
x_intra_sensor_test=[]
y_intra_sensor_test=[]
x_inter_sensor_test=[]
y_inter_sensor_test=[]



def append_to_list(l,a):
    #print("hello ")
    for i in range(a.shape[0]):
        #print("a:i",a[i])
        l.append(a[i].tolist())
    return l    


print("extracting feature ..............")
model.eval()
with torch.no_grad():
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        feat,outputs = model(inputs)
        x_train=append_to_list(x_train,feat.cpu().detach().numpy())
        y_train=append_to_list(y_train,labels.cpu().detach().numpy())


model.eval()
with torch.no_grad():
    for i, data in enumerate(test_intra_sensor_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        feat,outputs = model(inputs)
        x_intra_sensor_test=append_to_list(x_intra_sensor_test,feat.cpu().detach().numpy())
        y_intra_sensor_test=append_to_list(y_intra_sensor_test,labels.cpu().detach().numpy())


model.eval()
with torch.no_grad():
    for i, data in enumerate(test_inter_sensor_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        feat,outputs = model(inputs)
        x_inter_sensor_test=append_to_list(x_inter_sensor_test,feat.cpu().detach().numpy())
        y_inter_sensor_test=append_to_list(y_inter_sensor_test,labels.cpu().detach().numpy())

x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
x_intra_sensor_test=np.asarray(x_intra_sensor_test)
y_intra_sensor_test=np.asarray(y_intra_sensor_test)
x_inter_sensor_test=np.asarray(x_inter_sensor_test)
y_inter_sensor_test=np.asarray(y_inter_sensor_test)

print("feature extraction completed")

print("KNN training started ......")
neigh = KNeighborsClassifier(n_neighbors=args.k)
neigh.fit(x_train,y_train)
print("KNN traing completed")

target_names = ['colored', 'Normal', 'transparent']

print("testing on intra sensor image......")
y_pred_intra_sensor=neigh.predict(x_intra_sensor_test)
#print(classification_report(y_intra_sensor_test, y_pred_intra_sensor, target_names=target_names))
print(confusion_matrix(y_intra_sensor_test, y_pred_intra_sensor))

print("testing on intra sensor images completed")


print("testing on inter sensor image......")
y_pred_inter_sensor=neigh.predict(x_inter_sensor_test)
#print(classification_report(y_inter_sensor_test, y_pred_inter_sensor, target_names=target_names))
print(confusion_matrix(y_inter_sensor_test, y_pred_inter_sensor))
print("testing on inter sensor image completed")
