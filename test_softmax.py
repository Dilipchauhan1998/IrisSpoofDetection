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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--test_intra_sensor_dataset', default='home/dataset/test_intra_sensor/', help='path to intra sensor test dataset')
ap.add_argument('--test_inter_sensor_dataset', default='home/dataset/test_inter_sensor/', help='path to inter sensor test dataset')
ap.add_argument('--load', default='home/output/model/model_300.pth', help='path of trained model')
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

print("softmax testing started .......")
model.eval()
confusion_matrix_intra_sensor = torch.zeros(args.num_class, args.num_class)
with torch.no_grad():
    for i, data in enumerate(test_intra_sensor_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        _,outputs = model(inputs)
        _, predicted= torch.max(outputs.data, 1)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix_intra_sensor[t.long(), p.long()] += 1 

print("Confusion matrix intra sensor:",confusion_matrix_intra_sensor)
print("Accuracy intra sensor: ",confusion_matrix_intra_sensor.diag()/confusion_matrix_intra_sensor.sum(1))


model.eval()
confusion_matrix_inter_sensor = torch.zeros(args.num_class, args.num_class)
with torch.no_grad():
    for i, data in enumerate(test_inter_sensor_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        _,outputs = model(inputs)
        _, predicted= torch.max(outputs.data, 1)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix_inter_sensor[t.long(), p.long()] += 1 

print("Confusion matrix inter sensor:",confusion_matrix_inter_sensor)
print("Accuracy inter sensor: ",confusion_matrix_inter_sensor.diag()/confusion_matrix_inter_sensor.sum(1))