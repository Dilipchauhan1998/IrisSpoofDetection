# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsummary import summary
from scnet import scnet50_v1d


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='home/dataset/train/', help='path of training dataset')
ap.add_argument('--model_dir', default='home/output/model/', help='path of model where it has to be saved')
ap.add_argument('--load', default='home/pre_trained_scnet50_v1d/scnet50_v1d-4109d1e1.pth', help='path of pre-trained scnet50_v1d model')
ap.add_argument('--plot', type=str, default='home/output/plot/plot_cli.png', help='path to output loss/accuracy plot')
ap.add_argument('--image_width', type=int, default=256, help='width of image')
ap.add_argument('--image_height', type=int, default=192, help='height of image')
ap.add_argument('--num_class', type=int, default=2, help='number of class')
ap.add_argument('--latent_feature_dim', type=int, default=128)
ap.add_argument('--optimizer', type=str , default='Adam',help='either SGD or Adam')
ap.add_argument('--learning_rate', type=float, default=.0001, help='learning rate of model')
ap.add_argument('--momentum', type=float, default=0.9)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--epochs', type=int, default=300)
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
	transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
	transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, shear=[-0.2,0.2]),
    transforms.ColorJitter(brightness=0.5),
	# This makes it into [0,1]
	transforms.ToTensor(),
	# This makes it into [-1,1]
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(root=args.dataset, transform=train_trans)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)


#check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pretrained scnet50_v1d Model on cuda
model = scnet50_v1d(pretrained=True, model_path=args.load)
model = model.to(device)

#make parameters  of layer1 and layer2 non trainable
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False    


#finetune model on cuda
model = FineTuneModel(model,args.latent_feature_dim, args.num_class)
model = model.to(device)

print("SCnet_classifier: ")
summary(model, input_size=(3, args.image_height, args.image_width))

#loss and optimizer
criterion = nn.CrossEntropyLoss()

if args.optimizer=="SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


print("training_started......")
train_losses = []
train_accuracy = []
min_train_loss=10000.0

for epoch in range(1, args.epochs+ 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0

    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        _,outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
          
   
    
    # calculate-average-losses
    train_loss = train_loss/len(train_loader.sampler)
    train_losses.append(train_loss)
    train_acc = train_correct/train_total
    train_accuracy.append(train_acc)
    
    
    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Acc: {:.6f}'.format(epoch, train_loss, train_acc))
    if epoch>10 and min_train_loss>train_loss:
        min_train_loss=train_loss
        torch.save(model.state_dict(), args.model_dir+'model_at_min_train_loss.pth')

    if epoch%50==0:
    	torch.save(model.state_dict(), args.model_dir+'model_'+str(epoch)+'.pth')
    

# plot the training loss and accuracy
N = np.arange(0, args.epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, np.asarray(train_losses), label="train_loss")
plt.plot(N, np.asarray(train_accuracy), label="train_acc")
plt.title("Training Loss and Accuracy on CLI Vista Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args.plot)