from wsgiref.validate import validator
from torch.utils.data import random_split as rnd_splt
from chrom_dataset import Chromatin_Dataset
from torch.utils.data import DataLoader
from model import Chromatin_Network
import torch 
from torch import nn
from tqdm import tqdm 
import pdb
import matplotlib.pyplot as plt


def main():
   # Detect GPU or CPU
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   savePath = "model.pt"
   
   # Load the data
   c10_17_ctf_1 = DataLoader(Chromatin_Dataset( chromType="chr10-chr17", chromName="CTCF-1"), batch_size=256,shuffle=True)
   c10_17_H3K4me3_1 = DataLoader(Chromatin_Dataset( chromType="chr10-chr17", chromName="H3K4me3-1"), batch_size=32,shuffle=True)
   c10_17_p300_1 = DataLoader(Chromatin_Dataset( chromType="chr10-chr17", chromName="p300-1"), batch_size=32,shuffle=True)

   trainer = [c10_17_ctf_1]
   tester = [c10_17_H3K4me3_1]
   validator = [c10_17_p300_1]



   # Build the model 
   model = Chromatin_Network(input_shape=100)
   print(model)
   model = model.to(device)

   # Compile the model
   learning_rate = 1e-5
   epochs = 100
   optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
   loss_fn = nn.BCELoss()
   # used to log training loss per epcoh
   train_losses = []
   test_losses   = []

   # Train the model
   for epoch in tqdm(range(epochs)):
      # Set model to train mode and train on training data
      total_train_loss = 0
      for train_loader in trainer:
         train_loss = 0.0

         for data, label in train_loader:
            # Load data appropriatly
            data, label = data.to(device), label.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward Pass
            target = model(data)

            # Clean the data
            target = torch.flatten(target)

            # Calculate the Loss
            loss = loss_fn(target.to(torch.float32), label.to(torch.float32))

            # Calculate the gradient
            loss.backward()

            # Update Weight
            optimizer.step()

            # Calc loss
            train_loss += loss.item()
            total_train_loss += train_loss
      total_train_loss/=len(trainer)

      total_test_loss = 0
      for test_loader in tester:
         # Set model to validation 
         model.eval()
         test_loss = 0
         for test_data, test_label in test_loader:
            test_data , test_label = test_data.to(device) , test_label.to(device)

            target = model(test_data)
            test_label = test_label.unsqueeze(1)

            loss = loss_fn(target.to(torch.float32), test_label.to(torch.float32))
            test_loss += loss.item() 
         total_test_loss+=test_loss
      total_test_loss /= len(validator)

      train_losses.append(total_train_loss)
      test_losses.append(total_test_loss)
      
      plt.plot(train_losses)
      plt.savefig("training losses")
      plt.clf()
      plt.plot(test_losses)
      plt.savefig("testing loss")
      plt.clf()
      torch.save(model.state_dict(), savePath)
      print(f'Epoch {epoch+1} \t\t Training Loss: {total_train_loss} \t\t Testing Loss: {total_test_loss}')
      
   total_valid_loss = 0
   for valid_loader in validator:
      # Set model to validation 
      model.eval()
      valid_loss = 0
      for valid_data, valid_label in valid_loader:
         valid_data , valid_label = valid_data.to(device) , valid_label.to(device)

         target = model(valid_data)
         valid_label = valid_label.unsqueeze(1)

         loss = loss_fn(target.to(torch.float32), valid_label.to(torch.float32))
         valid_loss += loss.item() 
      total_valid_loss+=valid_loss
   total_valid_loss /= len(validator)
   print("TOTAL VALID LOSS:{}".format(total_valid_loss))

main()