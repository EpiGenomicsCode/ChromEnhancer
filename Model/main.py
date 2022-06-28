from torch.utils.data import random_split as rnd_splt
from chrom_dataset import Chromatin_Dataset
from torch.utils.data import DataLoader
from model import Chromatin_Network
import torch 
from torch import nn

def main():
   # Detect GPU or CPU
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   # Load the data
   chrom_data = Chromatin_Dataset()
   # Split the data (we are doing a 75/25 split)
   train_percent = .75
   train_set, val_set = rnd_splt(chrom_data, [int(chrom_data.length*train_percent), chrom_data.length-int(chrom_data.length*train_percent)])
   
   # Convert datasets into DataLoaders
   train_loader = DataLoader(train_set, batch_size=32,shuffle=True)
   test_loader = DataLoader(val_set, batch_size=32, shuffle=True)
   
   # Build the model 
   model = Chromatin_Network(input_shape=chrom_data.input_shape)
   print(model)
   model = model.to(device)

   # Compile the model
   learning_rate = 0.01 
   epochs = 3
   optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
   loss_fn = nn.BCELoss()

   # Train the model
   for epoch in range(epochs):
      train_loss = 0.0
      # Set model to train mode
      model.train()

      for data, label in train_loader:

         # Load data appropriatly
         data, label = data.to(device), label.to(device)

         # Clear gradients
         optimizer.zero_grad()

         # Forward Pass
         target = model(data)

         # Clean the data
         label = label.unsqueeze(1)
         
         # # Calculate the Loss
         loss = loss_fn(target, label)

         # Calculate the gradient
         loss.backward()

         # Update Weight
         optimizer.step()

         # Calc loss
         train_loss += loss.item()

      # Set model to validation 
      model.eval()
      valid_loss = 0
      for valid_data, valid_label in test_loader:
         valid_data , valid_label = valid_data.to(device) , valid_label.to(device)

         target = model(valid_data)
         label = label.unsqueeze(1)

         loss = loss_fn(target, valid_label)
         valid_loss += loss.item() 

      print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(test_loader)}')

main()