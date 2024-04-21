import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ComplexTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(ComplexTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = src.mean(dim=1)  # Global average pooling
        src = self.dropout(src)
        output = self.fc(src)
        return output

    def fit(self, train_loader, validation_loader, epochs, device):
        self.to(device)
        for epoch in range(epochs):
            self.train() 
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

            for input_values, labels in loop:
                input_values = input_values.to(device)
                labels = labels.to(device)

                outputs = self(input_values)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loop.set_postfix(loss=loss.item())

            avg_train_loss = self.validate(train_loader, device)
            self.train_losses.append(avg_train_loss)

            avg_validation_loss = self.validate(validation_loader, device)
            self.validation_losses.append(avg_validation_loss)

            self.scheduler.step()

    def validate(self, validation_loader, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():  
            for input_values, labels in validation_loader:
                input_values = input_values.to(device)
                labels = labels.to(device)
                outputs = self(input_values)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(validation_loader)
        return avg_loss

    def predict(self, dataloader, device):
        self.eval()
        predictions = []
        with torch.no_grad():
            for input_values, _ in dataloader:
                input_values = input_values.to(device)

                outputs = self(input_values)
                _, predicted_indices = torch.max(outputs, dim=1)

                predictions.extend(predicted_indices.cpu().numpy())
        return predictions