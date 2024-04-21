import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, BertConfig, BertModel

class AudioClassifierWav2Vec(nn.Module):
    def __init__(self, wav2vec_model_name, num_labels, learning_rate=0.001, weight_decay=0.01):
        super(AudioClassifierWav2Vec, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        transformer_config = BertConfig(
            hidden_size=self.wav2vec.config.hidden_size,
            num_attention_heads=self.wav2vec.config.num_attention_heads,
            num_hidden_layers=1,
        )
        self.transformer = BertModel(transformer_config)
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, num_labels)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.7 ** (epoch // 5))
        
        # Loss history
        self.train_losses = []
        self.validation_losses = []

    def forward(self, input_values):
        with torch.no_grad():
            extracted_features = self.wav2vec(input_values).last_hidden_state

        transformer_output = self.transformer(inputs_embeds=extracted_features).last_hidden_state
        cls_output = transformer_output[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits

    def fit(self, train_loader, validation_loader, epochs, device):
        self.to(device)
        for epoch in range(epochs):
            self.train() 
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

            for batch in loop:
                input_values, labels = batch
                input_values = input_values.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(input_values)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update tqdm postfix to show current average training loss
                loop.set_postfix(loss=loss.item())

            # Calculate average training loss for the epoch
            avg_train_loss = self.validate(train_loader, device)
            self.train_losses.append(avg_train_loss)

            # avg_train_loss = running_loss / len(train_loader)
            # self.train_losses.append(avg_train_loss)

            # Validation phase
            avg_validation_loss = self.validate(validation_loader, device)
            self.validation_losses.append(avg_validation_loss)

            # Update learning rate
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
            for batch in dataloader:
                input_values, _ = batch
                input_values = input_values.to(device)

                # Forward pass
                outputs = self(input_values)
                _, predicted_indices = torch.max(outputs, dim=1)

                predictions.extend(predicted_indices.cpu().numpy())
        return predictions
