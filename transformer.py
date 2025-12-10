import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig


class NPYInnerSplitDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, split_ratio=0.8, num_classes=25):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        self.samples = []

        self.classes = sorted(os.listdir(root_dir))[:num_classes]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_files = sorted(os.listdir(class_dir))[:2]

            for item in class_files:
                item_path = os.path.join(class_dir, item)
                data = np.load(item_path)
                num_samples = data.shape[0]

                split_idx = int(np.floor(split_ratio * num_samples))
                if self.train:
                    indices = np.arange(0, split_idx)
                else:
                    indices = np.arange(split_idx, num_samples)

                for idx in indices:
                    self.samples.append((item_path, idx, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, inner_idx, label = self.samples[idx]
        data = np.load(sample_path)
        sample = data[inner_idx]

        first_channel_sample = sample[0, :, :]
        real_part = np.real(first_channel_sample)
        imag_part = np.imag(first_channel_sample)
        combined_sample = np.stack([real_part, imag_part], axis=0)

        if self.transform:
            combined_sample = self.transform(combined_sample)

        return combined_sample, torch.tensor(label, dtype=torch.long)


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='bert-base-uncased'):
        super(TransformerClassifier, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        self.transformer = BertModel.from_pretrained(pretrained_model_name, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Define transformations if needed
transform = None

# Parameters
# data/wurenji/wurenji_stft_npy
root_dir = 'data/wurenji/classify_npy'
num_classes = 25
batch_size = 32
learning_rate = 1e-4
num_epochs = 100

# Create datasets
train_dataset = NPYInnerSplitDataset(root_dir=root_dir, transform=transform, train=True, split_ratio=0.8)
val_dataset = NPYInnerSplitDataset(root_dir=root_dir, transform=transform, train=False, split_ratio=0.8)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = TransformerClassifier(num_classes=num_classes).to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

    val_loss /= len(val_loader)
    accuracy = correct / total

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}, Accuracy: {accuracy}")

    if (epoch + 1) % 5 == 0:
        model.eval()  # Set model to evaluate mode
        correct = 0
        total = 0
        with torch.no_grad():  # In evaluation phase, we don't compute gradients
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation accuracy after epoch {epoch+1}: {accuracy:.2f}%')