# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
The problem is to develop a Recurrent Neural Network (RNN) model that can predict future stock prices using historical closing price data. The model should be trained on normalized data from the given datasets and accurately forecast stock trends.

<img width="1067" height="342" alt="image" src="https://github.com/user-attachments/assets/b18c9947-810f-423e-aadd-d7b0449dadc6" />

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: HARSHITHA V

### Register Number: 212223230074

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    return train_losses

train_losses = train_model(model, train_loader, criterion, optimizer, epochs=20)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="475" height="405" alt="image" src="https://github.com/user-attachments/assets/8ff3c5ce-1ebf-4c33-bca1-6147d0ad90a8" />


## True Stock Price, Predicted Stock Price vs time

<img width="703" height="462" alt="image" src="https://github.com/user-attachments/assets/4ae62e84-699b-4428-8ea0-a108b04afbf5" />


### Predictions
<img width="203" height="42" alt="image" src="https://github.com/user-attachments/assets/2c13d969-0c60-47f5-9596-b29fffb1137c" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.Include your result here
