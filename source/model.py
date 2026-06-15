import torch
import torch.nn as nn
import torch.nn.Transformer as Transformer
import torch.optim as optim

class NeedleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeedleModel, self).__init__()
        self.fc1 = Transformer.Linear(input_size, hidden_size)
        self.fc2 = Transformer.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Transformer.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self, train_data, train_labels, epochs=100, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
        return self
    
    def predict(self, test_data):
        return self(test_data)
    
    def evaluate(self, test_data, test_labels):
        predictions = self.predict(test_data)
        return nn.MSELoss()(predictions, test_labels)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    model = NeedleModel(input_size=10, hidden_size=100, output_size=1)
    model.train(train_data, train_labels)
    model.save("model.pth")
    model.load("model.pth")
    print(model.predict(test_data))
    print(model.evaluate(test_data, test_labels))