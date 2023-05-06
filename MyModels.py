
#CHANGES DONE IN CLASS MUST BE UPDATED ON coreConfig.py FILE 
import coreConfig as cc 
exec(cc.stmts) 

class RNN_GRU(nn.Module):
    def __init__(self , input_size, hidden_size, num_layers, output_size):
         
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True , bidirectional = True)
        self.fc = nn.Sequential(
                nn.Linear(hidden_size*2,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.gru(x,h0)
        out = self.fc(out[:, -1, :])        
        return out


    

class HybridModel(nn.Module):

    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.num_layers = 5
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.gru1 = nn.GRU(input_size=32, hidden_size=32, num_layers=self.num_layers, batch_first=True)  # Adding GRU layer 1

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.gru2 = nn.GRU(input_size=64, hidden_size=128, num_layers=self.num_layers, batch_first=True)  # Adding GRU layer 2

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25344, 128)  # Fixing the input size of linear layer to match the output of GRU layer 2
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x.to(next(self.parameters()).device))
        x = self.conv2(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)   # Reshape and permute for GRU layer 1


        h0 = torch.zeros(self.num_layers, x.size(0), 32).to(x.device)
        x, _ = self.gru1(x,h0)                                     # Pass through GRU layer 1
        x = x.permute(0, 2, 1).contiguous()                     # Reshape back to (batch_size, time_steps, channels)
        x = x.reshape(batch_size, channels, height, width)      # Reshape back to (batch_size, channels, height, width)
        x = self.conv3(x)
        batch_size, channels, height, width = x.size()
        x = x.reshape(batch_size, 64, -1).permute(0, 2, 1)      # Reshape and permute for GRU layer 2

        h1 = torch.zeros(self.num_layers, x.size(0), 128).to(x.device)
        x, _ = self.gru2(x,h1)                                     # Pass through GRU layer 2
        x = self.flatten(x)
        x = F.relu(self.linear1(x)) 
        logits = self.linear2(x)
        pred = self.softmax(logits)
        return pred

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        #self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        #self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x

if __name__ == '__main__' :
    model = HybridModel(*cc.models["HybridModel"]["params"])
    model = CNN_Net()
    x = torch.rand(32 , 1 , 64 , 173)

    pred = model(x) 
    print(pred)
    
    pass

    
    
