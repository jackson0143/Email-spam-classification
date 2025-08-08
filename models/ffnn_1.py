import torch.nn as nn
class SpamClassifier(nn.Module):

    def __init__(self,n_features,n_hidden=[128,64],n_output=1):
        super(SpamClassifier,self).__init__()
        
        in_features = n_features
        layers = []

        for hidden_layer in n_hidden:
            layers.append(nn.Linear(in_features,hidden_layer))
            layers.append(nn.ReLU())
            in_features = hidden_layer 
        layers.append(nn.Linear(in_features,n_output))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        
    def forward(self,x):
        x = self.model(x)
        return x


