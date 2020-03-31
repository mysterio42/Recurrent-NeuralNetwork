import torch.nn as nn
import torch


class Recurrent(nn.Module):

    def __init__(self,in_dim,hidden_dim,layer_dim,out_dim):
        super(Recurrent, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        # (batch_dim,seq_dim,in_dim)
        self.rnn = nn.RNN(input_size=in_dim,
                          hidden_size=hidden_dim,
                          num_layers=layer_dim,
                          batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(in_features=hidden_dim,out_features=out_dim)

    def forward(self, x):
        # layer_dim batch_size hidden_dim
        h0 = torch.zeros(self.layer_dim,x.size(0),self.hidden_dim)

        out,hn = self.rnn(x,h0)

        # index hidden state of last time step
        # batch_dim=100 seq_dim=28 hidden_dim=100
        # out.size(0)   100,28,100
        # out[:,-1,:]   100,100

        out = self.fc(out[:,-1,:])
        # out.size() 100,10

        return out

