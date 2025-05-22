from torch import nn


class MLP(nn.Module):

    def __init__(self, nb_hidden, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    