from torch import nn


class MLP(nn.Module):

    def __init__(self, nb_hidden, input_dim, output_dim, **kwargs):
        super(MLP, self).__init__()
        self.kwargs = kwargs
        self.model = nn.Sequential(
            nn.Linear(input_dim, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, output_dim),
            nn.Dropout(p=self.kwargs["dropout_rate"]),
        )

    def forward(self, x):
        return self.model(x)
