import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, hidden_activation, output_activation):
        super(MLP, self).__init__()

        # Camada de entrada
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()

        # Camadas escondidas
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Camada de saída
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Função de ativação para camadas escondidas
        self.hidden_activation = self.get_activation(hidden_activation)

        # Função de ativação para camada de saída
        self.output_activation = self.get_activation(output_activation)

    def forward(self, x):
        x = self.hidden_activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError("Activation function '{}' not supported.".format(activation))
