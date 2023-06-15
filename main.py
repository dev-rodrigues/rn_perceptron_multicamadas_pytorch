import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from domain.mlp import MLP

if __name__ == '__main__':
    # Exemplo de uso
    input_size = 784  # Tamanho da entrada (para MNIST: 28x28 = 784)
    hidden_layers = 2  # Número de camadas escondidas
    hidden_size = 128  # Número de neurônios em cada camada escondida
    output_size = 10  # Tamanho da saída (número de classes em MNIST)
    hidden_activation = 'relu'  # Função de ativação para camadas escondidas
    output_activation = 'softmax'  # Função de ativação para camada de saída

    # Criação da rede neural
    model = MLP(input_size, hidden_layers, hidden_size, output_size, hidden_activation, output_activation)

    # Carregamento do conjunto de dados MNIST

    # Transformações nos dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Carregamento do conjunto de treinamento MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Definição da função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Treinamento da rede neural
    num_epochs = 10

    # Definir a transformação para padronização
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    for epoch in range(num_epochs):
        # Loop pelos mini-batches do conjunto de treinamento
        for images, labels in train_dataloader:
            images = normalize(images)

            # Forward pass
            outputs = model(images)

            # Cálculo da perda
            loss = criterion(outputs, labels)

            # Backward pass e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


