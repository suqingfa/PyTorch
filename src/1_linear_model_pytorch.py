import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    model = LinearModel()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Loss: {}'.format(epoch, loss))

    print('model: ', model.linear.weight.item(), model.linear.bias.item())
    x = torch.tensor([[4.0]])
    print('predict: ', x, model(x))


if __name__ == '__main__':
    main()
