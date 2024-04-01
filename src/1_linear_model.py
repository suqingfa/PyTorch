import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

rate = 0.01

w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)


def forward(x: torch.tensor):
    return x * w + b


def loss(x: float, y: float) -> torch.tensor:
    y_p = forward(x)
    return (y_p - y) ** 2


def main():
    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            l = loss(x, y)
            l.backward()
            w.data = w.data - rate * w.grad.data
            w.grad.zero_()

    print('w: ', w, 'b: ', b)
    print('predict: 4 -> ', forward(4))


if __name__ == '__main__':
    main()
