import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import onnx

from argparse import ArgumentParser
from tqdm import tqdm


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, target_size)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        logits = self.linear(lstm_out[-1])
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


def get_dataloader(args):
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    return train_loader, valid_loader


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--epochs", type=int, default=1, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--report", type=int, default=100, help="")

    parser.add_argument("--classes", type=int, default=10, help="")
    parser.add_argument("--input_dim", type=int, default=28, help="")
    parser.add_argument("--hidden_dim", type=int, default=28, help="")

    args = parser.parse_args()

    train_loader, valid_loader = get_dataloader(args)

    model = LSTM(input_dim=args.input_dim,
                 hidden_dim=args.hidden_dim,
                 target_size=args.classes)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        train_loss = 0.0
        model.train()
        for step, batch in enumerate(train_loader, 1):
            inputs, labels = batch
            inputs = inputs.squeeze().transpose(0, 1)
            opt.zero_grad()
            log_probs = model(inputs)
            # _, topk_ids = log_probs.topk(1, dim=-1)
            # labels = topk_ids.view(-1)
            loss = criterion(log_probs, labels)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            if step % args.report == 0:
                correct_pred = torch.eq(torch.argmax(log_probs, -1), labels)
                train_acc = torch.mean(correct_pred.float())
                print("Step " + str(step) + ", train Loss= " + \
                      "{:.4f}".format(loss) + ", train Accuracy= " + \
                      "{:.3f}".format(train_acc))
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = 0.0, 0.0
            valid_size = 0.0
            for step, batch in enumerate(valid_loader, 1):
                inputs, labels = batch
                inputs = inputs.squeeze().transpose(0, 1)
                log_probs = model(inputs)
                loss = criterion(log_probs, labels)

                valid_loss += loss.item()
                correct_pred = torch.eq(torch.argmax(log_probs, -1), labels)
                valid_acc += torch.sum(correct_pred.float()).item()
                valid_size += inputs.shape[1]
            print("valid Loss= " + "{:.4f}".format(valid_loss/step) +
                  ", valid Accuracy= " + "{:.3f}".format(valid_acc/valid_size))

    torch.save(model, "./lstm.pt")
    model = torch.load("./lstm.pt")
    model.eval()
    inputs = Variable(torch.randn(28, 128, 28))
    torch.onnx.export(model,
                      inputs,
                      "lstm.onnx", verbose=True)
                      # opset_version=10,
                      # do_constant_folding=True,  # 是否执行常量折叠优化
                      # input_names=["input"],  # 输入名
                      # output_names=["output"],  # 输出名
                      # dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                      #               "output": {0: "batch_size"}})


if __name__ == '__main__':
    main()
