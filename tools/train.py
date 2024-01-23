import os
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from core.model import Net
from core.datasets import MyData


def make_parser():
    parser = argparse.ArgumentParser("Training a net on MNIST")
    parser.add_argument("--work_dir", default="work_dirs", help="output data dirs")
    parser.add_argument("--epochs", default=10, help="number of training epochs")
    parser.add_argument("--batch", default=128, help="batch size")
    return parser


def init_seed():
    torch.manual_seed(97)
    torch.cuda.manual_seed_all(97)


def setenv():
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, epochs, batch_size, work_dir) -> None:
        self.epochs = epochs
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.model = Net().cuda()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_dataset, test_dataset = MyData(True), MyData(False)
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
        self.loss_smooth = 0

    def run(self):
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.evaluation(epoch)
            self.save_ckpt(epoch)

    def train_one_epoch(self):
        for xTrain, yTrain in self.train_loader:
            xTrain = Variable(xTrain).cuda()
            yTrain = Variable(yTrain).cuda()

            self.optimizer.zero_grad()
            y = self.model(xTrain)
            loss = self.loss(y, yTrain)
            loss.backward()
            self.optimizer.step()

            self.loss_smooth = 0.4 * self.loss_smooth + 0.6 * loss.item()

    def evaluation(self, epoch):
        with torch.no_grad():
            acc = 0
            n = 0
            for xTest, yTest in self.test_loader:
                xTest = Variable(xTest).cuda()
                yTest = Variable(yTest).cuda()
                y = self.model(xTest)
                acc += (
                    torch.sum(
                        torch.argmax(y, dim=1)
                        == torch.matmul(yTest, torch.arange(10).to("cuda:0"))
                    )
                    .cpu()
                    .numpy()
                )
                n += xTest.shape[0]
            print(
                "%s, epoch %2d, loss = %f, test acc = %.2f"
                % (datetime.now(), epoch + 1,  self.loss_smooth, acc / n)
            )

    def save_ckpt(self, epoch):
        ckpt_file = os.path.join(self.work_dir, "epoch%s_ckpt.pth" % epoch)
        torch.save(self.model.state_dict(), ckpt_file)


def main(args):
    init_seed()
    setenv()

    trainer = Trainer(args.epochs, args.batch, args.work_dir)
    trainer.run()


if __name__ == "__main__":
    args = make_parser()
    main(args)
    print("training finished!")
