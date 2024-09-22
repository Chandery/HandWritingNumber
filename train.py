import torch
from dataloader import NumberDataset
import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-m',"--model", type=str, default="CCNet", help="Model to train")
    parser.add_argument('-l',"--init_lr", type=float, default=2e-4,help="Initial learning rate")
    parser.add_argument('-b',"--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('-e',"--epochs", type=int, default=10, help="Number of epochs")
    return parser.parse_args()

def train(args):
    # *取出命令行中的参数
    init_lr = args.init_lr
    bs = args.batch_size
    epochs = args.epochs
    model_name = args.model

    # *根据参数选择模型
    if(model_name == "CCNet"):
        from models.Net.CCNet import CCNet
        model = CCNet()

    # *设置训练集、验证集、测试集数据的路径
    img_path_train = "archive/train_imgs.npy"
    label_path_train = "archive/train_labels.npy"

    img_path_test = "archive/test_imgs.npy"
    label_path_test = "archive/test_labels.npy"

    # *根据数据划分不同的数据集并封装成Dataloader类
    train_dataset = NumberDataset(img_path_train, label_path_train, "train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=False)

    valid_dataset = NumberDataset(img_path_train, label_path_train, "val")
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False)

    test_dataset = NumberDataset(img_path_test, label_path_test, "test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # *选择合适的设备，如果有GPU就使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # *把模型放到设备上
    model = model.to(device)

    # *给模型初始化
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

    # *定义损失函数为交叉熵
    criterion = nn.CrossEntropyLoss()

    # *定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)
    
    # *学习率衰减的定义
    schedule = StepLR(optimizer, step_size=20, gamma=0.8)

    # *记录每个epoch的平局loss
    train_epochs_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []

    # *设置模型为训练模式
    model.train()

    for epoch in range(epochs):
        print("#"*20+f"epoch{epoch}/{epochs}"+"#"*20)
        
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []

        # *训练
        for idx, (img, label) in enumerate(train_dataloader, 1):
            
            img = img.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)

            output = model(img)

            optimizer.zero_grad()

            loss_now = criterion(output, label)

            loss_now.backward()
            optimizer.step()

            train_loss.append(loss_now.item())

            pred = output.argmax(dim=1)
            correct = pred.eq(label.argmax(dim=1)).sum().item()

            train_acc.append(correct/bs)

        train_epochs_loss.append(np.average(train_loss))
        train_epochs_acc.append(np.average(train_acc))

        # *验证
        with torch.no_grad():
            for idx, (img, label) in enumerate(valid_dataloader, 1):
                
                img = img.type(torch.float32).to(device)
                label = label.type(torch.float32).to(device)

                output = model(img)

                loss_now = criterion(output, label)

                valid_loss.append(loss_now.item())

                pred = output.argmax(dim=1)
                correct = pred.eq(label.argmax(dim=1)).sum().item()

                valid_acc.append(correct/bs)

        valid_epochs_loss.append(np.average(valid_loss))
        valid_epochs_acc.append(np.average(valid_acc))

        print(f"train_loss={train_epochs_loss[-1]}")
        print(f"valid_loss={valid_epochs_loss[-1]}")
        print(f"train_acc={train_epochs_acc[-1]}")
        print(f"valid_acc={valid_epochs_acc[-1]}")
        print(f"Learning rate={schedule.get_last_lr()[0]}")

        schedule.step()

        if (epoch+1)%20 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "epoch": epoch,
                },
                os.path.join('/home/cdy/HandWritingNumber/checkpoint', f"{model._get_name()}checkpoint_{epoch+1}.pt")
            )

    plt.figure(figsize=(5, 5))
    plt.title("loss")
    plt.plot(train_epochs_loss, label="train_loss")
    plt.plot(valid_epochs_loss, label="valid_loss")
    plt.plot(train_epochs_acc, label="train_acc")
    plt.plot(valid_epochs_acc, label="valid_acc")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()



if __name__ == '__main__':
    args = parse_args()
    # print(args)
    train(args)