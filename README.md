# 手写数字识别

> *author：Chandery*
>
> *for 2024 Generate Learning*
>
> *date* 2024/7/29

## 数据集

[MNIST: 60,000 hand written number images (kaggle.com)](https://www.kaggle.com/datasets/rakuraku678/mnist-60000-hand-written-number-images)

![image-20240725160835863](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240725160835863.png)

```python
import numpy as np
import pandas as pd


data = pd.read_csv('archive/mnist_train.csv')

labels = np.array(data.iloc[:, 0])

imgs = np.array(data.iloc[:, 1:])


print(labels.dtype)
print(imgs.dtype)

print("label_shape=", labels.shape)
print("imgs_shape=", imgs.shape)
```

```python
int64
int64
label_shape= (59999,)
imgs_shape= (59999, 784)
```

> **记录**
>
> 读取csv使用pandas库中的read_csv
>
> 按照索引从表中取行和列用 pd.iloc[row_index, column_index]

<img src="https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240725160858829.png" alt="image-20240725160858829" style="zoom:50%;" />

## 数据检查和保存

```python
imgs = imgs.reshape((imgs.shape[0], 28, 28))

imgs = imgs[:,None,:,:]

print("labels_shape=",labels.shape)
print("imgs_shape=",imgs.shape)

print(np.isnan(labels).sum())
print(np.isnan(imgs).sum())

np.save('archive/train_labels.npy', labels)
np.save('archive/train_imgs.npy', imgs)
```

```python
labels_shape= (59999,)
imgs_shape= (59999, 1, 28, 28)
0
0
```

![image-20240725164546817](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240725164546817.png)

## DataLoader

```python
class NumberDataset(Dataset):
    def __init__(self, imgs_path, labels_path, split="train"):
        print("Loading dataset...")

        self.imgs = np.load(imgs_path)
        self.labels = np.load(labels_path)
        self.split = split

        print("Dataset loaded.")

        start_index = 0
        end_index = 0

        if(self.split == "train"):
            start_index = 0
            end_index = int(0.8*len(self.imgs))
        elif(self.split == "val"):
            start_index = int(0.8*len(self.imgs))
            end_index = len(self.imgs)
        elif(self.split == "test"):
            start_index = 0
            end_index = len(self.imgs)

        # *按照比例把数据集分成训练集、验证集和测试集
        self.imgs_list = self.imgs[start_index:end_index]
        self.labels_list = self.labels[start_index:end_index]

        print("Dataset split: ", self.split)
        print("Shape of images: ", self.imgs_list.shape) 
        
    def __getitem__(self, index):
        img = self.imgs_list[index]
        label_idx = self.labels_list[index]

        label_one_hot = self.one_hot(label_idx)

        # *数据归一化
        img = img/255.0

        return img, label_one_hot
    def __len__(self):
        return len(self.labels_list)
    
    def one_hot(self, label):
        res = np.zeros(10)
        res[label] = 1
        return res   
```

```python
Loading dataset...
Dataset loaded.
Dataset split:  train
Shape of images:  (47999, 1, 28, 28)
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.7804, 0.9882, 0.9882, 0.9882, 0.9882, 0.9882, 0.9882, 0.1686,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000], dtype=torch.float64)
tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64) # * 独热
Loading dataset...
Dataset loaded.
Dataset split:  val
Shape of images:  (12000, 1, 28, 28)
torch.Size([32, 1, 28, 28])
torch.Size([32, 10])
Loading dataset...
Dataset loaded.
Dataset split:  test
Shape of images:  (9999, 1, 28, 28)
torch.Size([32, 1, 28, 28])
torch.Size([32, 10])
```

### 独热编码的高级写法

使用到np.eye()

这个函数本来的作用是返回一个二维数组(N, M)，对角线的位置为1，其余地方为0

在深度学习中可以用下列语法

```python
np.eye(3)[[1,2,0,1]]
[[0. 1. 0.]
[0. 0. 1.]
[1. 0. 0.]
[0. 1. 0.]]
```

因此上面的独热过程可以写为

```python
label_one_hot = np.eye(10)[label_idx]
```



## 模型结构

自己随便写的模型结构，基本参考AlexNet

```python
class CCNet(nn.Module):
    def __init__(self):
        super(CCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 18, kernel_size=3, padding=1)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(288,10)
        self.relu4 = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) # *label.shape = (bs, classnumber) 所以要保证所有类别加起来为1，因此dim=1

    def forward(self, x):
        x = self.relu1(self.MaxPool1(self.conv1(x)))
        
        x = self.relu2(self.MaxPool2(self.conv2(x)))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu3(self.MaxPool3(x))

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu4(self.fc(x))

        x = self.softmax(x)

        return x
```

## train

> **超参数：**
>
> 1. init lr = 2e-4
> 2. batch_size = 32
> 3. epochs = 100
> 4. schedule: 80%/20 epochs

```python
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
    parser.add_argument('-e',"--epochs", type=int, default=100, help="Number of epochs")
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
            correct = pred.eq(label.argmax(dim=1)).sum().item() #* 计算准确率

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
```

![image-20240729023853416](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240729023853416.png)

![image-20240729023912250](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240729023912250.png)

![image-20240729021509223](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240729021509223.png)

## predict

```python
import dataloader
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt
from dataloader import NumberDataset
from models.Net.CCNet import CCNet
from sklearn.metrics import confusion_matrix
import os


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='all')
    # cm = confusion_matrix(y_true=label_true, y_pred=label_pred)
    # print(type(cm))
    plt.figure(figsize=(12, 6))

    # plt.subplot(121)

    plt.grid(False)
    plt.imshow(cm, cmap='YlGnBu')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i==j else (0, 0, 0) 
            # color = (0, 0, 0)
            value = '{:.2%}'.format(cm[j, i])  
            # value = cm[j, i]
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CCNet()
    model = model.to(device)

    bs = 32

    # *load model
    ckpt = torch.load('/home/cdy/HandWritingNumber/checkpoint/CCNetcheckpoint_100.pt') 
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_path_test = "archive/test_imgs.npy"
    label_path_test = "archive/test_labels.npy"

    test_dataset = NumberDataset(img_path_test, label_path_test, "test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


    pred = None
    gt = None
    total = 0
    total_correct = 0

    for idx,(img, label) in enumerate(test_dataloader):
        img = img.type(torch.float32).to(device)
        # print(feature.shape)
        
        # with torch.no_grad():
        output = model(img) # *(batch_size, num_class)
        output_pred = output.argmax(dim=1) # *(batch_size)

        label = label.argmax(dim=1)
    

        pred = output_pred if idx == 0 else torch.cat((pred, output_pred))
        gt = label if idx == 0 else torch.cat((gt, label))

        # print(output_pred.shape)
    
    # eval

    gt = gt.type(type(pred)).to('cpu')
    pred = pred.to('cpu')
    True_list = (pred == gt)
    total_correct = True_list.sum().item()
    total = len(gt)

    draw_confusion_matrix(gt, pred, ['0','1','2','3','4','5','6','7','8','9'], title="Confusion Matrix", pdf_save_path="confusion_matrix.png", dpi=100)

    cm = confusion_matrix(y_true=gt, y_pred=pred)

    
if __name__ == '__main__':
    predict()

```

![image-20240729024021475](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240729024021475.png)

![image-20240729022327949](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240729022327949.png)
