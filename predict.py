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

    print(f"Accuracy={total_correct/total}")

    
if __name__ == '__main__':
    predict()
