# DSRSN
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import functional
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from collections import Counter
import torch.utils.data as torch_data
import time
from torch.utils.data import Dataset
from DSRNS import DSRSN


#  定义读取预处理后的数据
def load_data():
    # 读取数据
    x = np.load('data_save/train_data2.npy')
    y = np.load('data_save/label2.npy')
    num = len(Counter(y))
    print("类别数量为：", num)
    print("数据shape为：", x.shape)
    return x, y, num


# 定义划分训练集、测试集
def create_train_data(x, y, ratio=0.8):
    """
    x:数据
    y:类别
    ratio:生成训练集比率
    """
    # 打乱顺序
    # 读取data矩阵的第一维数（图片的个数）
    num_example = x.shape[0]
    # 产生一个num_example范围，步长为1的序列
    arr = np.arange(num_example)
    # 调用函数，打乱顺序
    np.random.seed(99)
    np.random.shuffle(arr)
    # 按照打乱的顺序，重新排序
    arr_data = x[arr]
    arr_label = y[arr]
    # 将数据集分为训练集80%、测试集20%
    s = int(num_example * ratio)
    x_train = arr_data[:s]
    y_train = arr_label[:s]
    x_val = arr_data[s:]
    y_val = arr_label[s:]
    print("训练集shape", x_train.shape)
    print("训练集类别：", Counter(y_train))
    print("测试集shape", x_val.shape)
    print("测试集类别：", Counter(y_val))
    return x_train, y_train, x_val, y_val


class MyTrainData(Dataset):
    def __init__(self):
        self.x = list(zip(train_data, train_label))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class MyValData(Dataset):
    def __init__(self):
        self.x = list(zip(val_data, val_label))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify CWRU')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=20, type=int, help='batch size')
    parser.add_argument('-epochs', default=4, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='StepLR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=50, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    args = parser.parse_args([])
    # 读取数据
    data, label, label_count = load_data()
    # 生成训练集测试集,70%用作训练，30%用作测试
    train_data, train_label, val_data, val_label = create_train_data(data, label, 0.8)
    print("*" * 10)
    print("训练集数量：", len(train_label))
    print("测试集数量：", len(val_label))
    train_dataset = MyTrainData()
    train_loader = torch_data.DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    val_dataset = MyValData()
    val_loader = torch_data.DataLoader(val_dataset, batch_size=args.b, shuffle=True)
    net = DSRSN()
    T = args.T

    net.to(args.device)

    optimizer = None
    # 采用adma优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    lr_scheduler = None
    # 初始学习率为0.01，每50个epoch更新一个lr，变为原来的1/10
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = None
    # if args.amp:
    #     scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_lr_{args.lr}_')
    out_dir += f'StepLR_{args.step_size}_{args.gamma}'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    # writer = SummaryWriter(os.path.join(out_dir, 'fmnist_logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tqdm(train_loader):
            optimizer.zero_grad()
            frame = frame.float().to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            for t in range(T):
                if t == 0:
                    out_fr = net(frame)
                else:
                    out_fr += net(frame)
            out_fr = out_fr/T

            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        train_loss /= train_samples
        train_acc /= train_samples

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in val_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                for t in range(T):
                    if t == 0:
                        out_fr = net(frame)
                    else:
                        out_fr += net(frame)
                out_fr = out_fr / T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        # print(args)
        # print(out_dir)
        print('\n')
        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

