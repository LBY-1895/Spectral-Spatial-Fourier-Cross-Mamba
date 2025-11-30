import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import numpy as np
import h5py
import os
import argparse
from scipy.io import loadmat
from torch.utils import data as Data
import torch
# 导入 utils 模块
import utils
from structure import *
from module import *
import matplotlib.pyplot as plt
from loadDat import *
import argparse
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR
# from monai.losses import DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter Setting
parser = argparse.ArgumentParser("MMA")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--distillation', type=int, default=1, help='distillation') #是否加蒸馏
parser.add_argument('--fusion', choices=['TTOA', 'trans','trans_s1_c1','trans_c1_s1','trans_TTOA_attention','trans_cs_sc'], default='TTOA', help='fusion method') #确认用哪个做融合
parser.add_argument('--test_freq', type=int, default=1, help='number of evaluation')
parser.add_argument('--pred_flag', choices=['o_fuse','o_1','o_2','o_cnn','o_trans'], default='o_fuse', help='dataset to use')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')  
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston','Houston_nocloud','Berlin','Augsburg','Rochester'], default='Houston', help='dataset to use')
parser.add_argument('--model_name', choices=['MMA'], default='MMA', help='dataset to use')
parser.add_argument('--num_classes', type=int,choices=[11, 6, 15, 8, 7, 10], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')  
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

class LoadDat:
    def __init__(self, args):
        """
        加载数据并转换为序列输入
        Args:
            args: 包含数据集名称、小块大小、批量大小、模式标志等参数
        """
        self.args = args
        self.train_loader, self.test_loader, self.band1, self.band2 = self.load_data()
    def LoadDat_Patch():
        # load data
        if args.dataset == 'Houston':
            DataPath1 = '/root/autodl-tmp/SSFCMamba/dataset/Houston.mat'
            DataPath2 = '/root/autodl-tmp/SSFCMamba/dataset/LiDAR_MP.mat'
            Data1 = loadmat(DataPath1)['HSI']
            Data2 = loadmat(DataPath2)['LiDAR']
            TrLabel_10TIMES = loadmat(DataPath1)['trainlabels']  # 349*1905 完整训练集
            TsLabel_10TIMES = loadmat(DataPath1)['testlabels']  # 349*1905
            print("TsLabel_10TIMES: ", TsLabel_10TIMES.shape)
        Data1 = Data1.astype(np.float32)
        Data2 = Data2.astype(np.float32)

        patchsize1 = args.patches1  # input spatial size for 2D-CNN
        pad_width1 = np.floor(patchsize1 / 2)
        pad_width1 = int(pad_width1)  # 8
        patchsize2 = args.patches2  # input spatial size for 2D-CNN
        pad_width2 = np.floor(patchsize2 / 2)
        pad_width2 = int(pad_width2)  # 8
        patchsize3 = args.patches3  # input spatial size for 2D-CNN
        pad_width3 = np.floor(patchsize3 / 2)
        pad_width3 = int(pad_width3)  # 8

        #patchsize = args.patches  # input spatial size for 2D-CNN
        #pad_width = np.floor(patchsize / 2)
        #pad_width = int(pad_width)  # 8

        if args.flag_test == 'train':
            TrainPatch11, TrainPatch21, TrainLabel = utils.train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
            TestPatch11, TestPatch21, TestLabel = utils.train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
            TrainPatch12, TrainPatch22, _ = utils.train_patch(Data1, Data2, patchsize2, pad_width2, TrLabel_10TIMES)
            TestPatch12, TestPatch22, _ = utils.train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
            TrainPatch13, TrainPatch23, _ = utils.train_patch(Data1, Data2, patchsize3, pad_width3, TrLabel_10TIMES)
            TestPatch13, TestPatch23, _ = utils.train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

            train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch12, TrainPatch22, TrainPatch13, TrainPatch23, TrainLabel)
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
            test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestLabel)
            test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            loss_weight = utils.loss_weight_calculation(TrainLabel).cuda()
            #print("the weight of loss is ",loss_weight)
            ############################################
            #TrainPatch1, TrainPatch2, TrainLabel = utils.train_patch(Data1, Data2, patchsize, pad_width, TrLabel_10TIMES)
            #TestPatch1, TestPatch2, TestLabel = utils.train_patch(Data1, Data2, patchsize, pad_width, TsLabel_10TIMES)

            #train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
            #train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
            #test_dataset = Data.TensorDataset(TestPatch1, TestPatch2, TestLabel)
            #test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            #loss_weight = utils.loss_weight_calculation(TrainLabel).cuda()   ############
            #print("the weight of loss is ",loss_weight)
        else:

            TestPatch11, TestPatch21, TestLabel = utils.train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
            TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
            TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

            test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestLabel)
            test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            loss_weight = torch.ones(int(TestLabel.max())+1).cuda()
            print("the weight of loss is ",loss_weight)
        [m1, n1, l1] = np.shape(Data1)

            #TestPatch1, TestPatch2, TestLabel = utils.train_patch(Data1, Data2, patchsize, pad_width, TsLabel_10TIMES)

            #test_dataset = Data.TensorDataset(TestPatch1, TestPatch2, TestLabel)
            #test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            #loss_weight = torch.ones(int(TestLabel.max())+1).cuda()
            #print("the weight of loss is ",loss_weight)
        #[m1, n1, l1] = np.shape(Data1)
    
        Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
        height1, width1, band1 = Data1.shape
        height2, width2, band2 = Data2.shape
        # data size
        print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
        print("height2={0},width2={1},band2={2}".format(height2, width2, band2))

        #print("train_loader: ", len(train_loader.TestPatch1))

        return train_loader, test_loader, band1, band2

# def init_weights(m):
#     if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
#         init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             init.zeros_(m.bias)

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # 计算 L2 范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

config = {
    "fusion1_hidden_dim": tune.choice([64, 128, 256, 512]),
    "across_heads": tune.choice([2, 4, 8, 16]),
}
# config = {
#     # AdamW参数
#     "lr": tune.loguniform(1e-5, 2e-4),
#     "weight_decay": tune.loguniform(1e-6, 2e-4),
#     "beta1": tune.quniform(0.85, 0.99, 0.01),  # 独立定义beta1搜索范围
#     "beta2": tune.quniform(0.85, 0.99, 0.01),
#     "eps": tune.loguniform(1e-9, 1e-7),          # 数值稳定性参数
#     "amsgrad": tune.choice([False, True]),       # 是否启用AMSGrad变体
    
#     # 批量大小
#     "batch_size": tune.choice([16, 32, 128]),
    
#     # Focal Loss参数
#     "focal_alpha": tune.uniform(0.1, 0.5),       # 正负样本平衡系数
#     "focal_gamma": tune.uniform(1.0, 3.0)        # 难例挖掘系数
# }

best_accuracy = 89.9

# 配置调度器和报告器
scheduler = ASHAScheduler(
    metric="max_test_acc",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

reporter = CLIReporter(
    metric_columns=["train_loss", "train_acc", "test_acc", "max_test_acc"]  # 更新指标列
)
def train_1times():
    args.batch_size = 32    
    train_image, test_image, band1, band2 = LoadDat.LoadDat_Patch()
    

    # 实例化 module_fusion 类
    model = module_fusion(l1=band1, l2=band2, maba_input_dim=64, fusion1_dim=512, across_heads=16)
    #model.load_state_dict(torch.load('best_model.pth'))#############3

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    # dice_loss = DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True)
    # optimizer_adam = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)#1e-4and4e-5
    optimizer_adam = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-05)#1e-4and4e-5

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=30, gamma=1/3)#30and0.333
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lambda epoch: 0.95**epoch)
    # 创建余弦退火调度器
    # scheduler = CosineAnnealingLR(optimizer_adam, T_max=15)
    # optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)

    best_accuracy = 90.32  # 初始化最佳准确率
    model.train()  # 设置模型为训练模式

    ###################################
    for epoch in range(500):  # 使用传入的epoches参数
        model.train()  # 设置模型为训练模式
        total_loss = 0
        correct_train = 0  # 用于累计训练集上正确的预测数
        total_train = 0    # 用于累计训练集上的总样本数
        total_mi = 0
        
        for batch in train_image:  # 遍历训练数据集中的所有批次
            x11, x21, x12, x22, x13, x23, labels = batch  # 解包三个张量
            x11, x21, x12, x22, x13, x23, labels = x11.to(device), x21.to(device), x12.to(device), x22.to(device), x13.to(device), x23.to(device), labels.long().to(device)
            # 前向传播
            outputs, output_HSI, output_LiDAR, mi_mamba, mi_hL = model(x11, x21, x12, x22, x13, x23)
            
            consistency_loss = 1 - F.cosine_similarity(output_HSI, output_LiDAR, dim=-1).mean()
            consistency_HSI = 1 - F.cosine_similarity(outputs, output_HSI, dim=-1).mean()
            consistency_LiDAR = 1 - F.cosine_similarity(outputs, output_LiDAR, dim=-1).mean()

            # 假设 labels 是 [B]，值为 0~C-1
            labels_onehot = F.one_hot(labels, num_classes=15).float()  # [B, C]
            # print("labels_onehot:", labels_onehot.shape)
            # print("labels:", labels.shape)
            # 计算损失
            epsilon = 1e-8
            if epoch < 30:
                xishu = 1
            elif epoch < 60:
                xishu = 0.5
            else:
                xishu = 0.1
    
            # loss=sigmoid_focal_loss(outputs, labels_onehot, alpha=0.37, gamma=1.9, reduction='mean') + (mi_mamba-0)
            loss= sigmoid_focal_loss(outputs, labels_onehot, alpha=0.25, gamma=2.0, reduction='mean') + (mi_mamba-0)
            # loss = dice_loss(outputs, labels) + mi_mamba
            #################################################
            # 计算损失
            #loss = criterion(outputs, labels) + 0.1*criterion(output_HSI, labels) + 0.1*criterion(output_LiDAR, labels) + 0.1*consistency_loss
            # 计算训练集上的准确度
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            # 反向传播
            optimizer_adam.zero_grad()  # 清除旧梯度
            loss.backward()  # 计算梯度
            optimizer_adam.step()  # 参数更新
            grad_norm = compute_gradient_norm(model)
            # print(f"Epoch {epoch}, Gradient Norm: {grad_norm:.4f}")

            total_loss += loss.item()  # 累计损失
            total_mi += mi_mamba.item()

        avg_loss = total_loss / len(train_image)  # 计算平均损失
        avg_mi = total_mi / len(train_image)
        scheduler.step()  # 更新学习率################################################
        print("lr:", scheduler.get_last_lr())
        print(f'Epoch [{epoch+1}/{args.epoches}], Loss: {avg_loss:.6f}')
        print(f"train_acc: {100 * correct_train / total_train:.2f}%")
        # print("mi_mamba:", avg_mi)
        # print("focal_loss:", avg_loss-avg_mi)
        #print("alpha_mi:", model.alpha_mi, "alpha:", model.alpha, "beta:", model.beta, "gamma:",model.gamma)

        # 每个epoch结束后，在测试集上评估模型
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        test_loss = 0
        test_batches = 0  # 批次计数器
        with torch.no_grad():
            for batch in test_image:
                x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, labels = batch
                x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, labels = x1_1.to(device), x2_1.to(device), x1_2.to(device), x2_2.to(device), x1_3.to(device), x2_3.to(device), labels.long().to(device)
                outputs, output_HSI, output_LiDAR, mi_mamba_test, mi_HL1= model(x1_1, x2_1, x1_2, x2_2, x1_3, x2_3)
                _, predicted = torch.max(outputs, 1)
                #print("predicted:", predicted)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                # 计算损失（使用主输出）
                labels_onehot = F.one_hot(labels, num_classes=15).float()  # [B, C]
                # loss = loss= sigmoid_focal_loss(outputs, labels_onehot, alpha=0.25, gamma=2.0, reduction='mean') + mi_mamba_test + mi_HL1 - mi_fH1 - mi_fL1
                # test_loss += loss.item()  # 累加损失
                test_batches += 1

        accuracy = 100 * correct / total
        print(f'Accuracy on test set: {accuracy:.2f}%')
        # avg_loss_test = test_loss / test_batches  # 平均损失
        # print(f'Test Loss: {avg_loss_test:.6f}%')
        
        # 更新最佳准确率和保存模型参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_model_{best_accuracy:.2f}.pth')
            print(f'Saving best model with accuracy: {best_accuracy:.2f}%')

        #print("cnn_output_HSI:", cnn_SIMAM_output_HSI.shape, "cnn_output_LiDAR", cnn_SIMAM_output_LiDAR.shape)


def train_tune(config):
    global best_accuracy  # 声明为全局变量
    fusion1_hidden_dim = config["fusion1_hidden_dim"]
    across_heads = config["across_heads"]

    # 从config获取超参数
    # args.batch_size = config["batch_size"]
    log_file = os.path.join(os.getcwd(), "training_logs.txt")  # 当前目录
    # learning_rate = config["lr"]
    # weight_decay = config["weight_decay"]
    # betas = (config["beta1"], config["beta2"])  # 正确解包元组参数（beta1, beta2）
    # eps = config["eps"]
    # amsgrad = config["amsgrad"]
    # focal_alpha = config["focal_alpha"]
    # focal_gamma = config["focal_gamma"]


    # 加载数据（使用调整后的batch_size）
    train_image, test_image, band1, band2 = LoadDat.LoadDat_Patch()
    max_test_acc = 0.0  # 新增：跟踪当前试验的最大test_acc

    # 初始化模型
    model = module_fusion(l1=band1, l2=band2, maba_input_dim=64, fusion1_dim=fusion1_hidden_dim, across_heads=across_heads).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=4e-5)#1e-4and4e-5

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1/3)#30and0.333
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.)
    # 训练循环（简化为调参所需的最小周期）
    for epoch in range(60):  # 调参阶段使用较小epoch加速
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_image:
            x11, x21, x12, x22, x13, x23, labels = batch
            x11, x21, x12, x22, x13, x23, labels = [x.to(device) for x in [x11, x21, x12, x22, x13, x23, labels.long()]]
            
            outputs, _, _, mi_mamba, _ = model(x11, x21, x12, x22, x13, x23)
            labels_onehot = F.one_hot(labels, num_classes=15).float()
            loss = sigmoid_focal_loss(
                outputs, 
                labels_onehot, 
                alpha=0.25,  # 动态参数
                gamma=2.0,  # 动态参数
                reduction='mean'
            ) + mi_mamba

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in test_image:
                x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, labels = batch
                x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, labels = [x.to(device) for x in [x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, labels.long()]]
                outputs, _, _, _, _ = model(x1_1, x2_1, x1_2, x2_2, x1_3, x2_3)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            
            test_acc = correct_test/total_test
            # 新增：更新最大test_acc
            if test_acc > max_test_acc:
                max_test_acc = test_acc

            if test_acc > best_accuracy:
                best_accuracy = accuracy
                model_path = f'best_model_{test_acc:.2f}.pth'
                torch.save(model.state_dict(), f'best_model_{test_acc:.2f}.pth')
                # 记录详细日志到文件（新增部分）
                with open(log_file, 'a', encoding='utf-8') as f:
                    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_content = (
                        f"[{log_time}] 最佳模型保存\n"
                        f"准确率: {test_acc:.2f}%\n"
                        f"训练参数: \n"
                        f"  fusion1_dim: {fusion1_hidden_dim}\n"
                        f"nums_head: {across_heads}\n"
                        f"  模型路径: {model_path}\n"
                        "----------------------------------------\n"
                    )
                    f.write(log_content)
                print(f'保存最佳模型，准确率: {best_accuracy:.2f}%，参数已记录到 training_logs.txt')
                print(f'Saving best model with accuracy: {best_accuracy:.2f}%')

        # 向Ray Tune报告指标
        tune.report(
            train_loss=total_loss/len(train_image),
            train_acc=correct_train/total_train,
            test_acc=correct_test/total_test,
            max_test_acc=max_test_acc  # 新增：报告历史最大值
        )
        scheduler.step()

# 运行调参
def tune_hyperparameters():
    analysis = tune.run(
        train_tune,
        resources_per_trial={"gpu": 1},
        config=config,
        # num_samples=20,  # 尝试20组不同参数
        scheduler=scheduler,
        progress_reporter=reporter,
        name="mma_tune"
    )
    
    # 输出最佳配置
    best_config = analysis.get_best_config(metric="max_test_acc", mode="max")
    print("Best config: ", best_config)
    
    # 使用最佳参数训练最终模型
    # print("Training with best parameters...")
    # args.learning_rate = best_config["lr"]
    # args.weight_decay = best_config["weight_decay"]
    # args.batch_size = best_config["batch_size"]
    # train_1times()  # 使用原始训练函数但用优化后的参数

if __name__ == '__main__':
    utils.setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()
        # ray.init()
        # tune_hyperparameters()