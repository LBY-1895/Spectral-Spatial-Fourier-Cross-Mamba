import numpy as np
import h5py
import os
import argparse
from scipy.io import loadmat
from torch.utils import data as Data
import torch
import utils
from structure import *
from module import *
import matplotlib.pyplot as plt
from loadDat import *
import argparse
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter Setting
parser = argparse.ArgumentParser("MMA")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--distillation', type=int, default=1, help='distillation') 
parser.add_argument('--fusion', choices=['TTOA', 'trans','trans_s1_c1','trans_c1_s1','trans_TTOA_attention','trans_cs_sc'], default='TTOA', help='fusion method') 
parser.add_argument('--test_freq', type=int, default=1, help='number of evaluation')
parser.add_argument('--pred_flag', choices=['o_fuse','o_1','o_2','o_cnn','o_trans'], default='o_fuse', help='dataset to use')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')  
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston','Houston_nocloud','Berlin','Augsburg','Rochester'], default='Houston', help='dataset to use')
parser.add_argument('--model_name', choices=['MMA'], default='MMA', help='dataset to use')
parser.add_argument('--num_classes', type=int,choices=[11, 6, 15, 8, 7, 10], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=16, help='number of batch size')
parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')  
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

class LoadDat:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.test_loader, self.band1, self.band2 = self.load_data()
    def LoadDat_Patch():
        # load data
        if args.dataset == 'Houston':
            DataPath1 = './dataset/Houston.mat'
            DataPath2 = './dataset/LiDAR_MP.mat'
            Data1 = loadmat(DataPath1)['HSI']
            Data2 = loadmat(DataPath2)['LiDAR']
            TrLabel_10TIMES = loadmat(DataPath1)['trainlabels']  # 349*1905 
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
        else:

            TestPatch11, TestPatch21, TestLabel = utils.train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
            TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
            TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

            test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestLabel)
            test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
            loss_weight = torch.ones(int(TestLabel.max())+1).cuda()
            print("the weight of loss is ",loss_weight)
        [m1, n1, l1] = np.shape(Data1)

    
        Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
        height1, width1, band1 = Data1.shape
        height2, width2, band2 = Data2.shape
        # data size
        print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
        print("height2={0},width2={1},band2={2}".format(height2, width2, band2))

        return train_loader, test_loader, band1, band2

def validate_model(model, test_loader, device, num_classes=15):
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    total_samples_expected = len(test_loader.dataset)
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].long().to(device)
            
            outputs, classify_HSI, classify_LiDAR, mi_mamba, HSI_redundancy_loss = model(*inputs)
            #loss = criterion(outputs, labels)
            
            #total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            cm += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=np.arange(num_classes))

    metrics = {
        'overall_accuracy': accuracy_score(all_labels, all_preds) * 100,
        #'average_loss': total_loss / total_samples,
        'confusion_matrix': cm,
        'kappa_score': cohen_kappa_score(all_labels, all_preds),
        'class_accuracy': []
    }

    print(f"Total samples processed: {total_samples}")
    print(f"Expected total samples: {total_samples_expected}")

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        class_acc = tp / (tp + fn + 1e-10)  
        metrics['class_accuracy'].append({
            'class': i,
            'accuracy': class_acc * 100,
            'tp': tp,
            'fn': fn
        })

    print("=" * 60)
    print(f"{'CLASS':<10}{'ACC(%)':<10}{'TP':<10}{'FN':<10}")
    print("=" * 60)
    for cls in metrics['class_accuracy']:
        print(f"{cls['class']:<10}{cls['accuracy']:.2f}%{' '*2}{cls['tp']:<10}{cls['fn']:<10}")
    
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Kappa Score: {metrics['kappa_score']:.4f}")
    #print(f"Average Loss: {metrics['average_loss']:.6f}")
    print("=" * 60)

    return metrics   

if __name__ == '__main__':
    utils.setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_image, test_image, band1, band2 = LoadDat.LoadDat_Patch()
        # new_train_loader, final_test_loader = dataset_merge.merge_and_split_datasets(train_image, test_image, batch_size=16)
        model_path = "model-90houston.pth"  
        checkpoint = torch.load(model_path)  
#        model = module_fusion(l1=band1, l2=band2, maba_input_dim=64).to(device)
        model = module_fusion(l1=band1, l2=band2, maba_input_dim=64, fusion1_dim=512, across_heads=16).to(device)

        model.load_state_dict(checkpoint)
        model.eval()  
    
        metrics = validate_model(
            model=model,
            test_loader=test_image,
            device=device,
            num_classes=15
        )
    
