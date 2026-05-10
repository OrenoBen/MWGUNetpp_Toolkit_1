import argparse
import numpy as np
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
import importlib

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入src中的模块
from src.data.landslide4sense import LandslideDataset
from src.utils.metrics import eval_image

name_classes = ['Non-Landslide','Landslide']
epsilon = 1e-14

def importName(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(  ), [name])
    except ImportError:
        return None
    return vars(module)[name]

def get_arguments():

    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")
    
    # 使用用户指定的数据集路径
    parser.add_argument("--data_dir", type=str, default='C:\\Users\\hongbo_0\\Desktop\\U_net_new\\Landslide4Sense-2022-main\\landslide4sense',
                        help="dataset path.")
    parser.add_argument("--model_module", type=str, default='src.models.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='model name in given module')
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="width and height of input images.")                     
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")               
    parser.add_argument("--batch_size", type=int, default=8,  # 减小batch size以避免内存问题
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,  # Windows上设为0避免多进程问题
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    
    # Create network   
    model_import = importName(args.model_module, args.model_name)
    # LandslideDataset提供14波段图像，所以n_channels需要设为14
    model = model_import(n_classes=args.num_classes, n_channels=14)
    model.train()
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for training")
    else:
        print("GPU not available, using CPU for training")

    # 使用LandslideDataset类，参数与src/data/landslide4sense.py匹配
    input_size_tuple = (int(args.input_size.split(',')[1]), int(args.input_size.split(',')[0]))
    
    train_dataset = LandslideDataset(
        data_root=args.data_dir,
        split="train",
        transform=None,  # 使用默认的标准化和数据增强
        target_transform=None
    )
    
    # 简单地将数据集分成训练集和测试集（80%/20%）
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    
    # 打印数据集大小信息
    print(f"原始数据集总样本数: {train_size + test_size}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    src_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility,
        pin_memory=True
    )


    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,3))
    F1_best = 0.5    
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    for batch_id, src_data in enumerate(src_loader):
        if batch_id==args.num_steps_stop:
            break
        tem_time = time.time()
        model.train()
        optimizer.zero_grad()
        
        # 获取数据，适配LandslideDataset的返回格式
        images = src_data["image"]
        labels = src_data["mask"]
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        pred = model(images)
        
        pred_interp = interp(pred)
              
        # CE Loss，将mask从[0,1]转为类别索引[0,1]
        labels = labels.squeeze(1).long()  # 移除通道维度并转为长整型
        cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
        _, predict_labels = torch.max(pred_interp, 1)
        predict_labels = predict_labels.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        batch_oa = np.sum(predict_labels==labels)*1./len(labels.reshape(-1))

            
        hist[batch_id,0] = cross_entropy_loss_value.item()
        hist[batch_id,1] = batch_oa
        
        cross_entropy_loss_value.backward()
        optimizer.step()

        hist[batch_id,-1] = time.time() - tem_time

        if (batch_id+1) % 10 == 0: 
            print('Iter %d/%d Time: %.2f Batch_OA = %.1f cross_entropy_loss = %.3f'%(batch_id+1,args.num_steps,10*np.mean(hist[batch_id-9:batch_id+1,-1]),np.mean(hist[batch_id-9:batch_id+1,1])*100,np.mean(hist[batch_id-9:batch_id+1,0])))
           
        # evaluation per 500 iterations
        if (batch_id+1) % 500 == 0:            
            print('Testing..........')
            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
        
            for _, batch in enumerate(test_loader):  
                # 适配LandslideDataset的返回格式
                image = batch["image"]
                label = batch["mask"]
                name = batch["path"]
                
                label = label.squeeze().numpy()
                if torch.cuda.is_available():
                    image = image.float().cuda()
                else:
                    image = image.float()
                
                with torch.no_grad():
                    pred = model(image)

                _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                       
                                
                # 确保pred和label具有相同的维度
                if pred.ndim > label.ndim:
                    pred = pred.squeeze()
                elif label.ndim > pred.ndim:
                    label = label.squeeze()
                
                TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), label.reshape(-1), args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all
            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                if i==1:
                    print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
                    print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))                
                    print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))

            mF1 = np.mean(F1)            
            print('===> mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))

            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')                     
                model_name = 'batch'+repr(batch_id+1)+'_F1_'+repr(int(F1[1]*10000))+'.pth'
                torch.save(model.state_dict(), os.path.join(
                    snapshot_dir, model_name))
 
if __name__ == '__main__':
    main()