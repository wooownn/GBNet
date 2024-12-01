import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.gbnet import Net
from utils.tdataloader import test_dataset
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure, DICEHandler, IOUHandler
import cv2
from tqdm import tqdm
import py_sod_metrics
sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/GBNet')
parser.add_argument('--name', type=str, default='GBNet')
opt = parser.parse_args()
file=open("evalresults.txt", "a")
file.write("***************************\n")
file.write(opt.name+"\n")
for pth in sorted(os.listdir(opt.pth_path)):
    pth = os.path.join(opt.pth_path,pth)
    print(pth)
    model = Net()
    model.load_state_dict(torch.load(pth))
    model.cuda()
    model.eval() 
    for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
        data_path = './data/TestDataset/{}/'.format(_data_name)
        save_path = './results/GBNet/{}/'.format(_data_name)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'edge/', exist_ok=True)
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            _, _, res,e = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
            e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
            e = e.data.cpu().numpy().squeeze()
            e = (e - e.min()) / (e.max() - e.min() + 1e-8)
            imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
        
    file.write(pth+"\n")
    method='GBNet'
    for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
        mask_root = './data/TestDataset/{}/GT'.format(_data_name)
        pred_root = './results/GBNet/{}/'.format(_data_name)
        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        
        FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        # 灰度数据指标
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
        "rec": py_sod_metrics.RecallHandler(**sample_gray),
        "fpr": py_sod_metrics.FPRHandler(**sample_gray),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
        "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
        "ber": py_sod_metrics.BERHandler(**sample_gray),
        "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
        "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
        "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
        "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
        "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
        # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
        "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
        "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
        "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
    }
)
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)
            FMv2.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]
        fmv2 = FMv2.get_results()
        

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
            "meaniou": fmv2["iou"]["dynamic"].mean(),
            "meandice": fmv2["dice"]["dynamic"].mean(),
        }
        print(results)
        file.write(method+' '+_data_name+' '+str(results)+'\n')
