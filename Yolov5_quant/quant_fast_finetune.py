
from pytorch_nndct.apis import torch_quantizer

import argparse
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch

from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

device = torch.device("cpu")

BATCHSIZE=1
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default="datasets/VHR-10/val",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument('--model_dir',default="weights",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth')
parser.add_argument('--subset_len',default=1,type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument('--batch_size',default=BATCHSIZE,type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode',default='float',choices=['float', 'calib', 'test'],
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune',dest='fast_finetune',default=False,action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy',dest='deploy',action='store_true',default=False,
    help='export xmodel for deployment')
args, _ = parser.parse_known_args()


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def _make_grid(anchors,stride,nx=20, ny=20, i=0):
    # d = anchors[i].device
    t = anchors[i].dtype
    shape = 1, 3, ny, nx, 2
    y, x = torch.arange(ny, device=device, dtype=t), torch.arange(nx, device=device, dtype=t)
    yv, xv = torch.meshgrid(y,x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (anchors[i] * stride[i]).view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid


@torch.no_grad()
def evaluate(
        model,
        batch_size=BATCHSIZE,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/quant',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        plots=True,
        callbacks=Callbacks(),
        ):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    model = model.to(device)
    # stride, pt= model.stride, model.pt
    imgsz = check_img_size(imgsz, s=32)  # check image size

    # Configure
    model.eval()
    nc =10 # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    names=['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
            'ground track field', 'harbor', 'bridge', 'vehicle']
    dataloader = create_dataloader(args.data_dir, imgsz=imgsz, batch_size=batch_size, pad=0.5,stride=32,
                                       workers=workers, prefix=colorstr(f'quant: '))[0]

    seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(names)}
    class_map = list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy w h )
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # out, train_out =model(im)  # inference, loss outputs
        x=model(im)
        # ckpt = torch.load(file_path, map_location=device)
        # paras=ckpt['model'].yaml
        nc = 10  # 1
        no = nc+5
        anchors = [[1.25, 1.625, 2, 3.75, 4.125, 2.875], [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375], [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]]
        nl = 3  # number of detection layers
        na = 3  # number of anchors
        grid = [torch.zeros(1)] * nl  # init grid
        anchors = torch.tensor(anchors).float().view(nl, -1, 2)
        # register_buffer('anchors', a)
        anchor_grid=[torch.zeros(1)] * nl
        stride = [8, 16, 32]

        z = []
        for i in range(nl):
            bs, _, ny, nx, _no= x[i].shape
            # x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                    grid[i], anchor_grid[i] = _make_grid(anchors,stride,nx, ny, i)
                    # grid[i]= _make_grid(nx, ny, i)

            y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
            y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
        out,train_out = torch.cat(z, 1), x
        dt[1] += time_sync() - t2

        # Loss
        # loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta

        # NMS
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True)  # list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        dt[2] += time_sync() - t3

        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            # if single_cls:
            #     pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                # if plots:
                #     confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            # if save_txt:
            #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # if save_json:
            #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # if not training:
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        # confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON

    # Return results
    # model.float()  # for training
    # if not training:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps, t


def quantization(title='optimize',file_path=''):
    # data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    # model.load_state_dict(torch.load(file_path))
    # model=torch.load(file_path,map_location=device)['model'].float()
    model = DetectMultiBackend(file_path)
    # ckpt = torch.load(file_path, map_location=device)  # load checkpoint
    # model = Model(ckpt['model'].yaml, ch=3, nc=16).to(device)  # create
    # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load

    input = torch.randn([1, 3, 640, 640],device=device)
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device,bitwidth=8)

        quant_model = quantizer.quant_model
    quant_model = quant_model.to(device)

    # names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
    #          'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
    #          'swimming-pool', 'helicopter', 'container-crane']
    # val_loader = create_dataloader("dataset/quant_dir", imgsz=1024, batch_size=batch_size, names=names, pad=0.5,prefix=colorstr('val: '))[0]
    # ft_loader = create_dataloader("dataset/quant_dir", imgsz=1024, batch_size=batch_size, names=names, pad=0.5,prefix=colorstr('val: '))[0]
    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        # ft_loader, _ = load_data(
        #     subset_len=1024,
        #     train=False,
        #     batch_size=batch_size,
        #     sample_method='None',
        #     data_dir=data_dir,
        #     model_name=model_name)
        if quant_mode == 'calib':
            quantizer.fast_finetune(evaluate, (quant_model,))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    print(evaluate(model=quant_model))

    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print('\n\n ========== Calibration Completed! ==========\n\n')
    if deploy:
        quantizer.export_xmodel(deploy_check=False)
        print('\n\n ========== Xmodel export Completed! ==========\n\n')

if __name__ == '__main__':

    model_name = "yolov5s_VHR_967"
    file_path = os.path.join(args.model_dir, model_name + '.pt')

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))



