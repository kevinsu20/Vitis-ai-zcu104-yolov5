import os
import sys
from pathlib import Path
from threading import Thread
import torch

from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import create_dataloader
from utils.general import (box_iou, check_img_size,
                           colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, LOGGER, )
from utils.metrics import  ap_per_class
from utils.plots import output_to_target, plot_images


from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

_divider = '-------------------------------'


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)    #(1,1024,1024,3)
    output_ndim0=(1,3,80,80,15)
    output_ndim1=(1,3,40,40,15)
    output_ndim2=(1,3,20,20,15)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids = []
    ids_max = 20
    outputData = []

    for i in range(ids_max):
        outputData.append([np.zeros(output_ndim0, dtype=np.int8),np.zeros(output_ndim1, dtype=np.int8),np.zeros(output_ndim2, dtype=np.int8)])

    while count < n_of_images:
        if (count + batchSize <= n_of_images):
            runSize = batchSize
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count = count + runSize
        if count < n_of_images:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            out_q[write_index] = outputData[index]
            '''store output vectors '''
            # for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)

                # write_index += 1
        ids = []

device = torch.device("cpu")
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
        image_dir,
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.3,  # NMS IoU threshold
        workers=0,  # max dataloader workers (per RANK in DDP mode)
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/FPGA_out',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        plots=True,
        threads=1,
        model='VHR.xmodel',
        ):
    listimage = os.listdir(image_dir)
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    imgsz = check_img_size(imgsz, s=32)  # check image size
    # Configure
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    names = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
             'ground track field', 'harbor', 'bridge', 'vehicle']
    dataloader = create_dataloader(image_dir, imgsz=imgsz, batch_size=batch_size,  pad=0.5, stride=32,
                      workers=workers, prefix=colorstr(f'FPGA: '))[0]
                      
    names = {k: v for k, v in enumerate(names)}
    # class_map = list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    ''' preprocess images '''
    print(_divider)
    print('Pre-processing', runTotal, 'images...')
    img_in = []
    targets_temp=[]
    paths_temp=[]
    shapes_temp=[]
    img_temp=[]
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        im = im.to(device, non_blocking=True)
        im = im.float()  # uint8 to fp16/32
        # targets = targets.to(device)
        targets_temp.append(targets)
        paths_temp.append(paths)
        shapes_temp.append(shapes)
        img_temp.append(im)

        nb, _, height, width = im.shape
        # height_temp.append(height)
        # width_temp.append(width)

        # im /= 255  # 0 - 255 to 0.0 - 1.0
        # im=im.reshape(640,640,3)
        im = im * (1 / 255.0) * input_scale
        # im = im * (1 / 255.0) * input_scale
        # im = im * input_scale
        # im = im * (1 / 255.0)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # if batch_i == 0:
        #     print(f"im.shape={im.shape}")
        im = im.numpy().astype(np.int8)
        # nb, _, height, width = im.shape  # batch size, channels, height, width
        img_in.append(im)

    print(_divider)
    print('Starting', threads, 'threads...')
    threadAll = []
    start = 0
    for i in range(threads):
        if (i == threads - 1):
            end = len(img_in)
        else:
            end = start + (len(img_in) // threads)
        in_q = img_in[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, runTotal, timetotal))

    ''' post-processing '''
    nc = 10  # 1
    no = nc + 5
    anchors = [[1.25, 1.625, 2, 3.75, 4.125, 2.875], [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375],
               [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]]
    nl = 3  # number of detection layers
    na = 3  # number of anchors
    grid = [torch.zeros(1)] * nl  # init grid
    anchors = torch.tensor(anchors).float().view(nl, -1, 2)
    anchor_grid = [torch.zeros(1)] * nl
    stride = [8, 16, 32]

    for imgout_index in range(len(out_q)):

        x = [torch.from_numpy(out_q[imgout_index][0] ),
             torch.from_numpy(out_q[imgout_index][1] ),
             torch.from_numpy(out_q[imgout_index][2] ),
        ]

        z = []
        for i in range(3):
            bs, _, ny, nx, _no = x[i].shape
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                grid[i], anchor_grid[i] = _make_grid(anchors, stride, nx, ny, i)
                # grid[i]= _make_grid(nx, ny, i)

            y = x[i].sigmoid()  # (tensor): (b, self.na, h, w, self.no)
            y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
        out,train_out = torch.cat(z, 1), x
        # NMS
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True)  # list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)

        # im=torch.from_numpy(img_in[imgout_index])
        # im=im.reshape(1,3,640,640)

        im = img_temp[imgout_index]
        targets=targets_temp[imgout_index]
        paths=paths_temp[imgout_index]
        shapes=shapes_temp[imgout_index]
        targets[:, 2:] *= torch.tensor((640, 640, 640, 640), device=device)
        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
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

        # Plot images
        if plots:
            plot_images(im, targets, paths, save_dir / f'FPGA_batch{imgout_index}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(out), paths, save_dir / f'FPGA_batch{imgout_index}_pred.jpg', names)

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
    pf = '%20s' + '%11i' * 1 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all',  nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c],nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds

    # Return results
    # model.float()  # for training
    # if not training:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # return (mp, mr, map50, map), maps, t
    return


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--image_dir', type=str, default='datasets/VHR-10/val/images', help='Path to folder of images. Default is images')
    ap.add_argument('-t', '--threads', type=int, default=8, help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model', type=str, default='weights/VHR.xmodel',help='Path of xmodel. Default is zcu102.xmodel')
    args = ap.parse_args()

    print('Command line options:')
    print(' --image_dir : ', args.image_dir)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    evaluate(image_dir=args.image_dir, threads=args.threads, model=args.model)

if __name__ == '__main__':
    main()



