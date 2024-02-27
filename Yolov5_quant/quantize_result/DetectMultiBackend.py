# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class DetectMultiBackend(torch.nn.Module):
    def __init__(self):
        super(DetectMultiBackend, self).__init__()
        self.module_0 = py_nndct.nn.Input() #DetectMultiBackend::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[6, 6], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[0]/Conv2d[conv]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[0]/ReLU[act]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[1]/Conv2d[conv]/input.4
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[1]/ReLU[act]/input.5
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv1]/Conv2d[conv]/input.6
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv1]/ReLU[act]/input.7
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.8
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.9
        self.module_9 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.10
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/7996
        self.module_11 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/7998
        self.module_12 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv2]/Conv2d[conv]/input.11
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv2]/ReLU[act]/8018
        self.module_14 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/input.12
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv3]/Conv2d[conv]/input.13
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[2]/Conv[cv3]/ReLU[act]/input.14
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[3]/Conv2d[conv]/input.15
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[3]/ReLU[act]/input.16
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv1]/Conv2d[conv]/input.17
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv1]/ReLU[act]/input.18
        self.module_21 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.19
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.20
        self.module_23 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.21
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8121
        self.module_25 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/input.22
        self.module_26 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.23
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.24
        self.module_28 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.25
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/8163
        self.module_30 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/8165
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv2]/Conv2d[conv]/input.26
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv2]/ReLU[act]/8185
        self.module_33 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/input.27
        self.module_34 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv3]/Conv2d[conv]/input.28
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[4]/Conv[cv3]/ReLU[act]/input.29
        self.module_36 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[5]/Conv2d[conv]/input.30
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[5]/ReLU[act]/input.31
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv1]/Conv2d[conv]/input.32
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv1]/ReLU[act]/input.33
        self.module_40 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.34
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.35
        self.module_42 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.36
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8288
        self.module_44 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/input.37
        self.module_45 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.38
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.39
        self.module_47 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.40
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/8330
        self.module_49 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/input.41
        self.module_50 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.42
        self.module_51 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/ReLU[act]/input.43
        self.module_52 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.44
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/ReLU[act]/8372
        self.module_54 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/8374
        self.module_55 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv2]/Conv2d[conv]/input.45
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv2]/ReLU[act]/8394
        self.module_57 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/input.46
        self.module_58 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv3]/Conv2d[conv]/input.47
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[6]/Conv[cv3]/ReLU[act]/input.48
        self.module_60 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[7]/Conv2d[conv]/input.49
        self.module_61 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[7]/ReLU[act]/input.50
        self.module_62 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv1]/Conv2d[conv]/input.51
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv1]/ReLU[act]/input.52
        self.module_64 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.53
        self.module_65 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.54
        self.module_66 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.55
        self.module_67 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8497
        self.module_68 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/8499
        self.module_69 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv2]/Conv2d[conv]/input.56
        self.module_70 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv2]/ReLU[act]/8519
        self.module_71 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/input.57
        self.module_72 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv3]/Conv2d[conv]/input.58
        self.module_73 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[8]/Conv[cv3]/ReLU[act]/input.59
        self.module_74 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/Conv[cv1]/Conv2d[conv]/input.60
        self.module_75 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/Conv[cv1]/ReLU[act]/8562
        self.module_76 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/8576
        self.module_77 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/8590
        self.module_78 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/8604
        self.module_79 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/input.61
        self.module_80 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/Conv[cv2]/Conv2d[conv]/input.62
        self.module_81 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/SPPF[model]/SPPF[9]/Conv[cv2]/ReLU[act]/input.63
        self.module_82 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[10]/Conv2d[conv]/input.64
        self.module_83 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[10]/ReLU[act]/input.65
        self.module_84 = py_nndct.nn.Interpolate() #DetectMultiBackend::DetectMultiBackend/Model[model]/Upsample[model]/Upsample[11]/8652
        self.module_85 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/Concat[model]/Concat[12]/input.66
        self.module_86 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv1]/Conv2d[conv]/input.67
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv1]/ReLU[act]/input.68
        self.module_88 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.69
        self.module_89 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.70
        self.module_90 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.71
        self.module_91 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8715
        self.module_92 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv2]/Conv2d[conv]/input.72
        self.module_93 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv2]/ReLU[act]/8735
        self.module_94 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/input.73
        self.module_95 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv3]/Conv2d[conv]/input.74
        self.module_96 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[13]/Conv[cv3]/ReLU[act]/input.75
        self.module_97 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[14]/Conv2d[conv]/input.76
        self.module_98 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[14]/ReLU[act]/input.77
        self.module_99 = py_nndct.nn.Interpolate() #DetectMultiBackend::DetectMultiBackend/Model[model]/Upsample[model]/Upsample[15]/8783
        self.module_100 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/Concat[model]/Concat[16]/input.78
        self.module_101 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv1]/Conv2d[conv]/input.79
        self.module_102 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv1]/ReLU[act]/input.80
        self.module_103 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.81
        self.module_104 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.82
        self.module_105 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.83
        self.module_106 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8846
        self.module_107 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv2]/Conv2d[conv]/input.84
        self.module_108 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv2]/ReLU[act]/8866
        self.module_109 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/input.85
        self.module_110 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv3]/Conv2d[conv]/input.86
        self.module_111 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[17]/Conv[cv3]/ReLU[act]/input.87
        self.module_112 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[18]/Conv2d[conv]/input.88
        self.module_113 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[18]/ReLU[act]/8909
        self.module_114 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/Concat[model]/Concat[19]/input.89
        self.module_115 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv1]/Conv2d[conv]/input.90
        self.module_116 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv1]/ReLU[act]/input.91
        self.module_117 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.92
        self.module_118 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.93
        self.module_119 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.94
        self.module_120 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/8972
        self.module_121 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv2]/Conv2d[conv]/input.95
        self.module_122 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv2]/ReLU[act]/8992
        self.module_123 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/input.96
        self.module_124 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv3]/Conv2d[conv]/input.97
        self.module_125 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[20]/Conv[cv3]/ReLU[act]/input.98
        self.module_126 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[21]/Conv2d[conv]/input.99
        self.module_127 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Conv[model]/Conv[21]/ReLU[act]/9035
        self.module_128 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/Concat[model]/Concat[22]/input.100
        self.module_129 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv1]/Conv2d[conv]/input.101
        self.module_130 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv1]/ReLU[act]/input.102
        self.module_131 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.103
        self.module_132 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.104
        self.module_133 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.105
        self.module_134 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/9098
        self.module_135 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv2]/Conv2d[conv]/input.106
        self.module_136 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv2]/ReLU[act]/9118
        self.module_137 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/input.107
        self.module_138 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv3]/Conv2d[conv]/input.108
        self.module_139 = py_nndct.nn.ReLU(inplace=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/C3[model]/C3[23]/Conv[cv3]/ReLU[act]/input
        self.module_140 = py_nndct.nn.Conv2d(in_channels=128, out_channels=45, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[0]/9160
        self.module_141 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9162
        self.module_142 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9169
        self.module_143 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9173
        self.module_144 = py_nndct.nn.Module('reshape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9179
        self.module_145 = py_nndct.nn.Module('permute') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9186
        self.module_146 = py_nndct.nn.Module('contiguous') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9188
        self.module_147 = py_nndct.nn.Conv2d(in_channels=256, out_channels=45, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[1]/9207
        self.module_148 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9209
        self.module_149 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9216
        self.module_150 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9220
        self.module_151 = py_nndct.nn.Module('reshape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9226
        self.module_152 = py_nndct.nn.Module('permute') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9233
        self.module_153 = py_nndct.nn.Module('contiguous') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9235
        self.module_154 = py_nndct.nn.Conv2d(in_channels=512, out_channels=45, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[2]/9254
        self.module_155 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9256
        self.module_156 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9263
        self.module_157 = py_nndct.nn.Module('shape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9267
        self.module_158 = py_nndct.nn.Module('reshape') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9273
        self.module_159 = py_nndct.nn.Module('permute') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9280
        self.module_160 = py_nndct.nn.Module('contiguous') #DetectMultiBackend::DetectMultiBackend/Model[model]/Detect[model]/Detect[24]/9282

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_7 = self.module_7(output_module_5)
        output_module_7 = self.module_8(output_module_7)
        output_module_7 = self.module_9(output_module_7)
        output_module_7 = self.module_10(output_module_7)
        output_module_11 = self.module_11(input=output_module_5, other=output_module_7, alpha=1)
        output_module_12 = self.module_12(output_module_0)
        output_module_12 = self.module_13(output_module_12)
        output_module_11 = self.module_14(dim=1, tensors=[output_module_11,output_module_12])
        output_module_11 = self.module_15(output_module_11)
        output_module_11 = self.module_16(output_module_11)
        output_module_11 = self.module_17(output_module_11)
        output_module_11 = self.module_18(output_module_11)
        output_module_19 = self.module_19(output_module_11)
        output_module_19 = self.module_20(output_module_19)
        output_module_21 = self.module_21(output_module_19)
        output_module_21 = self.module_22(output_module_21)
        output_module_21 = self.module_23(output_module_21)
        output_module_21 = self.module_24(output_module_21)
        output_module_25 = self.module_25(input=output_module_19, other=output_module_21, alpha=1)
        output_module_26 = self.module_26(output_module_25)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_30 = self.module_30(input=output_module_25, other=output_module_26, alpha=1)
        output_module_31 = self.module_31(output_module_11)
        output_module_31 = self.module_32(output_module_31)
        output_module_30 = self.module_33(dim=1, tensors=[output_module_30,output_module_31])
        output_module_30 = self.module_34(output_module_30)
        output_module_30 = self.module_35(output_module_30)
        output_module_36 = self.module_36(output_module_30)
        output_module_36 = self.module_37(output_module_36)
        output_module_38 = self.module_38(output_module_36)
        output_module_38 = self.module_39(output_module_38)
        output_module_40 = self.module_40(output_module_38)
        output_module_40 = self.module_41(output_module_40)
        output_module_40 = self.module_42(output_module_40)
        output_module_40 = self.module_43(output_module_40)
        output_module_44 = self.module_44(input=output_module_38, other=output_module_40, alpha=1)
        output_module_45 = self.module_45(output_module_44)
        output_module_45 = self.module_46(output_module_45)
        output_module_45 = self.module_47(output_module_45)
        output_module_45 = self.module_48(output_module_45)
        output_module_49 = self.module_49(input=output_module_44, other=output_module_45, alpha=1)
        output_module_50 = self.module_50(output_module_49)
        output_module_50 = self.module_51(output_module_50)
        output_module_50 = self.module_52(output_module_50)
        output_module_50 = self.module_53(output_module_50)
        output_module_54 = self.module_54(input=output_module_49, other=output_module_50, alpha=1)
        output_module_55 = self.module_55(output_module_36)
        output_module_55 = self.module_56(output_module_55)
        output_module_54 = self.module_57(dim=1, tensors=[output_module_54,output_module_55])
        output_module_54 = self.module_58(output_module_54)
        output_module_54 = self.module_59(output_module_54)
        output_module_60 = self.module_60(output_module_54)
        output_module_60 = self.module_61(output_module_60)
        output_module_62 = self.module_62(output_module_60)
        output_module_62 = self.module_63(output_module_62)
        output_module_64 = self.module_64(output_module_62)
        output_module_64 = self.module_65(output_module_64)
        output_module_64 = self.module_66(output_module_64)
        output_module_64 = self.module_67(output_module_64)
        output_module_68 = self.module_68(input=output_module_62, other=output_module_64, alpha=1)
        output_module_69 = self.module_69(output_module_60)
        output_module_69 = self.module_70(output_module_69)
        output_module_68 = self.module_71(dim=1, tensors=[output_module_68,output_module_69])
        output_module_68 = self.module_72(output_module_68)
        output_module_68 = self.module_73(output_module_68)
        output_module_68 = self.module_74(output_module_68)
        output_module_68 = self.module_75(output_module_68)
        output_module_76 = self.module_76(output_module_68)
        output_module_77 = self.module_77(output_module_76)
        output_module_78 = self.module_78(output_module_77)
        output_module_79 = self.module_79(dim=1, tensors=[output_module_68,output_module_76,output_module_77,output_module_78])
        output_module_79 = self.module_80(output_module_79)
        output_module_79 = self.module_81(output_module_79)
        output_module_79 = self.module_82(output_module_79)
        output_module_79 = self.module_83(output_module_79)
        output_module_84 = self.module_84(input=output_module_79, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_84 = self.module_85(dim=1, tensors=[output_module_84,output_module_54])
        output_module_86 = self.module_86(output_module_84)
        output_module_86 = self.module_87(output_module_86)
        output_module_86 = self.module_88(output_module_86)
        output_module_86 = self.module_89(output_module_86)
        output_module_86 = self.module_90(output_module_86)
        output_module_86 = self.module_91(output_module_86)
        output_module_92 = self.module_92(output_module_84)
        output_module_92 = self.module_93(output_module_92)
        output_module_86 = self.module_94(dim=1, tensors=[output_module_86,output_module_92])
        output_module_86 = self.module_95(output_module_86)
        output_module_86 = self.module_96(output_module_86)
        output_module_86 = self.module_97(output_module_86)
        output_module_86 = self.module_98(output_module_86)
        output_module_99 = self.module_99(input=output_module_86, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_99 = self.module_100(dim=1, tensors=[output_module_99,output_module_30])
        output_module_101 = self.module_101(output_module_99)
        output_module_101 = self.module_102(output_module_101)
        output_module_101 = self.module_103(output_module_101)
        output_module_101 = self.module_104(output_module_101)
        output_module_101 = self.module_105(output_module_101)
        output_module_101 = self.module_106(output_module_101)
        output_module_107 = self.module_107(output_module_99)
        output_module_107 = self.module_108(output_module_107)
        output_module_101 = self.module_109(dim=1, tensors=[output_module_101,output_module_107])
        output_module_101 = self.module_110(output_module_101)
        output_module_101 = self.module_111(output_module_101)
        output_module_112 = self.module_112(output_module_101)
        output_module_112 = self.module_113(output_module_112)
        output_module_112 = self.module_114(dim=1, tensors=[output_module_112,output_module_86])
        output_module_115 = self.module_115(output_module_112)
        output_module_115 = self.module_116(output_module_115)
        output_module_115 = self.module_117(output_module_115)
        output_module_115 = self.module_118(output_module_115)
        output_module_115 = self.module_119(output_module_115)
        output_module_115 = self.module_120(output_module_115)
        output_module_121 = self.module_121(output_module_112)
        output_module_121 = self.module_122(output_module_121)
        output_module_115 = self.module_123(dim=1, tensors=[output_module_115,output_module_121])
        output_module_115 = self.module_124(output_module_115)
        output_module_115 = self.module_125(output_module_115)
        output_module_126 = self.module_126(output_module_115)
        output_module_126 = self.module_127(output_module_126)
        output_module_126 = self.module_128(dim=1, tensors=[output_module_126,output_module_79])
        output_module_129 = self.module_129(output_module_126)
        output_module_129 = self.module_130(output_module_129)
        output_module_129 = self.module_131(output_module_129)
        output_module_129 = self.module_132(output_module_129)
        output_module_129 = self.module_133(output_module_129)
        output_module_129 = self.module_134(output_module_129)
        output_module_135 = self.module_135(output_module_126)
        output_module_135 = self.module_136(output_module_135)
        output_module_129 = self.module_137(dim=1, tensors=[output_module_129,output_module_135])
        output_module_129 = self.module_138(output_module_129)
        output_module_129 = self.module_139(output_module_129)
        output_module_140 = self.module_140(output_module_101)
        output_module_141 = self.module_141(input=output_module_140, dim=0)
        output_module_142 = self.module_142(input=output_module_140, dim=2)
        output_module_143 = self.module_143(input=output_module_140, dim=3)
        output_module_144 = self.module_144(input=output_module_140, shape=[output_module_141,3,15,output_module_142,output_module_143])
        output_module_144 = self.module_145(dims=[0,1,3,4,2], input=output_module_144)
        output_module_144 = self.module_146(output_module_144)
        output_module_147 = self.module_147(output_module_115)
        output_module_148 = self.module_148(input=output_module_147, dim=0)
        output_module_149 = self.module_149(input=output_module_147, dim=2)
        output_module_150 = self.module_150(input=output_module_147, dim=3)
        output_module_151 = self.module_151(input=output_module_147, shape=[output_module_148,3,15,output_module_149,output_module_150])
        output_module_151 = self.module_152(dims=[0,1,3,4,2], input=output_module_151)
        output_module_151 = self.module_153(output_module_151)
        output_module_129 = self.module_154(output_module_129)
        output_module_155 = self.module_155(input=output_module_129, dim=0)
        output_module_156 = self.module_156(input=output_module_129, dim=2)
        output_module_157 = self.module_157(input=output_module_129, dim=3)
        output_module_158 = self.module_158(input=output_module_129, shape=[output_module_155,3,15,output_module_156,output_module_157])
        output_module_158 = self.module_159(dims=[0,1,3,4,2], input=output_module_158)
        output_module_158 = self.module_160(output_module_158)
        return output_module_144,output_module_151,output_module_158
