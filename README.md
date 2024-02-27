# Vitis-ai-zcu104-yolov5

<!-- ABOUT THE PROJECT -->
## About The Project
Use Vitis-AI to deploy yolov5 on ZCU104

## ����vai_q_pytorch

### 1��Prepare 3 files��

| name          | description                      |
| ------------- | -------------------------------- |
| model.pt      | pre-training model ,  PTH file   |
| model.py      | Python script                    |
| dataset       | 100 - 1000 images                |


### 2��modify model
Ҫʹ PyTorch ģ�Ϳ���������Ҫ�޸�ģ�Ͷ��壬��ȷ���޸ĺ��ģ��������������:
- Ҫ������ģ��Ӧ����ǰ����������������������Ӧ�Ƴ�����Ǩ�����������ࡣ��Щ����ͨ����ΪԤ����ͺ���������������������Ƴ�����ô�� API ��������ģ���н����Ƴ�����������ǰ������ģ��ʱ�����쳣��Ϊ�� 
- ��������ģ��Ӧ��ͨ�� jit ׷�ٲ��ԡ�������ģ������Ϊ����״̬��Ȼ��ʹ�� torch.jit.trace ���������Ը���ģ�͡�
����ʹ�õ�yolov5ģ�ͣ���Ҫ��������ȡ��ǰ�������ȶ��ڲ���ȥ�����޸ĺ�������£�
```
        z = []
        for i in range(nl):
            bs, _, ny, nx, _no= x[i].shape
            # x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                    grid[i], anchor_grid[i] = _make_grid(anchors,stride,nx, ny, i)
```

### 3�� vai_q_pytorch API ���������ű�
- ���������ǰ�����о���ѵ���ĸ���ģ�ͺ� Python �ű����ڶ�ģ�;���/mAP ������ֵ����ô������ API �Ὣ����ģ���滻Ϊ����ģ�顣������ֵ������������ģ��ǰ������� quant_mode ��־��Ϊ��calib��������У׼������ֵ�� �����ж��������������衣У׼�󣬽� quant_mode ����Ϊ��test���Զ�����ģ�ͽ�����ֵ��
 - ���� vai_q_pytorch ģ��
 ```
    from?pytorch_nndct.apis?import?torch_quantizer,?dump_xmodel
 ```

 - ����Ҫ���������ģ��������������������ȡת�����ģ�͡�
 ```
   input?=?torch.randn([batch_size,?3,?224,?224])
   quantizer?=?torch_quantizer(quant_mode,?model,?(input))
   quant_model?=?quantizer.quant_model
 ```

 - ʹ��ת�����ģ��ǰ�������硣

### 4��������������ȡ���
��������һ��������ļ�Ŀ¼�ṹ������quantize.py��Ϊ�����ű������ݼ���label�����datasets�У����н����������runsĿ¼�¡�
![Alt text](image.png)

- ���к���--quant_mode calib��������������ģ�͡�
```
python?resnet18_quant.py?--quant_mode?calib?--subset_len
```
![Alt text](image-1.png)

 - ��ʱ������ʼ�����ҳɹ��������ݼ���ģ�͡����������󣬿�ʼ�����ݼ��е�ģ�ͽ���Ԥ�⡣
![Alt text](image-2.png)

 - ���Կ�����һ��calib������ɣ�����ȥruns�ļ����в鿴���н����
![Alt text](image-3.png)

 - ������������ģ�Ͷ����ݼ���Ԥ�⣬����������ģ�͵ĸ���ָ�������F1��P��R��PR����ͼΪ���ɵ�F1�������ߡ�
![Alt text](image-4.png)

- Ҫ���� xmodel ���б��루�Լ� onnx ��ʽ����ģ�ͣ������δ�СӦΪ 1������ subset_len=1 �ɱ�������������������������
```
python?resnet18_quant.py?--quant_mode?test?--subset_len?1?--batch_size=1?--deploy
```
![Alt text](image-5.png)

 - ���Կ�����ʼ��ȷ���У�������������ɽ����ͬ�������Ҳ��������runs�ļ����¡�
 ![Alt text](image-6.png)


### 5�� ���ɿ�ִ���ļ�Xmodel
- ������������ɣ���Ҫע�����Ҫѡ���Լ��忨����Ӧ��DPU�ͺš�
```
vai_c_xir?-x?./quantize_result/DetectMultiBackend_int.xmodel?-a?/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json?-o?./?-n?model
```
���н��������Կ�����������������Ҫ��Xmodel��ִ���ļ���
![Alt text](image-7.png)
