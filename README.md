## 综述

- 本模型基于pytorch深度学习框架、mmdetection开源工具搭建而成。
- 最终模型位于submit文件夹下

### 关于代码

- 基于mmdetection开源框架，使用Cascade-rcnn+fpn为模型主体，采用resnet-50提取特征。

- 在mmdetection官方配置文件**cascade_rcnn_r50_fpn_1x.py**基础上做如下改进：

  - 在cascade-rcnn提取特征的backbone最后一个阶段加入可变形卷积，提高模型对 scale, aspect ratio and rotation 等映射的泛化能力。

    ```python
    # 11~12 行：
    dcn=dict(modulated=False, deformable_groups=1, fallback_on_stride=False), stage_with_dcn=(False, False, False, True))
    ```

    

  - 根据样本瑕疵尺寸分布，在RPN中修改anchor的长宽比，增强模型对不同大小瑕点的鲁棒能力

    ```python
     # 24 行：
     anchor_ratios=[0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 50.0],  # 
    ```
  
  - 在rcnn的采样阶段将默认使用的随机采样RandomSampler，替换成OHEM采样，即每个级联层引入在线难样本学习。
  
    ```python
    # 110,125,140 行：
    type='OHEMSampler'
    ```
    
  - 使用全尺寸图片训练，不缩放，对小目标使网络捕捉更多的像素
    
    ```python
     # 169 行
     dict(type='Resize', img_scale=(2446, 1000), keep_ratio=True), #
    ```
    
  - 采用NVIDIA 1080Ti，每张gpu训练两张图片
    
  - ```python
    # 192 行
    imgs_per_gpu=2
    ```
  
  - 学习率的设置：lr = 0.00125xbatch_size，batch_size = gpu_num（2）x imgs_per_gpu
  
  - ```python
    - # 210 行：
    - optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) 
    ```

### 代码及数据组织方式
├── code (本组代码贡献)                                                   
│   ├── cascade_transform_skypool.py                                     
│   ├── creat_resule.py                                                 
│   ├── Fabric2COCO.py                                                   
├── data (数据集防止方式)                                                   
│   ├── guangdong1_round1_train1_20190818                               
│   │   ├──*.jpg  
│   ├── guangdong1_round1_train2_20190828  
│   │   ├──*.jpg  
│   ├── guangdong1_round1_testA_20190818  
│   │   ├──*.jpg  
│   └── **skypool_first_trainval**（数据处理后生成的coco格式训练集）  
├── data_process.sh （数据处理脚本）  
├── setup.sh （安装脚本）  
└── submit （训练后模型和日志文件输出文件夹）       

### 快速部署指南
- 依赖：
  - python版本：3.7 及以上
  - pytorch版本：1.0 及以上
  - CUDA版本：10.0 及以上
  - Anaconda3
  
- 本代码可以使用shell脚本快速部署，但是需要在执行shell脚本前安装**Anaconda3**

- Anaconda3 包管理器就绪后即可按以下流程对代码部署
  1. 添加脚本执行权限：```chmod u+x setup.sh```
  2. 编辑安装脚本 ```vim setup.sh```，修改21行CUDA_HOME，指定CUDA的位置
  3. 执行安装脚本：```source setup.sh```
  4. 若第三步使用```./setup.sh```则需要激活环境：```source activate skypool_lmc```
  
- 代码环境部署完成后，需要对原始数据进行处理，转换为COCO数据集格式，原始数据按上述方式准备
  1. 添加脚本执行权限：```chmod u+x data_process.sh```
  2. 执行数据处理：```./data_process.sh```
  
- 预训练模型：
  
  - 下载COCO数据集预训练模型，并放入```mmdetection/checkpoints```下，预训练模型下载地址：
    
    - ```html
      https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth
      ```
    
    - 模型转换：需要将COCO数据集模型转换为当前数据集可用的类型，执行命令：
    
      ```shell
      python code/model_transform.py
      ```
  
- 模型训练：
  
  - 单卡训练：
  
    ```shell
    python mmdetection/tools/train.py code/cascade_transform_skypool.py --gpus 1 --validate
    ```
  
  - 多卡分布训练：
  
    ```shell
    # 双卡训练
    bash mmdetection/tools/dist_train.sh code/cascade_transform_skypool.py 2
    ```
  
- 模型测试：

  ```shell
  python mmdetection/tools/test.py code/cascade_transform_skypool.py submit/训练生成的权重 --out results.pkl --eval bbox
  ```
  
- 结果分析：

  ```shell
  python mmdetection/tools/analyze_logs.py plot_curve submit/train/xxx.log.json --keys loss --legend loss
  ```

  

- 生成提交结果：
  
  ```shell
  python code/creat_resule.py --wight-path=submit/train/epoch_62.pth
  ```
### 模型和数据集
 - 数据集可以通过url.txt文件中的连接自行下载
 - 模型请从[这里下载](https://pan.baidu.com/s/1CH5AnxAmRFFygvmm9ysy0g)

