# Refined Segmentation R-CNN
A Deep Learning method to segment punctate white matter lesions (PWMLs); Brain tumor segmentation.  
**This repository includes:**  
1.T1WI of 10 patients for test(Full dataset is not allowed to be exposed)  
2.Full code for model training and inference   
3.The link of pre-trained weights on google drive   

## Requirements
Python 3.6.3  
Tensorflow-gpu 1.12.0  
CUDA 9.0  
## Getting Started
1.Download the repositories and [weights](https://anonfiles.com/I2xeTaS5m1/model_enhancedrpn_enlargeroi1.3_segnet_crf_pwml_98765_h5).  
2.Choose a mode in the main.py('inference' or 'training').  
3.Change parameters in configs.py according to the comment  in the file.  
Enjoy!

## Acknowledgment
This repo borrows tons of code from  
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)  
## Results
Performance:  

Index|--------Original MRI--------|---------SOTA----------|-----Mask R-CNN------|-----Our Method--------
:--|:--:|:--:|:--:|:--:


&nbsp;8/77&nbsp;|![8_77](./results/8_77.png)
:--|:--:|
&nbsp;79/67&nbsp;|![79_67](./results/79_67.png)  
&nbsp;82/67&nbsp;|![82_67](./results/82_67.png)  
&nbsp;83/48&nbsp;|![83_48](./results/83_48.png)  
&nbsp;83/54&nbsp;|![83_54](./results/83_54.png)
&nbsp;83/80&nbsp;|![83_80](./results/83_80.png)














