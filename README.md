# Trisomy21Net
### End-to-end Predictions
This project is a tool to construct Trisomy21Net models, written in Keras and tensorflow.
Trisomy21Net is a CNN algorithm that can screen fetuses with trisomy 21 from normal cases based on ultrasound images. 
Our architecture is consisted of 11 layers.
## Acknowledgement
I would like to thank Pranav Rajpurkar (Stanford ML group) and Xinyu Weng (北京大學) for sharing their experiences on this task. Also I would like to thank Felix Yu for providing DenseNet-Keras source code.
## Authors
Liwen Zhang, Di Dong, Yongqing Sun, Chaoen Hu, Xin Yang, Qingqing Wu, Jie Tian.
##Environment
We train our model in ubuntu with 4 Titan xp. If you use >= CUDA 9, make sure you set tensorflow_gpu >= 1.10.0.
## Experiments
Model comparison and visualization is based on Gad-CAM, more details please see the paper : https://arxiv.org/pdf/1610.02391v1.pdf.
Our module of callback function refers to https://arxiv.org/pdf/1711.05225.pdf.
#Setting
For training, we set batch size of 48, initial learning rate of 0.0001,epoch of 300.
The trained weight is 'titansun09160837-300-0.941-0.956.h5'.
