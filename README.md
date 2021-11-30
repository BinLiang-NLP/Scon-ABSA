# Enhancing Aspect-Based Sentiment Analysis with Supervised Contrastive Learning.

This repo contains the PyTorch code and implementation for the paper *Enhancing Aspect-Based Sentiment Analysis with Supervised Contrastive Learning*.

[**Enhancing Aspect-Based Sentiment Analysis with Supervised Contrastive Learning**](https://dl.acm.org/doi/pdf/10.1145/3459637.3482096) 
<br>
Bin Liang<sup>#</sup>, Wangda Luo<sup>#</sup>, Xiang Li, Lin Gui, Min Yang, Xiaoqi Yu, and Ruifeng Xu<sup>*</sup>. *Proceedings of CIKM 2020*
  
Please cite our paper and kindly give a star for this repository if you use this code. 

For any question, plaese email luowangda_hitsz@163.com or bin.liang@stu.hit.edu.cn.

### Model Overview
![model](./img/model_overview.png)

### Requirement

* pytorch >= 0.4.0
* numpy >= 1.13.3
* sklearn
* python 3.6 / 3.7
* CUDA 9.0
* [transformers](https://github.com/huggingface/transformers)

To install requirements, run `pip install -r requirements.txt`.  

### Dataset

you can directly use the processed dataset located in `datasets/`:  
Note that you need to extract the data from the datasets folder: `unzip datasets.zip`
```
├── data
│   │   ├── semeval14(res14，laptop14)
│   │   ├── semeval15(res15)
│   │   ├── semeval16(res16)
│   │   ├── MAMS
```

The dataSet contains with cl_2X3 is the dataSet obtained after label argment, and each data is as follows:  
Context  
Aspect  
Aspect-sentiment-label(-1:negative;0:netrual;1:positive)  
Contrastive-label(aspect-dependent/aspect-invariant)  
Contrastive-aspect-label(0:negative;1:netrual;2:positive)  

### Preparation
a) Download the pytorch version pre-trained bert-base-uncased model and vocabulary from the link provided by huggingface. Then change the value of parameter --bert_model_dir to the directory of the bert model.
you can get the pre-trained bert-base-uncased model in https://github.com/huggingface/transformers.

b) Label enhancement method. For new data, additional supervised signals need to be obtained through label enhancement;  
&nbsp;&nbsp;&nbsp;&nbsp;i) Through BERT overfitting the training set, the acc can reach more than 97%;  
&nbsp;&nbsp;&nbsp;&nbsp;ii) Replace aspect with other or mask, and get the emotional label of the aspect after replacing the aspect;  
&nbsp;&nbsp;&nbsp;&nbsp;iii) Determine whether the output label is consistent with the real label, and fill in the aspect-dependent/aspect-invariant label for the data.  

c) The data defaults are in data_utils.py, which you can view if you want to change the data entered into the model.

### Training

1. Adjust the parameters and set the experiment.  
    --model:Selection model.(bert_spc_cl)  
    --dataset:Select dataSet.(acl14,res14,laptop14,res15,res16,mams and so on)  
    --num_epoch：Iterations of the model.  
    --is_test 0:Verify module.(1 is data verification, 0 is model training)  
    --type: Select a task type.(normal,cl2,cl6,cl2X3)  
2. Run the shell script to start the program.

```bash
./run.sh
```
For run.sh code:
```angular2

CUDA_VISIBLE_DEVICES=3 \
  python train_cl.py \
  --model_name bert_spc_cl \
  --dataset cl_mams_2X3 \
  --num_epoch 50 \
  --is_test 0 \
  --type cl2X3

```
For dataset implementation, you can choose these datasets: "cl_acl2014_2X3" "cl_res2014_2X3" "cl_laptop2014_2X3" "cl_res2015_2X3" "cl_res2016_2X3" "cl_mams_2X3".
### Testing
```bash
./run_test.sh
```


### Citation
```
@inproceedings{10.1145/3459637.3482096,
author = {Liang, Bin and Luo, Wangda and Li, Xiang and Gui, Lin and Yang, Min and Yu, Xiaoqi and Xu, Ruifeng},
title = {Enhancing Aspect-Based Sentiment Analysis with Supervised Contrastive Learning},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482096},
doi = {10.1145/3459637.3482096},
abstract = {Most existing aspect-based sentiment analysis (ABSA) research efforts are devoted to extracting the aspect-dependent sentiment features from the sentence towards the given aspect. However, it is observed that about 60% of the testing aspects in commonly used public datasets are unknown to the training set. That is, some sentiment features carry the same polarity regardless of the aspects they are associated with (aspect-invariant sentiment), which props up the high accuracy of existing ABSA models when inevitably inferring sentiment polarities for those unknown testing aspects. Therefore, in this paper, we revisit ABSA from a novel perspective by deploying a novel supervised contrastive learning framework to leverage the correlation and difference among different sentiment polarities and between different sentiment patterns (aspect-invariant/-dependent). This allows improving the sentiment prediction for (unknown) testing aspects in the light of distinguishing the roles of valuable sentiment features. Experimental results on 5 benchmark datasets show that our proposed approach substantially outperforms state-of-the-art baselines in ABSA. We further extend existing neural network-based ABSA models with our proposed framework and achieve improved performance.},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {3242–3247},
numpages = {6},
keywords = {sentiment analysis, contrastive learning, aspect sentiment analysis},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}
```
or

```
@inproceedings{liang2021enhancing,
  title={Enhancing Aspect-Based Sentiment Analysis with Supervised Contrastive Learning},
  author={Liang, Bin and Luo, Wangda and Li, Xiang and Gui, Lin and Yang, Min and Yu, Xiaoqi and Xu, Ruifeng},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3242--3247},
  year={2021}
}
```

### Credits
* The code of this repository partly relies on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).
* We would like to express my gratitude to the authors of the [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) repository.
* The Supervised Contrastive Loss devised in this paper is partly inspired by [Supervised Contrastive Learning (Khosla and Tian, et al.)](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf), we would like to express my gratitude to the authors of this paper.
