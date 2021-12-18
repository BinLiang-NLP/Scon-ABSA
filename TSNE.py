#!/usr/bin/env python
# coding: utf-8

# # Bert

# In[67]:


from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import random


# In[74]:


def Tsne_graph(fname,index):
    
    f = open(fname)
    lines = f.readlines()
    label=[]
    embedding=[]
    
    print("加载数据结束")
    
    for line in lines:
        data = line.split("\t")

        label.append(data[0])
        data[1] = eval(data[1].replace("\n",""))
        embedding.append(data[1])
    
    
    print("数据格式化结束")
    
    tsne = TSNE()
    tsne.fit_transform(embedding) #进行数据降维
    tsne = pd.DataFrame(tsne.embedding_) #转换数据格式
    print("数据降维")
    
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    
    #不同类别用不同颜色和样式绘图
    print("绘图")
    
    import collections
    hashmap = collections.defaultdict(int)
    for i in range(len(label)):
        hashmap[label[i]]+=1
    print(hashmap)
    
    n1=0
    n2=0
    n3=0
    
    for i in range(len(label)):
        d = tsne.iloc[i]

        if label[i]=='0' and n1<200:
                plt.scatter(d[0]*1/3, d[1]*1/3,c= 'r',marker='.') #可调整坐标范围
                n1+=1
        elif label[i]=='1' and n2<200:
            
            plt.scatter(d[0]*1/3, d[1]*1/3,c= 'g',marker='.')
            n2+=1
        elif label[i]=='2' and n3<240:
            
            plt.scatter(d[0]*1/3, d[1]*1/3,c= 'b',marker='.')
            n3+=1
    
    print("绘图结束")
    
    
    if "normal" in fname:
        plt.savefig('bert_normal_'+str(index)+'.png',dpi=600)
    elif "multi" in fname:
        plt.savefig('bert_multi_'+str(index)+'.png',dpi=600)
    elif "cl_6" in fname:
        plt.savefig('bert_cl_6_'+str(index)+'.png',dpi=600)
    elif "cl_2X3" in fname:
        plt.savefig('bert_cl_2X3_'+str(index)+'.png',dpi=600)
    elif "cl" in fname:
        plt.savefig('bert_cl_'+str(index)+'.png',dpi=600)
    
    
    plt.show()


# # bert_normal

# In[71]:


for i in range(10):
    Tsne_graph("../../ABSA_CL/save_result_text/bert_spc_cl_normal.txt",i)


# # bert_multi

# In[73]:


for i in range(10):
    Tsne_graph("../../ABSA_CL/save_result_text/bert_spc_cl_multi.txt",i)


# In[75]:


for i in range(10):
    Tsne_graph("../../ABSA_CL/save_result_text/bert_spc_cl_cl.txt",i)


# # bert_cl_6

# In[76]:


for i in range(10):
    Tsne_graph("./Bert/bert_spc_cl_cl_6.txt",i)


# # bert_cl_2X3

# In[77]:


for i in range(10):
    Tsne_graph("./Bert/bert_spc_cl_cl_2X3.txt",i)


# In[ ]:




