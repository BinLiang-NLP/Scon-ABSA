# -*- coding: utf-8 -*-
# file: train_cl.py
# author: bin Liang and wangda Luo

import logging
import argparse
import math
import os
import sys
import random
import numpy

from losses import SupConLoss
from infoNCE import InfoNCE
from criterion import CL_auxiliary, CL_sentiment

from sklearn import metrics
from time import strftime, localtime

from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from models.bert_spc_cl import BERT_SPC_CL

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            print("Loading BERT")
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            print("End for loading BERT")

        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)


        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        print("Train size:",len(self.trainset))
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        print("Test size:",len(self.testset))


        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, contrastiveLoss_auxiliary, contrastiveLoss_sentiment, criterion2, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_acc_f1=0

        max_val_f1 = 0
        max_val_f1_acc=0


        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]

                outputs = self.model(inputs)

                # for multi-task classification and use it in ablation experiments
                if self.opt.type=="muti":

                    targets = batch['polarity'].to(self.opt.device)
                    cllabel = batch['cllabel'].to(self.opt.device)
                    loss2 = criterion2(outputs[2], cllabel)
                    loss1 = criterion(outputs[0], targets)

                    loss = loss1 + loss2

                # for our model
                elif "_cl_2X3" in self.opt.dataset_file['train']:

                    targets = batch['polarity'].to(self.opt.device)
                    cllabel = batch['cllabel'].to(self.opt.device)
                    polabel = batch['polabel'].to(self.opt.device)

                    loss2 = contrastiveLoss_auxiliary(outputs[1], cllabel)
                    loss1 = criterion(outputs[0], targets)
                    #loss3 = contrastiveLoss_sentiment(outputs[1],cllabel, polabel)
                    loss3 = contrastiveLoss_sentiment(outputs[1], polabel)
                    # logger.info('loss: {:.4f}, loss1: {:.4f},loss2: {:.4f},loss3: {:.4f}'.format(train_loss,loss1,loss2,loss3))
                    loss = loss1 + loss2 + 0 * loss3

                # for contrastive learning for 6 label (2*3) , sentiment(0,1,2)* contrast-label(1:aspect-depencent,0:aspect-invariant)
                elif "_cl_6" in self.opt.dataset_file['train'] or "_cl" in self.opt.dataset_file['train']:

                    targets = batch['polarity'].to(self.opt.device)
                    cllabel = batch['cllabel'].to(self.opt.device)

                    loss2 = contrastiveLoss_auxiliary(outputs[1], cllabel)
                    loss1 = criterion(outputs[0], targets)

                    loss = loss1 + loss2
                    # logger.info('loss: {:.4f}, loss1: {:.4f},loss2: {:.4f}'.format(loss,loss1,loss2))

                # normal for aspect-sentiment classification
                else:
                    targets = batch['polarity'].to(self.opt.device)
                    loss1 = criterion(outputs[0], targets)
                    loss=loss1
                    # logger.info('loss: {:.4f}, acc: {:.4f}'.format(loss, loss1))


                loss.backward()
                optimizer.step()
                # loss=0

                n_correct += (torch.argmax(outputs[0], -1) == targets).sum().item()

                n_total += len(outputs[0])
                loss_total += loss.item() * len(outputs[0])
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total

                    if "_cl_2X3" in self.opt.dataset_file['train']:
                        logger.info('loss: {:.4f}, acc: {:.4f},loss1: {:.4f},loss2: {:.4f},loss3: {:.4f}'.format(train_loss, train_acc,loss1,loss2,loss3))
                    elif "_cl_6" in self.opt.dataset_file['train'] or "_cl" in self.opt.dataset_file['train']:
                        logger.info('loss: {:.4f}, acc: {:.4f},loss1: {:.4f},loss2: {:.4f}'.format(train_loss, train_acc,loss1,loss2))
                    else:
                        logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))


            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_acc_f1 = val_f1

                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')

                path = 'state_dict/{0}_{1}_val_type_{2}_acc_{3}'.format(self.opt.model_name, self.opt.dataset,self.opt.type, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_f1_acc = val_acc

            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path,max_val_acc,max_val_acc_f1,max_val_f1,max_val_f1_acc

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)[0]

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1


    def _evaluate_acc_f1_Test(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()

        List_label = []


        index=0

        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs[0], -1) == t_targets).sum().item()
                n_total += len(t_outputs[0])

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs[0]
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs[0]), dim=0)

                for i in range(len(t_outputs[0])):

                    label = torch.argmax(t_outputs[0][i], -1).cpu().data.numpy()

                    hiddenState = t_outputs[1][i].cpu().data.numpy().tolist()

                    Str = str(label) + "\t" + str(hiddenState) + "\n"

                    Str = str(label) + "\t" + str(hiddenState) + "\n"

                    List_label.append(Str)



        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')


        # you need use instruction: mkdir save_result_text to build a folder.
        # Output aspect-sentiment for each data
        #---------------------------------------------------
        fname = str(self.opt.model_name + "_" + str(self.opt.type)) + "_" + str(round(acc, 4)) + "_" + str(
            round(f1, 4)) + ".txt"
        f = open("./save_result_text/" + fname, "w")

        for i in List_label:
            f.write(i)
        f.close()
        #----------------------------------------------------


        return acc, f1


    def run_Test(self):
        # Loss and Optimizer

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self.model.load_state_dict(torch.load('./state_dict/'+self.opt.testfname))


        test_acc, test_f1 = self._evaluate_acc_f1_Test(test_data_loader)

        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    def run(self):
        # Loss and Optimizer
        # contrastiveLoss = InfoNCE()

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        contrastiveLoss_auxiliary = CL_auxiliary(self.opt)
        contrastiveLoss_sentiment = CL_sentiment(self.opt)
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path,max_val_acc,max_val_acc_f1,max_val_f1,max_val_f1_acc= self._train(criterion, contrastiveLoss_auxiliary, contrastiveLoss_sentiment, criterion2,optimizer, train_data_loader, val_data_loader)

        max_val_acc = round(max_val_acc, 4)
        max_val_acc_f1 = round(max_val_acc_f1, 4)
        max_val_f1_acc = round(max_val_f1_acc, 4)
        max_val_f1 = round(max_val_f1, 4)

        # output result
        print(self.opt.model_name, self.opt.dataset_file['train'], "Test accMAX：", max_val_acc, max_val_acc_f1)
        print(self.opt.model_name, self.opt.dataset_file['train'], "Test f1MAX：", max_val_f1_acc, max_val_f1)
        print(str(self.opt.seed) + "_" + str(self.opt.lr))

        # save result
        #------------------------------------------
        if not os.path.exists('save_result'):
            os.mkdir('save_result')
        fname = str(self.opt.dataset_file['test'].split("/")[-2] + "_" + self.opt.dataset_file['test'].split("/")[
            -1]) + "_result_new" + str(self.opt.model_name) +"_"+str(self.opt.type)+ ".txt"
        f = open("./save_result/" + fname, "a+")

        f.write(" Test： acc_MAX:" + str(max_val_acc) + " f1_MAX:" + str(max_val_f1) + " " + str(
            self.opt.model_name) + "_" + str(self.opt.dataset_file['train']) + "_seed" + str(self.opt.seed) +"_"+ str(
            self.opt.lr) + "\n")
        f.close()
        #------------------------------------------

        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='cl_acl2014_2X3,cl_res2014_2X3,cl_laptop2014_2X3,cl_res2015_2X3,cl_res2016_2X3,cl_mams_2X3')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--is_test', default=0, type=int)
    parser.add_argument('--type', default="normal", type=str)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--testfname', default=0, type=str)
    parser.add_argument('--temperatureP', default=0.07, type=float)
    parser.add_argument('--temperatureY', default=0.14, type=float)
    parser.add_argument('--alpha', default=1, type=float,required=False)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_spc': BERT_SPC,
        'bert_spc_cl': BERT_SPC_CL,
        # default hyper-parameters for contrative-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 20
        #---------------------------------
        # 'lstm': LSTM,
        # 'atae_lstm': ATAE_LSTM,
        # 'ian': IAN,
        # 'ram': RAM,
        # 'aoa': AOA,
        # 'mgan': MGAN,
        # 'asgcn': ASGCN,

    }
    dataset_files = {
        'acl2014': {
            'train': './datasets/acl-14-short-data/train.raw',
            # 'test': './datasets/acl-14-short-data/train.raw'
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'res2014': {
            'train': './datasets/semeval14/restaurant_train.raw',
            # 'test': './datasets/semeval14/restaurant_train.raw',
            # 'test': './datasets/semeval14/mask_restaurant_train.raw'
            'test': './datasets/semeval14/restaurant_test.raw',
            # 'test': './datasets/semeval14/restaurant_train.raw',

        },
        'laptop': {
            'train': './datasets/semeval14/laptop_train.raw',
            # 'test': './datasets/semeval14/mask_laptop_train.raw',
            # 'test': './datasets/semeval14/laptop_train.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        'res2015': {
            'train': './datasets/semeval15/restaurant_train.raw',
            # 'test': './datasets/semeval15/restaurant_train.raw'
            'test': './datasets/semeval15/restaurant_test.raw'
        },
        'res2016': {
            'train': './datasets/semeval16/restaurant_train.raw',
            # 'test': './datasets/semeval16/restaurant_train.raw',
            # 'test': './datasets/semeval16/mask_restaurant_train.raw',
            'test': './datasets/semeval16/restaurant_test.raw',
        },
        'mams': {
            'train': './datasets/MAMS/mams_train.raw',
            'test': './datasets/MAMS/mams_test.raw'
        },
        "cl_acl2014": {
            'train': './datasets/cl_data/2014acl_cl.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        "cl_res2014":{
            'train': './datasets/cl_data/2014res_cl.raw',
            'test': './datasets/semeval14/restaurant_test.raw'
        },
        "cl_laptop2014":{
            'train': './datasets/cl_data/2014laptop_cl.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        "cl_res2015":{
            'train': './datasets/cl_data/2015res_cl.raw',
            'test': './datasets/semeval15/restaurant_test.raw'
        },
        "cl_res2016":{
            'train': './datasets/cl_data/2016res_cl.raw',
            'test': './datasets/semeval16/restaurant_test.raw'
        },
        "cl_mams":{
            'train': './datasets/cl_data/mams_cl.raw',
            'test': './datasets/MAMS/mams_test.raw'
        },
        "cl_acl2014_6": {
            'train': './datasets/cl_data/2014acl_cl_6.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        "cl_laptop2014_6":{
            'train': './datasets/cl_data/2014laptop_cl_6.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        "cl_res2014_6":{
            'train': './datasets/cl_data/2014res_cl_6.raw',
            'test': './datasets/semeval14/restaurant_test.raw'
        },
        "cl_res2015_6":{
            'train': './datasets/cl_data/2015res_cl_6.raw',
            'test': './datasets/semeval15/restaurant_test.raw'
        },
        "cl_res2016_6":{
            'train': './datasets/cl_data/2016res_cl_6.raw',
            'test': './datasets/semeval16/restaurant_test.raw'
        },
        'cl_mams_6': {
            'train': './datasets/cl_data/mams_cl_6.raw',
            'test': './datasets/MAMS/mams_test.raw'
        },

        #------------
        "cl_acl2014_2X3": {
            'train': './datasets/cl_data_2X3/2014acl_cl_2X3.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        "cl_laptop2014_2X3": {
            'train': './datasets/cl_data_2X3/2014laptop_cl_2X3.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        "cl_res2014_2X3": {
            'train': './datasets/cl_data_2X3/2014res_cl_2X3.raw',
            'test': './datasets/semeval14/restaurant_test.raw'
        },
        "cl_res2015_2X3": {
            'train': './datasets/cl_data_2X3/2015res_cl_2X3.raw',
            'test': './datasets/semeval15/restaurant_test.raw'
        },
        "cl_res2016_2X3": {
            'train': './datasets/cl_data_2X3/2016res_cl_2X3.raw',
            'test': './datasets/semeval16/restaurant_test.raw'
        },
        'cl_mams_2X3': {
            'train': './datasets/cl_data_2X3/mams_cl_2X3.raw',
            'test': './datasets/MAMS/mams_test.raw'
        },
    }
    input_colses = {
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_spc_cl': ['concat_bert_indices', 'concat_segments_indices'],
        # 'lstm': ['text_indices'],
        # 'atae_lstm': ['text_indices', 'aspect_indices'],
        # 'ian': ['text_indices', 'aspect_indices'],
        # 'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        # 'aoa': ['text_indices', 'aspect_indices'],
        # 'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        # 'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.is_test==0:
        if not os.path.exists('log'):
            os.mkdir('log')

        log_file = './log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))

        ins = Instructor(opt)
        ins.run()
    else:
        print("Model Testing-----")
        ins = Instructor(opt)
        ins.run_Test()

if __name__ == '__main__':
    main()
