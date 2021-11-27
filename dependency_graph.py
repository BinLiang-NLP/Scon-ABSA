# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 

def cl_process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 4):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

def cl2X3_process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 5):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':


    # process("./datasets/semeval14/restaurant_train.raw")
    # process("./datasets/semeval14/restaurant_test.raw")
    #
    # process("./datasets/semeval15/restaurant_train.raw")
    # process("./datasets/semeval15/restaurant_test.raw")
    #
    # process("./datasets/semeval16/restaurant_train.raw")
    # process("./datasets/semeval16/restaurant_test.raw")
    #
    # process("./datasets/MAMS/mams_test.raw")
    # process("./datasets/MAMS/mams_train.raw")
    #
    # process("./datasets/acl-14-short-data/train.raw")
    # process("./datasets/acl-14-short-data/test.raw")
    #
    # process("./datasets/semeval14/laptop_train.raw")
    # process("./datasets/semeval14/laptop_test.raw")

    #-----

    # cl_process("./datasets/cl_data/2014acl_cl.raw")
    # cl_process("./datasets/cl_data/2014acl_cl_6.raw")
    #
    # cl_process("./datasets/cl_data/2014laptop_cl.raw")
    # cl_process("./datasets/cl_data/2014laptop_cl_6.raw")
    #
    # cl_process("./datasets/cl_data/2014res_cl.raw")
    # cl_process("./datasets/cl_data/2014res_cl_6.raw")
    #
    # cl_process("./datasets/cl_data/2015res_cl.raw")
    # cl_process("./datasets/cl_data/2015res_cl_6.raw")
    #
    # cl_process("./datasets/cl_data/2016res_cl.raw")
    # cl_process("./datasets/cl_data/2016res_cl_6.raw")
    #
    # cl_process("./datasets/cl_data/mams_cl.raw")
    # cl_process("./datasets/cl_data/mams_cl_6.raw")

    #----

    # cl2X3_process("./datasets/cl_data_2X3/2014acl_cl_2X3.raw")
    # cl2X3_process("./datasets/cl_data_2X3/2014laptop_cl_2X3.raw")
    # cl2X3_process("./datasets/cl_data_2X3/2014res_cl_2X3.raw")
    # cl2X3_process("./datasets/cl_data_2X3/2015res_cl_2X3.raw")
    # cl2X3_process("./datasets/cl_data_2X3/2016res_cl_2X3.raw")
    # cl2X3_process("./datasets/cl_data_2X3/mams_cl_2X3.raw")

    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_acl14_test.raw")
    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_laptop_test.raw")
    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_rest15_test.raw")
    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_rest16_test.raw")
    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_rest14_test.raw")
    process("./datasets/No_overlap_aspect_data/not_overlap_aspect_mams_test.raw")


    pass