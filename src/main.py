import os
import torch
import time
import json
import numpy as np
import argparse

from probe import mi_bert_ptb, mi_bert_amr, mi_mlps_ptb, mi_mlps_amr
from evaluation_ptb import test_ge_ptb, test_bert_ptb, test_mi_ptb, mi_noise_ptb, test_random_ptb
from evaluation_amr import test_ge_amr, test_bert_amr, test_mi_amr, mi_noise_amr, test_random_amr

def main_func(args):
    # cpu or gpu
    if args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device))
    args.device = device

    '''
    "ptb" = penn tree bank
    "amr" = amr bank
    "_bert" = no data split
    "_graph" = train/dev/test split (default)
    '''

    ## function descriptions
    ## please refer to specific py file for sub-function descriptions
    '''
    mi_bert_ptb: probe BERT layers with PTB dataset (uncontext=True for ELMo0)
    mi_bert_amr: probe BERT layers with AMR dataset (uncontext=True for ELMo0)
    mi_mlps_ptb: probe GloVe embeddings with PTB dataset
    mi_mlps_amr: probe GloVe embeddings with AMR dataset
    test_ge_ptb: test the graph embedding quality with PTB dataset
    test_ge_amr: test the graph embedding quality with AMR dataset
    test_bert_ptb: test the BERT embedding quality (recover original graphs) with PTB dataset
    test_bert_amr: test the BERT embedding quality (recover original graphs) with AMR dataset
    test_mi_ptb: calculate MI upper bound (global) with PTB dataset (different levels of noise)
    test_mi_amr: calculate MI upper bound (global) with AMR dataset (different levels of noise)
    mi_noise_ptb: calculate MI estimate I(X,G) (local) with PTB dataset (different corrupted types of edges)
    mi_noise_amr: calculate MI estimate I(X,G) (local) with AMR dataset (different corrupted types of edges)
    test_random_ptb: calculate classifier accuracy (local) with PTB dataset (different corrupted types of edges)
    test_random_amr: calculate classifier accuracy (local) with AMR dataset (different corrupted types of edges)
    '''
    
    if args.task == 'ptb_bert':
        # mi_noise_ptb(args, pos=True)
        mi_bert_ptb(args)
        # mi_bert_ptb(args, uncontext=True)
        # test_mi_ptb(args)
        # mi_noise_ptb(args)
        # test_ge_ptb(args)
        # test_ge_ptb(args, data_split=False)
        # mi_mlps_ptb(args)
        pass
    elif args.task == 'ptb_graph':
        # test_random_ptb(args)
        # test_random_ptb(args, corrupt=True)
        # test_ge_ptb(args)
        # test_bert_ptb(args)
        # test_bert_ptb(args, model_name='elmo')
        # test_bert_ptb(args, model_name='glove')
        pass
    elif args.task == 'amr_bert':
        # mi_bert_ptb(args, npeet=True)
        # mi_bert_amr(args)
        # mi_bert_amr(args, model='elmo')
        # test_mi_amr(args)
        # mi_noise_amr(args)
        # test_ge_ptb(args)
        # test_ge_amr(args, data_split=False)
        # mi_mlps_amr(args)
        pass
    elif args.task == 'amr_graph':
        # test_random_amr(args)
        # test_random_amr(args, corrupt=True)
        # test_ge_amr(args)
        # test_bert_amr(args)
        # test_bert_amr(args, model_name='elmo')
        # test_bert_amr(args, model_name='glove')
        pass
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bird\'s Eye Probing. Please refer main_func for specific task selection')
    parser.add_argument("--device", type=int, default=0,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--patience", type=int, default=2,
                        help="early stop patience number")
    parser.add_argument("--baselines", type=bool, default=False,
                        help="whether calculate baselines of MI")
    parser.add_argument("--repeat", type=int, default=5,
                        help="number of repeat time of MI calculation")
    parser.add_argument("--classifier-layers-num", type=int, default=5,
                        help="number of layers of binary classifier")
    parser.add_argument("--bert-layers-num", type=int, default=13,
                        help="number of layers of BERT model + 1")
    parser.add_argument("--bert-hidden-num", type=int, default=768,
                        help="number of hidden units of BERT model")
    parser.add_argument("--hidden-num", type=int, default=128,
                        help="number of hidden units of mutual information")
    parser.add_argument("--model-name", type=str, default='bert-base-uncased',
                        help="select which the model to probe: e.g., bert-base-uncased")
    parser.add_argument('--task', type=str, default='ptb_bert',
                        help="tasks: penn_treebank")
    args = parser.parse_args()
    print(args)

    main_func(args)
