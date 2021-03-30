import os
import penman
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertForPreTraining
from utils import random_walks, load_data, clean_string


## sub-function descriptions
## please refer to main py file for function descriptions
'''
graph_embeddings: load / calculate graph embedding (PTB dataset)
bert_embeddings: load / calculate BERT embedding (PTB dataset)
get_embeddings: load / calculate graph embedding and BERT embedding (AMR dataset)
'''


def graph_embeddings(args, global_graph, doc_id, sen_id, data_div=''):
    if not os.path.exists('./tmp/ge_'+args.task+data_div+'.npz'):
        '''
        # get global graph embedding ge
        print('1.2 start to calculate global graph embedding...')
        global_walks = random_walks(global_graph)
        global_model = Word2Vec(global_walks, 
                                size=640, 
                                window=2, 
                                min_count=0, 
                                sg=1, 
                                hs=1, 
                                workers=20)
        '''
        # get local graph embedding le
        print('1.3 start to calculate local graphs embedding...')
        ge_vec = []
        for i in tqdm(range(len(doc_id))):
            # global_idx = [str(idx[1]) for idx in doc_id[i]]
            local_idx = [str(idx[1]) for idx in sen_id[i]]
            # get local graph embeddings
            local_graph = nx.Graph()
            for (s, t) in sen_id[i]: local_graph.add_edge(s, t)
            local_walks = random_walks(local_graph, 100, 10)
            if len(local_idx) > 1:
                local_model = Word2Vec(local_walks, 
                                size=128, 
                                window=2, 
                                min_count=0, 
                                sg=1, 
                                hs=1, 
                                workers=20)
                local_vec = local_model.wv[local_idx]
            else:
                local_vec = np.zeros((1,128))
            # save graph embeddings (global + local)
            # global_vec = global_model.wv[global_idx]
            # ge_vec.append(np.concatenate((global_vec, local_vec), axis=1))
            ge_vec.append(local_vec)

        # save graph embeddings
        savez_dict = {}
        for i in range(len(ge_vec)): savez_dict['s'+str(i)] = ge_vec[i]
        np.savez('./tmp/ge_'+args.task+data_div+'.npz', **savez_dict)

    return np.load('./tmp/ge_'+args.task+data_div+'.npz')


def bert_embeddings(args, sentences, data_div=''):
    if args.model_name == 'bert-base-uncased':
        data_path = './tmp/be12_'+args.task+data_div+'.npz'
    else:
        data_path = './tmp/be24_'+args.task+data_div+'.npz'
    if not os.path.exists(data_path):
        # get BERT hidden representations
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        model = BertForPreTraining.from_pretrained(args.model_name, 
                                                    return_dict=True, 
                                                    output_hidden_states = True)
        bert_hs = [[] for l in range(args.bert_layers_num)]
        for s in tqdm(sentences):
            inputs = tokenizer(s, return_tensors="pt", is_split_into_words = True)
            outputs = model(**inputs)
            outputs = outputs.hidden_states

            # average word pieces to get whole word embedding
            s_pieces = tokenizer.tokenize(' '.join(s))
            w_ids = []
            for i in range(len(s_pieces)):
                if len(s_pieces[i]) < 2 or s_pieces[i][:2] != '##':
                    w_ids.append(i)
            # check piece number
            if len(w_ids) != len(s):
                print('Error! failed to get whole word embedding!', s, s_pieces, w_ids)
            hidden_s = []
            for l in range(len(outputs)):
                # remove EOS BOS tokens
                piece_embed = torch.squeeze(outputs[l].data)[1:-1]
                # get word embeddings
                word_embed = piece_embed[w_ids]
                # average word embedding
                for i in range(len(w_ids)-1):
                    if w_ids[i+1] - w_ids[i] != 1:
                        tmp_idx = [w_ids[i]+j for j in range(w_ids[i+1] - w_ids[i])]
                        word_embed[i] = torch.mean(piece_embed[tmp_idx], dim=0)
                hidden_s.append(word_embed)
            # bert embedding for sentence s: len(s) * 768
            for l in range(len(hidden_s)):
                bert_hs[l].append(hidden_s[l].detach().cpu().data.numpy())

        if len(data_div):
            l = args.bert_layers_num - 1
            savez_dict = {}
            for i in range(len(bert_hs[l])): savez_dict['s'+str(i)] = bert_hs[l][i]
            np.savez('./tmp/be'+str(l)+'_'+args.task+data_div+'.npz', **savez_dict)
        else:
            # save bert embeddings
            for l in range(args.bert_layers_num):
                savez_dict = {}
                for i in range(len(bert_hs[l])): savez_dict['s'+str(i)] = bert_hs[l][i]
                np.savez('./tmp/be'+str(l)+'_'+args.task+data_div+'.npz', **savez_dict)

    return ['./tmp/be'+str(l)+'_'+args.task+data_div+'.npz' for l in range(args.bert_layers_num)]


def get_embeddings(args, amr_s, data_div=''):
    print('1. start to parse and embed the sentences...')
    if args.model_name == 'bert-base-uncased':
        data_path = './tmp/be12_'+args.task+data_div+'.npz'
    else:
        data_path = './tmp/be24_'+args.task+data_div+'.npz'
    if not os.path.exists(data_path):
        # get BERT hidden representations
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        model = BertForPreTraining.from_pretrained(args.model_name, 
                                                    return_dict=True, 
                                                    output_hidden_states = True)
        ge_vecs = []
        bert_hs = [[] for l in range(args.bert_layers_num)]
        for s in tqdm(amr_s):
            # parse
            penman_g = penman.decode(s)
            s = penman_g.metadata.get('tok').split(' ')
            wid = []
            var = []  # k=word id; v=variable
            for k, v in penman_g.epidata.items():
                if k[1] == ':instance':
                    if len(v):
                        if type(v[0]) == penman.surface.Alignment:
                            wid.append(v[0].indices[0])
                            var.append(k[0])

            # BERT embedding
            c_s = []
            for w in s:
                c_w = clean_string(w)
                if len(c_w) == 0: c_w = ','
                c_s.append(c_w)
            inputs = tokenizer(c_s, return_tensors="pt", is_split_into_words = True)
            outputs = model(**inputs)
            outputs = outputs.hidden_states
            # average word pieces to get whole word embedding
            s_pieces = tokenizer.tokenize(' '.join(c_s))
            w_ids = []
            for i in range(len(s_pieces)):
                if len(s_pieces[i]) < 2 or s_pieces[i][:2] != '##':
                    w_ids.append(i)
            # check piece number
            if len(w_ids) != len(c_s):
                print('Error! failed to get BERT word embedding!', c_s, s_pieces, w_ids)
            hidden_s = []
            for l in range(len(outputs)):
                # remove EOS BOS tokens
                piece_embed = torch.squeeze(outputs[l].data)[1:-1]
                # get word embeddings
                word_embed = piece_embed[w_ids]
                # average word embedding
                for i in range(len(w_ids)-1):
                    if w_ids[i+1] - w_ids[i] != 1:
                        tmp_idx = [w_ids[i]+j for j in range(w_ids[i+1] - w_ids[i])]
                        word_embed[i] = torch.mean(piece_embed[tmp_idx], dim=0)
                hidden_s.append(word_embed[wid])
            # bert embedding for sentence c_s: 13 - len(c_s) * 768
            for l in range(len(hidden_s)):
                bert_hs[l].append(hidden_s[l].detach().cpu().data.numpy())

            # graph embedding
            g = nx.Graph()
            for v in penman_g.variables(): g.add_node(v)
            for e in penman_g.edges(): g.add_edge(e.source, e.target)
            walks = random_walks(g, 50, 10, True)
            if len(var) > 1:
                ge_model = Word2Vec(walks, 
                                size=128, 
                                window=2, 
                                min_count=0, 
                                sg=1, 
                                hs=1, 
                                workers=20)
                ge_vec = ge_model.wv[var]
            else:
                ge_vec = np.zeros((1,args.bert_hidden_num))
            ge_vecs.append(ge_vec)

        if len(data_div):
            l = args.bert_layers_num - 1
            savez_dict = {}
            for i in range(len(bert_hs[l])): savez_dict['s'+str(i)] = bert_hs[l][i]
            np.savez('./tmp/be'+str(l)+'_'+args.task+data_div+'.npz', **savez_dict)
        else:
            # save bert embedding
            for l in range(args.bert_layers_num):
                savez_dict = {}
                for i in range(len(bert_hs[l])): savez_dict['s'+str(i)] = bert_hs[l][i]
                np.savez('./tmp/be'+str(l)+'_'+args.task+data_div+'.npz', **savez_dict)
        # save graph embeddings
        savez_dict = {}
        for i in range(len(ge_vecs)): savez_dict['s'+str(i)] = ge_vecs[i]
        np.savez('./tmp/ge_'+args.task+data_div+'.npz', **savez_dict)

    return np.load('./tmp/ge_'+args.task+data_div+'.npz'), \
                ['./tmp/be'+str(l)+'_'+args.task+data_div+'.npz' for l in range(args.bert_layers_num)]