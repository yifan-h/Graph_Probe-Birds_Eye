import os
import gc
import random
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from models import graph_probe
from embed import graph_embeddings, bert_embeddings, get_embeddings
from utils import load_data, construct_graph, load_glove, load_elmo, load_elmos


## sub-function descriptions
## please refer to main py file for function descriptions
'''
mine_probe: probing function. maximizing lower bound as estimation
'''


def mine_probe(args, graph_emb, bert_emb, sen_num, task_name, noisy_id=[]):
    bert_dim = bert_emb['s0'].shape[1]
    graph_dim = graph_emb['s0'].shape[1]
    if task_name == 'upper':
        bert_dim = graph_dim
    model = graph_probe(graph_dim, bert_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bad_np = [39927]
    mi_es = [-1 for _ in range(args.patience)]
    model.train()
    for epoch in range(10):  # epoch
        mi_train = []
        for i in range(sen_num):  # batch
            if i in bad_np: continue
            graph_vec = graph_emb['s'+str(i)]
            if task_name == 'lower':
                feat_vec = torch.randn(size=bert_emb['s'+str(i)].shape)
            elif task_name == 'upper':
                feat_vec = graph_vec
            elif type(task_name) == int:
                feat_vec = bert_emb['s'+str(i)]
            elif type(task_name) == float:
                vec_shape = bert_emb['s'+str(i)].shape
                feat_vec = (1-task_name) * bert_emb['s'+str(i)] + \
                            task_name * np.random.randn(vec_shape[0], vec_shape[1])
            elif task_name == 'noisy':
                feat_vec = bert_emb['s'+str(i)]
                graph_vec = graph_emb['s'+str(i)]
                vec_shape = graph_emb['s'+str(i)].shape
                rand_vec = np.random.randn(vec_shape[0], vec_shape[1])
                # graph_vec[noisy_id[i]] = 0.5*rand_vec[noisy_id[i]] +\
                #                          0.5*graph_vec[noisy_id[i]]
                graph_vec[noisy_id[i]] = rand_vec[noisy_id[i]]
            else:
                print('Error probe task name: ', task_name)
            graph_vec = torch.FloatTensor(graph_vec).to(args.device)
            feat_vec = torch.FloatTensor(feat_vec).to(args.device)

            optimizer.zero_grad()
            if graph_vec.shape[0] <= 1: continue
            if feat_vec.shape[0] <= 1: continue
            joint = model(graph_vec, feat_vec)
            feat_shuffle = feat_vec[torch.randperm(feat_vec.shape[0])]
            marginal = torch.exp(torch.clamp(model(graph_vec, feat_shuffle), max=88))
            mi = torch.mean(joint) - torch.log(torch.mean(marginal))
            loss = -mi
            mi_train.append(mi.data.item())

            loss.backward()
            optimizer.step()
        '''
        print("    Training probe model: {} | Epoch {:05d} | MI: {:.4f}".format(
                                                        task_name,
                                                        epoch + 1, 
                                                        sum(mi_train)/len(mi_train)))
        '''
        # early stop
        if sum(mi_train)/len(mi_train) > min(mi_es):
            mi_es.remove(min(mi_es))
            mi_es.append(sum(mi_train)/len(mi_train))
        else:
            break

    mi_eval = []
    model.eval()
    for i in range(sen_num):  # batch
        graph_vec = graph_emb['s'+str(i)]
        if task_name == 'lower':
            feat_vec = torch.randn(size=bert_emb['s'+str(i)].shape)
        elif task_name == 'upper':
            feat_vec = graph_vec
        elif type(task_name) == int:
            feat_vec = bert_emb['s'+str(i)]
        elif type(task_name) == float:
            vec_shape = bert_emb['s'+str(i)].shape
            feat_vec = (1-task_name) * bert_emb['s'+str(i)] + \
                        task_name * np.random.randn(vec_shape[0], vec_shape[1])
        elif task_name == 'noisy':
            feat_vec = bert_emb['s'+str(i)]
            graph_vec = graph_emb['s'+str(i)]
            vec_shape = graph_emb['s'+str(i)].shape
            rand_vec = np.random.randn(vec_shape[0], vec_shape[1])
            graph_vec[noisy_id[i]] = rand_vec[noisy_id[i]]
        else:
            print('Error probe task name: ', task_name)
        graph_vec = torch.FloatTensor(graph_vec).to(args.device)
        feat_vec = torch.FloatTensor(feat_vec).to(args.device)

        optimizer.zero_grad()
        if graph_vec.shape[0] <= 1: 
            mi_eval.append(0.)
            continue
        if feat_vec.shape[0] <= 1: 
            mi_eval.append(0.)
            continue
        joint = model(graph_vec, feat_vec)
        feat_shuffle = feat_vec[torch.randperm(feat_vec.shape[0])]
        marginal = torch.exp(torch.clamp(model(graph_vec, feat_shuffle), max=88))
        mi = torch.mean(joint) - torch.log(torch.mean(marginal))
        loss = -mi
        mi_eval.append(mi.data.item())

        loss.backward()
        optimizer.step()

    print(" ----Testing probe model: {} | Epoch {:05d} | MI: {:.4f}".format(
                                                        task_name,
                                                        epoch + 1, 
                                                        sum(mi_eval)/len(mi_eval)))

    # free memory
    model = None
    optimizer = None
    torch.cuda.empty_cache()
    gc.collect()

    # return [max(0, min(1, m)) for m in mi_eval]
    return mi_eval


'''
def npeet_probe(args, graph_emb, bert_emb, sen_num, task_name, noisy_id=[]):
    print(' ----Start to train autoencoder...')
    # train graph autoencoder
    graph_model = autoencoder(graph_emb['s0'].shape[1]).to(args.device)
    if not os.path.exists('./tmp/g_model_'+args.task+'.pkl'):
        optimizer = torch.optim.Adam(graph_model.parameters(), lr=args.lr)
        loss_fcn = torch.nn.MSELoss()
        graph_model.train()
        train_losses = [999 for _ in range(args.patience)]
        for _ in range(100):
            loss_train = 0
            for i in range(sen_num):  # batch
                graph_vec = graph_emb['s'+str(i)]
                if graph_vec.shape[0] <= 3: continue
                graph_vec = torch.FloatTensor(graph_vec).to(args.device)
                optimizer.zero_grad()
                pred = graph_model(graph_vec)
                loss = loss_fcn(pred, graph_vec)
                loss.backward()
                optimizer.step()
                loss_train += loss.data.item()
            loss_train = loss_train / sen_num
            print('----Training graph autoencoder loss: {:.4f}'.format(loss_train))
            # early stop
            if loss_train < max(train_losses):
                train_losses.remove(max(train_losses))
                train_losses.append(loss_train)
            else:
                break
        torch.save(graph_model.state_dict(), './tmp/g_model_'+args.task+'.pkl')
    graph_model.load_state_dict(torch.load('./tmp/g_model_'+args.task+'.pkl'))
    graph_model = graph_model.to(args.device)

    # train bert autoencoder
    bert_model = autoencoder(bert_emb['s0'].shape[1]).to(args.device)
    if not os.path.exists('./tmp/b_model_'+args.task+'.pkl'):
        optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr)
        loss_fcn = torch.nn.MSELoss()
        bert_model.train()
        train_losses = [999 for _ in range(args.patience)]
        for _ in range(100):
            loss_train = 0
            for i in range(sen_num):  # batch
                feat_vec = bert_emb['s'+str(i)]
                if feat_vec.shape[0] <= 3: continue
                feat_vec = torch.FloatTensor(feat_vec).to(args.device)
                optimizer.zero_grad()
                pred = bert_model(feat_vec)
                loss = loss_fcn(pred, feat_vec)
                loss.backward()
                optimizer.step()
                loss_train += loss.data.item()
            loss_train = loss_train / sen_num
            print('----Training BERT autoencoder loss: {:.4f}'.format(loss_train))
            # early stop
            if loss_train < max(train_losses):
                train_losses.remove(max(train_losses))
                train_losses.append(loss_train)
            else:
                break
        torch.save(bert_model.state_dict(), './tmp/b_model_'+args.task+'.pkl')
    bert_model.load_state_dict(torch.load('./tmp/b_model_'+args.task+'.pkl'))
    bert_model = bert_model.to(args.device)

    from npeet import entropy_estimators as ee
    from pycit import itest
    print(' ----Start to calculate low-dimensional representations...')
    graph_model.eval()
    bert_model.eval()
    mi_estimate = []
    graph_vecs, bert_vecs = [], []
    for i in tqdm(range(sen_num)):  # batch
        graph_vec = torch.FloatTensor(graph_emb['s'+str(i)]).to(args.device)
        graph_vec = graph_model(graph_vec, True)
        if task_name == 'lower':
            vec_shape = graph_vec.shape
            feat_vec = np.random.randn(vec_shape[0], vec_shape[1])
        elif task_name == 'upper':
            feat_vec = graph_vec
        elif type(task_name) == int:
            feat_vec = torch.FloatTensor(bert_emb['s'+str(i)]).to(args.device)
            feat_vec = bert_model(feat_vec, True)
        elif type(task_name) == float:
            feat_vec = torch.FloatTensor(bert_emb['s'+str(i)]).to(args.device)
            feat_vec = bert_model(feat_vec, True)
            feat_vec = (1 - task_name) * feat_vec + \
                            task_name * np.random.randn(feat_vec[0], feat_vec[1])
        else:
            print('Error probe task name: ', task_name)
        graph_vecs.append(graph_vec)
        bert_vecs.append(feat_vec)
    graph_vecs = np.concatenate(graph_vecs)
    bert_vecs = np.concatenate(bert_vecs)

    print(' ----Start to calculate MI with NPEET...')
    batch_size = 1000
    ksg_mis, bi_ksg_mis = [], []
    for i in tqdm(range(int(graph_vecs.shape[0]/batch_size))):
        range_lower = i*batch_size
        range_upper = min(graph_vecs.shape[0], (i+1)*batch_size)
        ksg_mis.append(itest(graph_vecs[range_lower:range_upper], 
                             bert_vecs[range_lower:range_upper], 
                             test_args={'statistic': 'ksg_mi', 'n_jobs': 10}))
        bi_ksg_mis.append(itest(graph_vecs[range_lower:range_upper], 
                             bert_vecs[range_lower:range_upper], 
                             test_args={'statistic': 'bi_ksg_mi', 'n_jobs': 10}))
        # mi_estimate.append(ee.mi(graph_vecs[range_lower:range_upper], 
        #                          bert_vecs[range_lower:range_upper], k=knn_k))
        # if graph_vec.shape[0] <= knn_k+8: continue
        # if feat_vec.shape[0] <= knn_k+8: continue
        # mi_estimate.append(ee.mi(graph_vec, feat_vec, k=knn_k))
    # mi_estimate = sum(mi_estimate) / len(mi_estimate)
    # print(' ----Testing estimate MI value: {:.4f}'.format(mi_estimate))
    ksg_mi_val = sum(ksg_mis) / len(ksg_mis)
    bi_ksg_mi_val = sum(bi_ksg_mis) / len(bi_ksg_mis)
    print(' ---Testing estimate MI value: {:.4f} | {:.4f}'.format(ksg_mi_val, bi_ksg_mi_val))
    # free memory
    gc.collect()

    return mi_estimate
'''


def mi_bert_ptb(args, npeet=False, uncontext=False):
    # load data
    s_train, p_train = load_data('penn_treebank_dataset', 'train')
    s_dev, p_dev = load_data('penn_treebank_dataset', 'dev')
    s_test, p_test = load_data('penn_treebank_dataset', 'test')
    sentences = s_train + s_dev + s_test
    parsed = p_train + p_dev + p_test
    doc_id, sen_id, global_graph = construct_graph(parsed)
    s_train, p_train, s_dev, p_dev, s_test, p_test = [], [], [], [], [], []

    # load embeddings
    graph_emb = graph_embeddings(args, global_graph, doc_id, sen_id)
    if uncontext:
        bert_emb = load_glove(args, sentences)
        # bert_emb = load_elmo(args, sentences)
    else:
        bert_emb_paths = bert_embeddings(args, sentences)
        # bert_emb_paths = load_elmos(args, sentences)
        bert_emb = np.load(bert_emb_paths[0], allow_pickle=True)

    # initialize mi
    mir, mig, mib = [], [], []
    for l in range(args.bert_layers_num): mib.append([])
    for s in range(len(sentences)):
        mir.append(0.)
        mig.append(0.)
        for l in range(args.bert_layers_num):
            mib[l].append(0.)

    if args.baselines:
        print('3.1 start to calculate baselines of MI...')
        # calculate MI baselines
        for r in range(args.repeat):
            tmp_mir = mine_probe(args, graph_emb, bert_emb, len(sentences), 'lower')
            tmp_mig = mine_probe(args, graph_emb, bert_emb, len(sentences), 'upper')
            # get sum value
            mir = [mir[s]+tmp_mir[s] for s in range(len(tmp_mir))]
            mig = [mig[s]+tmp_mig[s] for s in range(len(tmp_mig))]

    print('3.2 start to calculate BERT hidden states of MI...')
    if uncontext:
        for r in range(args.repeat):
            tmp_mib = mine_probe(args, graph_emb, bert_emb, len(sentences), 
                                                        args.bert_layers_num - 1)
            mib[-1] = [mib[-1][s]+tmp_mib[s] for s in range(len(tmp_mib))]
        mib_layers = sum(mib[-1]) / (len(mib[-1]) * args.repeat)
        print('MI(G, Glove): {} |'.format(mib_layers))
    else:
        # calculate MI of BERT
        for l in range(args.bert_layers_num):
            bert_emb = np.load(bert_emb_paths[l], allow_pickle=True)
            for r in range(args.repeat):
                tmp_mib = mine_probe(args, graph_emb, bert_emb, len(sentences), l)
                mib[l] = [mib[l][s]+tmp_mib[s] for s in range(len(tmp_mib))]
        # compute average values for all results
        mir = [mi/args.repeat for mi in mir]
        mig = [mi/args.repeat for mi in mig]
        for l in range(args.bert_layers_num):
            mib[l] = [mi/args.repeat for mi in mib[l]]
        mib_layers = [sum(mib[l])/len(mib[l]) for l in range(len(mib))]

        # print general results
        results = {'lower:': mir, 'upper': mig, 'bert': mib}
        # print('\n', results, '\n')
        
        print('MI(G, R): {} | MI(G, G): {}| MI(G, BERT): {} |'.format(sum(
                                                                    mir)/len(mir),
                                                                    sum(mig)/len(mig),
                                                                    mib_layers))

    return


def mi_bert_amr(args, uncontext=False):
    # load data & embeddings
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    amr_s = s_train + s_dev + s_test
    print(amr_s[45672], amr_s[599])
    graph_emb, bert_emb_paths = get_embeddings(args, amr_s)
    # bert_emb_paths = load_elmos(args, amr_s, dataset='amr')
    s_num = len(graph_emb)
    if uncontext:
        bert_emb = load_glove(args, amr_s, dataset='amr')
        # bert_emb = load_elmo(args, amr_s, dataset='amr')
    else:
        bert_emb = np.load(bert_emb_paths[0], allow_pickle=True)

    print('2.1 start to calculate baselines of MI...')
    # initialize mi
    mir, mig, mib = [], [], []
    for l in range(args.bert_layers_num): mib.append([])

    if args.baselines:
        print('3.1 start to calculate baselines of MI...')
        # calculate MI baselines
        for r in range(args.repeat):
            tmp_mir = mine_probe(args, graph_emb, bert_emb, s_num, 'lower')
            tmp_mig = mine_probe(args, graph_emb, bert_emb, s_num, 'upper')
            # get sum value
            if len(mir) == 0: 
                mir = tmp_mir
            else:
                mir = [mir[s]+tmp_mir[s] for s in range(len(tmp_mir))]
            if len(mig) == 0:
                mig = tmp_mig
            else:
                mig = [mig[s]+tmp_mig[s] for s in range(len(tmp_mig))]

    print('2.2 start to calculate BERT hidden states of MI...')
    # calculate MI of BERT
    if uncontext:
        for r in range(args.repeat):
            tmp_mib = mine_probe(args, graph_emb, bert_emb, s_num, args.bert_layers_num-1)
            if len(mib[-1]) == 0: 
                mib[-1] = tmp_mib
            else:
                mib[-1] = [mib[-1][s]+tmp_mib[s] for s in range(len(tmp_mib))]
        mib_layers = sum(mib[-1]) / (len(mib[-1]) * args.repeat)
        print('MI(G, Glove): {} |'.format(mib_layers))
    else:
        for l in range(args.bert_layers_num):
            bert_emb = np.load(bert_emb_paths[l], allow_pickle=True)
            for r in range(args.repeat):
                tmp_mib = mine_probe(args, graph_emb, bert_emb, s_num, l)
                if len(mib[l]) == 0:
                    mib[l] = tmp_mib
                else:
                    mib[l] = [mib[l][s]+tmp_mib[s] for s in range(len(tmp_mib))]

        # compute average values for all results
        mir = [mi/args.repeat for mi in mir]
        mig = [mi/args.repeat for mi in mig]
        for l in range(args.bert_layers_num):
            mib[l] = [mi/args.repeat for mi in mib[l]]

        # print general results
        results = {'lower:': mir, 'upper': mig, 'bert': mib}
        print('\n', results, '\n')
        mib_layers = [sum(mib[l])/len(mib[l]) for l in range(len(mib)) if len(mib)]
        print('MI(G, R): {} | MI(G, G): {}| MI(G, BERT): {} |'.format(sum(
                                                                    mir)/len(mir),
                                                                    sum(mig)/len(mig),
                                                                    mib_layers))

    return


def mi_mlps_ptb(args):
    # load data
    s_train, p_train = load_data('penn_treebank_dataset', 'train')
    s_dev, p_dev = load_data('penn_treebank_dataset', 'dev')
    s_test, p_test = load_data('penn_treebank_dataset', 'test')
    sentences = s_train + s_dev + s_test
    parsed = p_train + p_dev + p_test
    doc_id, sen_id, global_graph = construct_graph(parsed)
    s_train, p_train, s_dev, p_dev, s_test, p_test = [], [], [], [], [], []

    # load embeddings
    graph_emb = graph_embeddings(args, global_graph, doc_id, sen_id)
    bert_emb = load_glove(args, sentences)
    # bert_emb = load_elmo(args, sentences)

    # bert_emb_paths = bert_embeddings(args, sentences)
    # bert_emb = np.load(bert_emb_paths[0], allow_pickle=True)


    # initialize mi
    mir, mig, mib = [], [], []
    for l in range(args.bert_layers_num): mib.append([])
    for s in range(len(sentences)):
        mir.append(0.)
        mig.append(0.)
        for l in range(args.bert_layers_num):
            mib[l].append(0.)

    if args.baselines:
        print('3.1 start to calculate baselines of MI...')
        # calculate MI baselines
        for r in range(args.repeat):
            tmp_mir = mine_probe(args, graph_emb, bert_emb, len(sentences), 'lower')
            tmp_mig = mine_probe(args, graph_emb, bert_emb, len(sentences), 'upper')
            # get sum value
            mir = [mir[s]+tmp_mir[s] for s in range(len(tmp_mir))]
            mig = [mig[s]+tmp_mig[s] for s in range(len(tmp_mig))]

    print('3.2 start to calculate BERT hidden states of MI...')
    for r in range(args.repeat):
        tmp_mib = mine_probe(args, graph_emb, bert_emb, len(sentences), 
                                                    args.bert_layers_num - 1)
        mib[-1] = [mib[-1][s]+tmp_mib[s] for s in range(len(tmp_mib))]
    mib_layers = sum(mib[-1]) / (len(mib[-1]) * args.repeat)
    print('MI(G, Glove): {} |'.format(mib_layers))



def mi_mlps_amr(args):
    return
