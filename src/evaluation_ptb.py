import torch
import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
from tqdm import tqdm

from probe import mine_probe
from models import binary_classifer
from embed import graph_embeddings, bert_embeddings
from utils import get_edge_idx, load_data, construct_graph, get_graph_emb, \
                    load_noisy_trees, uuas_score, load_tree_labels, load_split_emb


## sub-function descriptions
## please refer to main py file for function descriptions
'''
load_graph: load graph embedding
load_bert: load BERT embedding
load_embeddings: load graph embedding and BERT embedding (only top layer)
'''


def load_graph(args, data_split=True):
    if not data_split:
        _, p_train = load_data('penn_treebank_dataset', 'train')
        doc_id, sen_id_train, global_graph = construct_graph(p_train)
        _, p_dev = load_data('penn_treebank_dataset', 'dev')
        doc_id, sen_id_dev, global_graph = construct_graph(p_dev)
        _, p_test = load_data('penn_treebank_dataset', 'test')
        doc_id, sen_id_test, global_graph = construct_graph(p_test)
        parsed = p_train + p_dev + p_test
        sen_id = sen_id_train + sen_id_dev + sen_id_test
        graph_emb = graph_embeddings(args, global_graph, doc_id, sen_id)
        return graph_emb, sen_id
    else:
        _, p_train = load_data('penn_treebank_dataset', 'train')
        doc_id, sen_id_train, global_graph = construct_graph(p_train)
        ge_train = graph_embeddings(args, global_graph, doc_id, sen_id_train, '_train')
        _, p_dev = load_data('penn_treebank_dataset', 'dev')
        doc_id, sen_id_dev, global_graph = construct_graph(p_dev)
        ge_dev = graph_embeddings(args, global_graph, doc_id, sen_id_dev, '_dev')
        _, p_test = load_data('penn_treebank_dataset', 'test')
        doc_id, sen_id_test, global_graph = construct_graph(p_test)
        ge_test = graph_embeddings(args, global_graph, doc_id, sen_id_test, '_test')
        return ge_train, ge_dev, ge_test, sen_id_train, sen_id_dev, sen_id_test


def load_bert(args):
    s_train, p_train = load_data('penn_treebank_dataset', 'train')
    doc_id, sen_id_train, global_graph = construct_graph(p_train)
    bert_train_paths = bert_embeddings(args, s_train, '_train')
    bert_train = np.load(bert_train_paths[-1])

    s_dev, p_dev = load_data('penn_treebank_dataset', 'dev')
    doc_id, sen_id_dev, global_graph = construct_graph(p_dev)
    bert_dev_paths = bert_embeddings(args, s_dev, '_dev')
    bert_dev = np.load(bert_dev_paths[-1])

    s_test, p_test = load_data('penn_treebank_dataset', 'test')
    doc_id, sen_id_test, global_graph = construct_graph(p_test)
    bert_test_paths = bert_embeddings(args, s_test, '_test')
    bert_test = np.load(bert_test_paths[-1])

    return bert_train, bert_dev, bert_test, sen_id_train, sen_id_dev, sen_id_test


def load_embeddings(args):
    # load data
    s_train, p_train = load_data('penn_treebank_dataset', 'train')
    s_dev, p_dev = load_data('penn_treebank_dataset', 'dev')
    s_test, p_test = load_data('penn_treebank_dataset', 'test')
    sentences = s_train + s_dev + s_test
    parsed = p_train + p_dev + p_test
    # sentences = s_test
    # parsed = p_test
    doc_id, sen_id, global_graph = construct_graph(parsed)
    # load embeddings
    graph_emb = graph_embeddings(args, global_graph, doc_id, sen_id)
    bert_emb_paths = bert_embeddings(args, sentences)
    # graph_emb = graph_embeddings(args, global_graph, doc_id, sen_id, '_test')
    # bert_emb_paths = bert_embeddings(args, sentences, '_test')
    bert_emb = np.load(bert_emb_paths[-1])

    return graph_emb, bert_emb


def test_ge_ptb(args, data_split=True):
    # load data
    if data_split:
        ge_train, ge_dev, ge_test, sid_train, sid_dev, sid_test \
                                        = load_graph(args, data_split)
    else:
        graph_emb, sen_id = load_graph(args, data_split)
        ge_train, ge_dev, ge_test = graph_emb, graph_emb, graph_emb
        sid_train, sid_dev, sid_test = sen_id, sen_id, sen_id

    # test_tasks = ['all', 'local', 'global']
    test_tasks = ['local']
    aucs, jaccards, uuass = {}, {}, {}

    for test_task in test_tasks:
        feat_dim = ge_train['s0'].shape[1]
        model = binary_classifer(args.classifier_layers_num,
                                    feat_dim).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fcn = torch.nn.BCELoss()

        print('2. start to train model: ', test_task)
        # train
        model.train()
        train_losses = [999 for _ in range(args.patience)]
        for _ in range(10):  # epoch
            loss_train = 0
            for s in range(len(sid_train)):
                # get graph embedding
                graph_emb = ge_train['s'+str(s)]
                graph_emb = torch.FloatTensor(graph_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx(sid_train[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue
                optimizer.zero_grad()
                edge_pred = model(graph_emb[src_idx], graph_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred)
                loss = loss_fcn(edge_pred, edge_labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.data.item()
            loss_train = loss_train/len(sid_train)
            print('   the training loss is: {:.4f}'.format(loss_train))
            # early stop
            if loss_train < max(train_losses):
                train_losses.remove(max(train_losses))
                train_losses.append(loss_train)
            else:
                break

        # validation
        if data_split:
            print('2. start to validate model: ', test_task)
            # validation
            model.eval()
            loss_dev = 0
            for s in range(len(sid_dev)):
                # get graph embedding
                graph_emb = ge_dev['s'+str(s)]
                graph_emb = torch.FloatTensor(graph_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx(sid_dev[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue
                edge_pred = model(graph_emb[src_idx], graph_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred)
                loss = loss_fcn(edge_pred, edge_labels)
                loss_dev += loss.data.item()
            loss_dev = loss_dev/len(sid_dev)
            print('2. start to test model: {} |Train loss: {:.4f} |Val loss: {:.4f}'.format(
                                                            test_task, loss_train, loss_dev))

        # test
        model.eval()
        auc, jaccard, uuas = [], [], []
        print('2. start to test model: ', test_task)
        for s in range(len(sid_test)):
            # get graph embedding
            graph_emb = ge_test['s'+str(s)]
            graph_emb = torch.FloatTensor(graph_emb).to(args.device)
            # get ground-truth graph
            src_idx, dst_idx, edge_labels = get_edge_idx(sid_test[s])
            edge_labels = torch.FloatTensor(edge_labels).to(args.device)
            if len(src_idx) <= 1: continue
            edge_pred = model(graph_emb[src_idx], graph_emb[dst_idx])
            edge_pred = torch.squeeze(edge_pred).detach().cpu().numpy()
            edge_labels = edge_labels.detach().cpu().numpy()
            if edge_labels.sum() > 0: 
                auc.append(roc_auc_score(edge_labels, edge_pred))
                edge_pred = np.where(edge_pred > 0.5, 1, 0)
                jaccard.append(jaccard_score(edge_labels, edge_pred))
                # uuas.append(uuas_score(src_idx, dst_idx, edge_labels, edge_pred))
        aucs[test_task] = auc
        jaccards[test_task] = jaccard
        uuass[test_task] = uuas

    print(sum(aucs['local']) / len(aucs['local']), \
            sum(jaccards['local']) / len(jaccards['local']))

    return


def test_bert_ptb(args, model_name='bert'):
    # load data
    bert_train, bert_dev, bert_test, sid_train, sid_dev, sid_test = load_bert(args)
    if model_name != 'bert':
        bert_train, bert_dev, bert_test = load_split_emb(len(bert_train), len(bert_dev),
                                                        len(bert_test), model_name, 'ptb')
    feat_dim = bert_train['s0'].shape[1]
    model = binary_classifer(args.classifier_layers_num, 
                                feat_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = torch.nn.BCELoss()

    print('2. start to train model...')
    # train
    model.train()
    train_losses = [999 for _ in range(args.patience)]
    for _ in range(10):  # epoch
        loss_train = 0
        for s in range(len(sid_train)):
            # get graph embedding
            bert_emb = bert_train['s'+str(s)]
            bert_emb = torch.FloatTensor(bert_emb).to(args.device)
            # get ground-truth graph
            src_idx, dst_idx, edge_labels = get_edge_idx(sid_train[s])
            edge_labels = torch.FloatTensor(edge_labels).to(args.device)
            if len(src_idx) <= 1: continue
            optimizer.zero_grad()
            edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
            edge_pred = torch.squeeze(edge_pred)
            loss = loss_fcn(edge_pred, edge_labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.data.item()
        loss_train = loss_train/len(sid_train)
        print('   the training loss is: {:.4f}'.format(loss_train))
        # early stop
        if loss_train < max(train_losses):
            train_losses.remove(max(train_losses))
            train_losses.append(loss_train)
        else:
            break

    print('2. start to validate model...')
    # validation
    model.eval()
    loss_dev = 0
    for s in range(len(sid_dev)):
        # get graph embedding
        bert_emb = bert_dev['s'+str(s)]
        bert_emb = torch.FloatTensor(bert_emb).to(args.device)
        # get ground-truth graph
        src_idx, dst_idx, edge_labels = get_edge_idx(sid_dev[s])
        edge_labels = torch.FloatTensor(edge_labels).to(args.device)
        if len(src_idx) <= 1: continue

        edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
        edge_pred = torch.squeeze(edge_pred)
        loss = loss_fcn(edge_pred, edge_labels)
        loss_dev += loss.data.item()
    loss_dev = loss_dev/len(sid_dev)

    print('2. | Train loss: {:.4f} | Val loss: {:.4f}'.format(loss_train, loss_dev))
    # test
    model.eval()
    auc, jaccard, uuas = [], [], []
    print('2. start to test model...')
    for s in range(len(sid_test)):
        # get graph embedding
        bert_emb = bert_test['s'+str(s)]
        bert_emb = torch.FloatTensor(bert_emb).to(args.device)
        # get ground-truth graph
        src_idx, dst_idx, edge_labels = get_edge_idx(sid_test[s])
        edge_labels = torch.FloatTensor(edge_labels).to(args.device)
        if len(src_idx) <= 1: continue
        edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
        edge_pred = torch.squeeze(edge_pred).detach().cpu().numpy()
        edge_labels = edge_labels.detach().cpu().numpy()
        if edge_labels.sum() > 0: 
            auc.append(roc_auc_score(edge_labels, edge_pred))
            edge_pred = np.where(edge_pred > 0.5, 1, 0)
            jaccard.append(jaccard_score(edge_labels, edge_pred))
            # uuas.append(uuas_score(src_idx, dst_idx, edge_labels, edge_pred))

    print(sum(auc)/len(auc), sum(jaccard)/len(jaccard))

    return


def test_mi_ptb(args):
    print('4.1 start to load embeddings...')
    graph_emb, bert_emb = load_embeddings(args)

    print('4.2 start to test graph MI...')
    # graph task = graph_emb: {all, local, global} X graph noise {0.0 ~ 1.0}
    graph_results = {'local':[]}
    '''
    graph_results = {'all':[], 'local':[], 'global':[]}
    for k, _ in graph_results.items():
        graph_emb = get_graph_emb(graph_emb, k)
        task_mi = []
        for n in tqdm(range(11)):
            noise_mi = []
            for r in range(args.repeat):
                mi_s = mine_probe(args, graph_emb, graph_emb, len(graph_emb), n/10)
                mi_s = sum(mi_s)/len(mi_s)
                noise_mi.append(mi_s)
            task_mi.append(noise_mi)
        graph_results[k] = task_mi
    '''

    print('4.2 start to test bert MI...')
    # bert task = graph_emb: {all, local, global} X bert noise {0.0 ~ 1.0}
    # bert_results = {'all':[], 'local':[], 'global':[]}
    bert_results = {'local':[]}
    for k, _ in bert_results.items():
        task_mi = []
        for n in tqdm(range(11)):
            noise_mi = []
            for r in range(args.repeat):
                mi_s = mine_probe(args, graph_emb, bert_emb, len(graph_emb), n/10)
                mi_s = sum(mi_s)/len(mi_s)
                noise_mi.append(mi_s)
            task_mi.append(noise_mi)
        bert_results[k] = task_mi
    print(graph_results, bert_results)
    return


def mi_noise_ptb(args, pos=False):
    graph_emb, bert_emb = load_embeddings(args)
    noisy_g = load_noisy_trees(args, pos)
    results = {}
    for k, v in noisy_g.items():
        if k not in results: results[k] = []
        for r in range(args.repeat):
            mi = mine_probe(args, graph_emb, bert_emb, len(graph_emb), 'noisy', v)
            results[k].append(sum(mi) / len(mi))
    print(results)
    return


def test_random_ptb(args, pos=False, corrupt=False):
    # load data
    bert_train, bert_dev, bert_test, sid_train, sid_dev, sid_test = load_bert(args)
    feat_dim = args.bert_hidden_num
    noisy_g = load_noisy_trees(args, data_split=True)

    for noisy_tag, noisy_id in noisy_g.items():
        print('2. corrupt type: ', noisy_tag)
        model = binary_classifer(args.classifier_layers_num, feat_dim).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fcn = torch.nn.BCELoss()
        print('2. start to train model...')
        # train
        model.train()
        train_losses = [999 for _ in range(args.patience)]
        for _ in range(10):  # epoch
            loss_train = 0
            for s in range(len(sid_train)):
                # get graph embedding
                bert_emb = bert_train['s'+str(s)]
                if corrupt:
                    rand_vec = np.random.randn(bert_emb.shape[0], bert_emb.shape[1])
                    bert_emb[noisy_id[s]] = rand_vec[noisy_id[s]]
                bert_emb = torch.FloatTensor(bert_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx(sid_train[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue

                optimizer.zero_grad()
                edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred)
                loss = loss_fcn(edge_pred, edge_labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.data.item()
            loss_train = loss_train/len(sid_train)
            print('   the training loss is: {:.4f}'.format(loss_train))
            # early stop
            if loss_train < max(train_losses):
                train_losses.remove(max(train_losses))
                train_losses.append(loss_train)
            else:
                break

        # test
        model.eval()
        label_ids = load_tree_labels(args, pos)
        auc_dict = {}
        print('2. start to test model...')
        for k, v_l in label_ids.items():
            auc_tmp = []
            for s in range(len(sid_test)):
                # get graph embedding
                bert_emb = bert_test['s'+str(s)]
                bert_emb = torch.FloatTensor(bert_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx(sid_test[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue
                edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred).detach().cpu().numpy()
                edge_labels = edge_labels.detach().cpu().numpy()
                edge_mask = []
                for i in range(len(sid_test[s])):
                    for j in range(len(sid_test[s])):
                        if i in v_l[s] or j in v_l[s]:
                            edge_mask.append(1)
                        else:
                            edge_mask.append(0)
                edge_mask = np.array(edge_mask)
                edge_pred = edge_pred[edge_mask==1]
                edge_labels = edge_labels[edge_mask==1]
                if edge_labels.sum() > 0: 
                    auc_tmp.append(roc_auc_score(edge_labels, edge_pred))
            auc_dict[k] = sum(auc_tmp) / (1e-5 + len(auc_tmp))
        print(auc_dict)

    return