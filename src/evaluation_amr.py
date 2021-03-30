import torch
import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
from tqdm import tqdm

from probe import mine_probe
from models import binary_classifer
from embed import get_embeddings
from utils import load_data, get_edge_idx_amr, load_noisy_graphs, uuas_score,\
                    load_graph_labels, load_split_emb

def test_ge_amr(args, data_split=True):
    # load data & graph embeddings
    if data_split:
        s_train = load_data('amr_dataset', 'train')
        s_dev = load_data('amr_dataset', 'dev')
        s_test = load_data('amr_dataset', 'test')
        ge_train, _ = get_embeddings(args, s_train, data_div='_train')
        ge_dev, _ = get_embeddings(args, s_dev, data_div='_dev')
        ge_test, _ = get_embeddings(args, s_test, data_div='_test')
    else:
        s_train = load_data('amr_dataset', 'train')
        s_dev = load_data('amr_dataset', 'dev')
        s_test = load_data('amr_dataset', 'test')
        s_train = s_train + s_dev + s_test
        s_dev, s_test = s_train, s_train
        ge_train, _ = get_embeddings(args, s_train)
        ge_dev, ge_test = ge_train, ge_train

    model = binary_classifer(args.classifier_layers_num, 
                                ge_train['s0'].shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = torch.nn.BCELoss()
    print('2. start to train model... ')
    # train
    model.train()
    train_losses = [999 for _ in range(args.patience)]
    for _ in range(10):
        loss_train = 0
        for s in range(len(s_train)):
            # get graph embedding
            graph_emb = ge_train['s'+str(s)]
            graph_emb = torch.FloatTensor(graph_emb).to(args.device)
            # get ground-truth graph
            src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_train[s])
            edge_labels = torch.FloatTensor(edge_labels).to(args.device)
            if len(src_idx) <= 1: continue
            optimizer.zero_grad()
            edge_pred = model(graph_emb[src_idx], graph_emb[dst_idx])
            edge_pred = torch.squeeze(edge_pred)
            loss = loss_fcn(edge_pred, edge_labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.data.item()
        loss_train = loss_train/len(s_train)
        print('   the training loss is: {:.4f}'.format(loss_train))
        # early stop
        if loss_train < max(train_losses):
            train_losses.remove(max(train_losses))
            train_losses.append(loss_train)
        else:
            break

    if data_split:
        # validation
        model.eval()
        loss_dev = 0
        for s in range(len(s_dev)):
            # get graph embedding
            graph_emb = ge_dev['s'+str(s)]
            graph_emb = torch.FloatTensor(graph_emb).to(args.device)
            # get ground-truth graph
            src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_dev[s])
            edge_labels = torch.FloatTensor(edge_labels).to(args.device)
            if len(src_idx) <= 1: continue
            edge_pred = model(graph_emb[src_idx], graph_emb[dst_idx])
            edge_pred = torch.squeeze(edge_pred)
            loss = loss_fcn(edge_pred, edge_labels)
            loss_dev += loss.data.item()
        loss_dev = loss_dev/len(s_dev)
        print('2. start to test model... | Train loss: {:.4f} | Val loss: {:.4f}'.format(
                                                                    loss_train, loss_dev))

    # test
    model.eval()
    auc, jaccard, uuas = [], [], []
    print('2. start to test graph embedding model... ')
    for s in range(len(s_test)):
        # get graph embedding
        graph_emb = ge_test['s'+str(s)]
        graph_emb = torch.FloatTensor(graph_emb).to(args.device)
        # get ground-truth graph
        src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_test[s])
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

    print(sum(auc)/len(auc), sum(jaccard)/len(jaccard))

    return


def test_bert_amr(args, model_name='bert'):
    # load data & graph embeddings
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    _, s_train_paths = get_embeddings(args, s_train, data_div='_train')
    _, s_dev_paths = get_embeddings(args, s_dev, data_div='_dev')
    _, s_test_paths = get_embeddings(args, s_test, data_div='_test')
    bert_train = np.load(s_train_paths[-1])
    bert_dev = np.load(s_dev_paths[-1])
    bert_test = np.load(s_test_paths[-1])
    if model_name != 'bert':
        bert_train, bert_dev, bert_test = load_split_emb(len(bert_train), len(bert_dev),
                                                        len(bert_test), model_name, 'amr')

    feat_dim = bert_train['s0'].shape[1]
    model = binary_classifer(args.classifier_layers_num, 
                                feat_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = torch.nn.BCELoss()

    print('2. start to train model...')
    # train
    model.train()
    train_losses = [999 for _ in range(args.patience)]
    for _ in range(10):
        loss_train = 0
        for s in range(len(bert_train)):
            # get graph embedding
            bert_emb = bert_train['s'+str(s)]
            bert_emb = torch.FloatTensor(bert_emb).to(args.device)
            # get ground-truth graph
            src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_train[s])
            edge_labels = torch.FloatTensor(edge_labels).to(args.device)
            if len(src_idx) <= 1: continue

            optimizer.zero_grad()
            edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
            edge_pred = torch.squeeze(edge_pred)
            loss = loss_fcn(edge_pred, edge_labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.data.item()
        loss_train = loss_train/len(bert_train)
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
    for s in range(len(bert_dev)):
        # get graph embedding
        bert_emb = bert_dev['s'+str(s)]
        bert_emb = torch.FloatTensor(bert_emb).to(args.device)
        # get ground-truth graph
        src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_dev[s])
        edge_labels = torch.FloatTensor(edge_labels).to(args.device)
        if len(src_idx) <= 1: continue

        edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
        edge_pred = torch.squeeze(edge_pred)
        loss = loss_fcn(edge_pred, edge_labels)
        loss_dev += loss.data.item()
    loss_dev = loss_dev/len(bert_dev)

    print('2. | Train loss: {:.4f} | Val loss: {:.4f}'.format(loss_train, loss_dev))
    # test
    model.eval()
    auc, jaccard, uuas = [], [], []
    print('2. start to test model...')
    for s in range(len(bert_test)):
        # get graph embedding
        bert_emb = bert_test['s'+str(s)]
        bert_emb = torch.FloatTensor(bert_emb).to(args.device)
        # get ground-truth graph
        src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_test[s])
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


def test_mi_amr(args):
    # load data & embeddings
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    amr_s = s_train + s_dev + s_test
    graph_emb, _ = get_embeddings(args, amr_s)

    print('2.2 start to test bert MI...')
    task_mi = []
    for n in tqdm(range(11)):
        noise_mi = []
        for r in range(args.repeat):
            mi_s = mine_probe(args, graph_emb, graph_emb, len(graph_emb), n/10)
            mi_s = sum(mi_s)/len(mi_s)
            noise_mi.append(mi_s)
        task_mi.append(noise_mi)

    print(task_mi)

    return


def mi_noise_amr(args):    
    # load data & embeddings
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    amr_s = s_train + s_dev + s_test
    graph_emb, bert_emb_paths = get_embeddings(args, amr_s)
    bert_emb = np.load(bert_emb_paths[-1])
    noisy_g = load_noisy_graphs(args)

    results = {}
    for k, v in noisy_g.items():
        if k not in results: results[k] = []
        for r in range(args.repeat):
            mi = mine_probe(args, graph_emb, bert_emb, len(graph_emb), 'noisy', v)
            results[k].append(sum(mi) / len(mi))
    print(results)
    return


def test_random_amr(args, corrupt=False):
    # load data & graph embeddings
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    _, s_train_paths = get_embeddings(args, s_train, data_div='_train')
    _, s_dev_paths = get_embeddings(args, s_dev, data_div='_dev')
    _, s_test_paths = get_embeddings(args, s_test, data_div='_test')
    bert_train = np.load(s_train_paths[-1])
    bert_dev = np.load(s_dev_paths[-1])
    bert_test = np.load(s_test_paths[-1])
    feat_dim = args.bert_hidden_num
    noisy_g = load_noisy_graphs(args)

    for noisy_tag, noisy_id in noisy_g.items():
        print('2. corrupt type: ', noisy_tag)
        model = binary_classifer(args.classifier_layers_num, feat_dim).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fcn = torch.nn.BCELoss()

        print('2. start to train model...')
        # train
        model.train()
        train_losses = [999 for _ in range(args.patience)]
        for _ in range(10):
            loss_train = 0
            for s in range(len(bert_train)):
                # get graph embedding
                bert_emb = bert_train['s'+str(s)]
                if corrupt:
                    rand_vec = np.random.randn(bert_emb.shape[0], bert_emb.shape[1])
                    bert_emb[noisy_id[s]] = rand_vec[noisy_id[s]]
                bert_emb = torch.FloatTensor(bert_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_train[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue

                optimizer.zero_grad()
                edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred)
                loss = loss_fcn(edge_pred, edge_labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.data.item()
            loss_train = loss_train/len(bert_train)
            print('   the training loss is: {:.4f}'.format(loss_train))
            # early stop
            if loss_train < max(train_losses):
                train_losses.remove(max(train_losses))
                train_losses.append(loss_train)
            else:
                break

        # test
        model.eval()
        label_ids = load_graph_labels(args)
        auc_dict = {}
        print('2. start to test model...')
        for k, v_l in label_ids.items():
            auc_tmp = []
            for s in range(len(bert_test)):
                # get graph embedding
                bert_emb = bert_test['s'+str(s)]
                bert_emb = torch.FloatTensor(bert_emb).to(args.device)
                # get ground-truth graph
                src_idx, dst_idx, edge_labels = get_edge_idx_amr(s_test[s])
                edge_labels = torch.FloatTensor(edge_labels).to(args.device)
                if len(src_idx) <= 1: continue
                edge_pred = model(bert_emb[src_idx], bert_emb[dst_idx])
                edge_pred = torch.squeeze(edge_pred).detach().cpu().numpy()
                edge_labels = edge_labels.detach().cpu().numpy()
                edge_mask = []
                for i in range(bert_emb.shape[0]):
                    for j in range(bert_emb.shape[0]):
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