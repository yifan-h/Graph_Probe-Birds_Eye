import os
import penman
import networkx as nx
import numpy as np
import random
import re
import penman
from tqdm import tqdm
from conllu import parse
# from torchnlp.datasets import penn_treebank_dataset


## sub-function descriptions
## please refer to main py file for function descriptions
'''
clean_string: get clean (no special character) string
load_data: load raw dataset
construct_graph: get graph structure (PTB dataset)
random_walks: run random walk for graph embedding
get_edge_idx: generate index list for link prediction (PTB) dataset
get_edge_idx_amr: generate index list for link prediction (AMR) dataset
uuas_score (deprecated): calculate UUAS score
get_graph_emb (deprecated): get global / local graph embedding
load_split_emb: load graph embedding in the split setting
load_noisy_trees: load corrupted PTB tree structures
load_tree_labels: load corrupted PTB tree labels
load_noisy_graphs: load corrupted AMR tree structures
load_graph_labels: load corrupted AMR tree labels
load_glove: load GloVe embedding
load_elmo (python version <= 3.7): load ELMo0 embedding
load_elmos (python version <= 3.7): load ELMo embedding
'''


def clean_string(s):
    return re.sub('[^A-Za-z0-9]+', '', s)

def load_data(data_name, data_type):
    if data_name == 'penn_treebank_dataset':
        # process for stanza dependency parsing
        sentences, tmp = [], []
        if data_type == 'train':
            with open ("./sample_data/ptb-gold/train.conllx", "r") as f:
                data = f.read()
        elif data_type == 'test':
            with open ("./sample_data/ptb-gold/test.conllx", "r") as f:
                data = f.read()
        if data_type == 'dev':
            with open ("./sample_data/ptb-gold/dev.conllx", "r") as f:
                data = f.read()
        parsed = parse(data)
        for s in parsed:
            tmp = []
            for t in s:
                tmp_txt = clean_string(t.get('form'))
                if len(tmp_txt): tmp.append(tmp_txt)
            if len(tmp): sentences.append(tmp)
        return sentences, parsed

    elif data_name == 'amr_dataset':
        # process for stanza dependency parsing
        sentences, tmp = [], []
        if data_type == 'train':
            with open ("./sample_data/amr-split/amr-training.txt", "r") as f:
                data = f.read()
        elif data_type == 'test':
            with open ("./sample_data/amr-split/amr-test.txt", "r") as f:
                data = f.read()
        if data_type == 'dev':
            with open ("./sample_data/amr-split/amr-dev.txt", "r") as f:
                data = f.read()
        data = data.split('\n\n')
        for s in data:
            if s[:4] != '# ::':
                data.remove(s)
        return data

    else:
        print('Error data name!')
        return

def construct_graph(parsed):
    '''
    # stanza doc type to python dictionary
    nlp = stanza.Pipeline(lang='en', 
                            processors='tokenize, pos, lemma, depparse',
                            tokenize_pretokenized=True)
    '''
    word_dict = {'global_sentence_root': 0}
    word_id = 1
    global_graph = nx.Graph()
    doc_id, sen_id = [], []
    print('1. start to tokenize and construct global graph...')
    for s in parsed:
        # construct word dictionary
        for t in s:
            if len(clean_string(t.get('form'))) == 0: continue
            if clean_string(t.get('form')) not in word_dict:
                word_dict[clean_string(t.get('form'))] = word_id
                word_id += 1
        # construct doc_id and global graph
        tmp1, tmp2 = [], []
        for t in s:
            tail_txt = clean_string(t.get('form'))
            if len(tail_txt) == 0: continue
            tail_id = word_dict.get(tail_txt)
            head_idx = t.get('head')
            if head_idx == 0:
                head_id = 0
            else:
                head_txt = clean_string(s[head_idx-1].get('form'))
                if len(head_txt) == 0: 
                    head_id = tail_id
                else:
                    head_id = word_dict.get(head_txt)
            global_graph.add_edge(head_id, tail_id)
            tmp1.append((t.get('head'), t.get('id')))
            tmp2.append((head_id, tail_id))
        if len(tmp1): sen_id.append(tmp1)
        if len(tmp2): doc_id.append(tmp2)
    '''
    print('    global graph (V,E) numbers: ({}, {})'.format(
                                            global_graph.number_of_nodes(), 
                                            global_graph.number_of_edges()))
    '''
    return doc_id, sen_id, global_graph


def random_walks(G, num_walks=100, walk_len=10, string_nid=False):
    paths = []
    # add self loop
    for nid in G.nodes(): G.add_edge(nid, nid)
    if not string_nid:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [str(nid)]
                for j in range(walk_len):
                    neighbors = [str(n) for n in G.neighbors(int(tmp_path[-1]))]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)
    else:
        for nid in G.nodes():
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [nid]
                for j in range(walk_len):
                    neighbors = [n for n in G.neighbors(tmp_path[-1])]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)

    return paths


def get_edge_idx(edge_list):
    batch_ids = []
    for (_, j) in edge_list:
        batch_ids.append(j)
    batch_ids = list(set(batch_ids))
    tmp_dict = {}
    for i in range(len(batch_ids)):
        tmp_dict[batch_ids[i]] = i

    sort_edge_list = []
    for (s, d) in edge_list:
        sort_edge_list.append((tmp_dict.get(s), tmp_dict.get(d)))

    edge_space = []
    for i in range(len(batch_ids)):
        for j in range(len(batch_ids)):
            edge_space.append((i, j))
    # random.shuffle(edge_space)
    src_idx = [i for (i, j) in edge_space]
    dst_idx = [j for (i, j) in edge_space]

    edge_labels = []
    for e in edge_space:
        if e in sort_edge_list:
            edge_labels.append(1)
        elif (e[-1], e[0]) in sort_edge_list:
            edge_labels.append(1)
        else:
            edge_labels.append(0)

    return src_idx, dst_idx, np.array(edge_labels)


def get_edge_idx_amr(s):
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
    # graph construction
    g = nx.Graph()
    for v in penman_g.variables(): g.add_node(v)
    for e in penman_g.edges(): g.add_edge(e.source, e.target)

    edge_space = []
    for i in range(len(var)):
        for j in range(len(var)):
            edge_space.append((i, j))
    # random.shuffle(edge_space)
    src_idx = [i for (i, j) in edge_space]
    dst_idx = [j for (i, j) in edge_space]

    edge_labels = []
    for e in edge_space:
        if (var[e[0]], var[e[1]]) in g.edges():
            edge_labels.append(1)
        elif (var[e[1]], var[e[0]]) in g.edges():
            edge_labels.append(1)
        else:
            edge_labels.append(0)

    return src_idx, dst_idx, np.array(edge_labels)


def uuas_score(src_idx, dst_idx, edge_labels, edge_pred):
    g_label = nx.Graph()
    g_pred = nx.Graph()
    tmp_num = max(src_idx) + 1
    for src in src_idx:
        for dst in dst_idx:
            # add edge in label (ground-truth) graph
            if edge_labels[src*tmp_num+dst] == 1:
                g_label.add_edge(src, dst)
            # add edge in predicted graph
            if (src, dst) not in g_pred.edges():
                weight_1 = edge_pred[src*tmp_num+dst]
                weight_2 = edge_pred[dst*tmp_num+src]
                g_pred.add_edge(src, dst, weight=weight_1+weight_2)
    g_mst = nx.minimum_spanning_tree(g_pred)
    total_num = g_mst.number_of_edges()
    uuas_num = 0
    for e in g_mst.edges():
        if e in g_label:
            uuas_num += 1
    return uuas_num / total_num


def get_graph_emb(graph_emb, task_name):
    new_graph_emb = {}
    if task_name == 'local':
        for s in range(len(graph_emb)):
            new_graph_emb['s'+str(s)] = graph_emb['s'+str(s)][:,640:]
    elif task_name == 'global':
        for s in range(len(graph_emb)):
            new_graph_emb['s'+str(s)] = graph_emb['s'+str(s)][:,:640]
    else:
        new_graph_emb = graph_emb

    return new_graph_emb


def load_split_emb(train_len, dev_len, test_len, model_name, task='ptb'):
    if task == 'ptb':
        if model_name == 'elmo':
            emb_path = './tmp/elmo_ptb_bert.npz'
        else:
            emb_path = './tmp/glove_ptb_bert.npz'
    else:
        if model_name == 'elmo':
            emb_path = './tmp/elmo_amr_bert.npz'
        else:
            emb_path = './tmp/glove_amr_bert.npz'
    all_emb = np.load(emb_path)
    train_emb, dev_emb, test_emb = {}, {}, {}
    if len(all_emb) != train_len + dev_len + test_len:
        print('Error of length !', len(all_emb), train_len, dev_len, test_len)
    tmp_count = 0
    for i in range(len(all_emb)):
        curr_key = 's' + str(i)
        if i < train_len:
            train_key = curr_key
            train_emb[train_key] = all_emb[curr_key]
        elif i < train_len + dev_len:
            dev_key = 's' + str(i-train_len)
            dev_emb[dev_key] =  all_emb[curr_key]
        else:
            test_key = 's' + str(i-train_len-dev_len)
            test_emb[test_key] =  all_emb[curr_key]

    return train_emb, dev_emb, test_emb


def load_noisy_trees(args, pos=False, data_split=False):
    print('2. start to calculate noisy id...')

    if data_split:
        sentences, parsed = load_data('penn_treebank_dataset', 'train')
    else:
        s_train, p_train = load_data('penn_treebank_dataset', 'train')
        s_dev, p_dev = load_data('penn_treebank_dataset', 'dev')
        s_test, p_test = load_data('penn_treebank_dataset', 'test')
        sentences = s_train + s_dev + s_test
        parsed = p_train + p_dev + p_test
    # get noisy node ids
    edge_labels = {}
    if pos:  # pos noisy
        for s in parsed:
            for w in s:
                if w.get('upos') not in edge_labels:
                    edge_labels[w.get('upos')] = 1
                else:
                    edge_labels[w.get('upos')] += 1
        noisy_id = {}
        k_list = ['IN', 'NNP', 'DT', 'JJ', 'NNS']
        for k, v in edge_labels.items():
            if k in k_list:
                noisy_id[k] = []
    else:  # edge noisy
        for s in parsed:
            for w in s:
                if w.get('deprel') not in edge_labels:
                    edge_labels[w.get('deprel')] = 1
                else:
                    edge_labels[w.get('deprel')] += 1
        noisy_id = {}
        k_list = ['prep', 'det', 'nn', 'pobj', 'nsubj']
        for k in k_list:
            noisy_id[k] = []

    for k, _ in noisy_id.items():
        sen_id = []
        for s in parsed:
            tmp = set()
            tmp_embed = []
            for w in s:
                if len(clean_string(w.get('form'))) == 0: 
                    continue
                tmp_embed.append((w.get('head'), w.get('id')))
                if pos:  # pos noisy
                    if w.get('upos') == k: tmp.add(w.get('id'))
                else:  # edge noisy
                    if w.get('deprel') == k:
                        tmp.add(w.get('id'))
                        tmp.add(w.get('head'))
            tmp_set = set()
            for _, dst in tmp_embed:
                tmp_set.add(dst)
            tmp_dict = {}
            count = 0
            for idx in tmp_set:
                if idx > 0:
                    tmp_dict[idx] = count
                    count += 1
            tmp_new = []
            for idx in tmp:
                if idx in tmp_dict:
                    tmp_new.append(tmp_dict[idx])
            tmp_new.sort()
            if len(tmp_embed): sen_id.append(tmp_new)
        noisy_id[k] = sen_id

    # leverage noisy node number
    drop_ratio = {}
    for k, v in noisy_id.items():
        drop_ratio[k] = 0
        for s in v:
            drop_ratio[k] += len(s)
    min_count = min([v for k, v in drop_ratio.items()])
    for k, v in drop_ratio.items():
        drop_ratio[k] = min_count / drop_ratio[k]

    noisy_id_new = {}
    for k, _ in noisy_id.items():
        noisy_id_new[k] = []
    for k, v in noisy_id.items():
        tmp_s = []
        for s in v:
            tmp_w = []
            for w in s:
                if random.random() < drop_ratio[k]:
                    tmp_w.append(w)
            tmp_s.append(tmp_w)
        noisy_id_new[k] = tmp_s

    '''
    test_dict = {}
    for k, v in noisy_id_new.items():
        count = 0
        for s in v:
            count += len(s)
        test_dict[k] = count
    print(test_dict)
    '''

    return noisy_id_new


def load_tree_labels(args, pos=False):
    print('2. start to calculate noisy id...')
    sentences, parsed = load_data('penn_treebank_dataset', 'test')
    # get noisy node ids
    edge_labels = {}
    if pos:  # pos noisy
        for s in parsed:
            for w in s:
                if w.get('upos') not in edge_labels:
                    edge_labels[w.get('upos')] = 1
                else:
                    edge_labels[w.get('upos')] += 1
        noisy_id = {}
        for k, v in edge_labels.items():
            noisy_id[k] = []
    else:  # edge noisy
        for s in parsed:
            for w in s:
                if w.get('deprel') not in edge_labels:
                    edge_labels[w.get('deprel')] = 1
                else:
                    edge_labels[w.get('deprel')] += 1
        noisy_id = {}
        for k, v in edge_labels.items():
            noisy_id[k] = []

    for k, _ in noisy_id.items():
        sen_id = []
        for s in parsed:
            tmp = set()
            tmp_embed = []
            for w in s:
                if len(clean_string(w.get('form'))) == 0: 
                    continue
                tmp_embed.append((w.get('head'), w.get('id')))
                if pos:  # pos noisy
                    if w.get('upos') == k: tmp.add(w.get('id'))
                else:  # edge noisy
                    if w.get('deprel') == k:
                        tmp.add(w.get('id'))
                        tmp.add(w.get('head'))
            tmp_set = set()
            for _, dst in tmp_embed:
                tmp_set.add(dst)
            tmp_dict = {}
            count = 0
            for idx in tmp_set:
                if idx > 0:
                    tmp_dict[idx] = count
                    count += 1
            tmp_new = []
            for idx in tmp:
                if idx in tmp_dict:
                    tmp_new.append(tmp_dict[idx])
            tmp_new.sort()
            if len(tmp_embed): sen_id.append(tmp_new)
        noisy_id[k] = sen_id

    '''
    test_dict = {}
    for k, v in noisy_id.items():
        count = 0
        for s in v:
            count += len(s)
        test_dict[k] = count
    print(test_dict, edge_labels)
    '''

    return noisy_id


def load_noisy_graphs(args):
    print('2. start to calculate noisy id...')
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    amr_s = s_train + s_dev + s_test

    label_arg = [':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4',\
                 ':ARG5', ':ARG6', ':ARG7', ':ARG8', ':ARG9']
    label_op = [':op1', ':op2', ':op3', ':op4', ':op5', ':op6', ':op7',\
                ':op8', ':op9', ':op10', ':op11', ':op12', ':op13', ':op14',\
                 ':op15', ':op16', ':op17', ':op18', ':op19']
    label_general = [':accompanier', ':age', ':beneficiary', ':concession',\
                     ':condition', ':consist', ':degree', ':destination',\
                      ':direction', ':domain', ':duration', ':example', \
                      ':extent', ':frequency', ':instrument', ':location',\
                      ':manner', ':medium', ':mod', ':name', ':part', ':path',\
                      ':polarity', ':poss', ':purpose', ':source', ':subevent',\
                      ':subset', ':time', ':topic', ':value', ':ord', ':range']
    labels_dict = {'arg': label_arg, 'op': label_op, 'general': label_general}

    # get edge label
    noisy_id = {}
    for k, _ in labels_dict.items():
        noisy_id[k] = []
    for k_label, _ in noisy_id.items():
        sen_id = []
        for s in amr_s:
            penman_g = penman.decode(s)
            var = []  # k=word id; v=variable
            for k, v in penman_g.epidata.items():
                if k[1] == ':instance':
                    if len(v):
                        if type(v[0]) == penman.surface.Alignment:
                            var.append(k[0])
            tmp_idx, tmp_set = [], set()
            for e in penman_g.edges():
                if e.role in labels_dict[k_label]:
                    tmp_set.add(e.source)
                    tmp_set.add(e.target)
            for n in tmp_set:
                if n in var:
                    tmp_idx.append(var.index(n))
            tmp_idx.sort()
            sen_id.append(tmp_idx)
        noisy_id[k_label] = sen_id

    # leverage edge number
    drop_ratio = {}
    for k, v in noisy_id.items():
        drop_ratio[k] = 0
        for s in v:
            drop_ratio[k] += len(s)
    min_count = min([v for k, v in drop_ratio.items()])
    for k, v in drop_ratio.items():
        drop_ratio[k] = min_count / drop_ratio[k]

    noisy_id_new = {}
    for k, _ in noisy_id.items():
        noisy_id_new[k] = []
    for k, v in noisy_id.items():
        tmp_s = []
        for s in v:
            tmp_w = []
            for w in s:
                if random.random() < drop_ratio[k]:
                    tmp_w.append(w)
            tmp_s.append(tmp_w)
        noisy_id_new[k] = tmp_s

    return noisy_id_new


def load_graph_labels(args):
    print('2. start to calculate noisy id...')
    s_train = load_data('amr_dataset', 'train')
    s_dev = load_data('amr_dataset', 'dev')
    s_test = load_data('amr_dataset', 'test')
    amr_s = s_train + s_dev + s_test

    label_arg = [':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4',\
                 ':ARG5', ':ARG6', ':ARG7', ':ARG8', ':ARG9']
    label_op = [':op1', ':op2', ':op3', ':op4', ':op5', ':op6', ':op7',\
                ':op8', ':op9', ':op10', ':op11', ':op12', ':op13', ':op14',\
                 ':op15', ':op16', ':op17', ':op18', ':op19']
    label_general = [':accompanier', ':age', ':beneficiary', ':concession',\
                     ':condition', ':consist', ':degree', ':destination',\
                      ':direction', ':domain', ':duration', ':example', \
                      ':extent', ':frequency', ':instrument', ':location',\
                      ':manner', ':medium', ':mod', ':name', ':part', ':path',\
                      ':polarity', ':poss', ':purpose', ':source', ':subevent',\
                      ':subset', ':time', ':topic', ':value', ':ord', ':range']
    label_quantities = [':quant', ':scale', ':unit']
    label_date = [':dayperiod', ':calendar', ':season', ':timezone', ':weekday']
    labels_dict = {'arg': label_arg, 'op': label_op, 'general': label_general,
                    'quantities': label_quantities, 'date': label_date}

    # get edge label
    noisy_id = {}
    for k, _ in labels_dict.items():
        noisy_id[k] = []
    for k_label, _ in noisy_id.items():
        sen_id = []
        for s in amr_s:
            penman_g = penman.decode(s)
            var = []  # k=word id; v=variable
            for k, v in penman_g.epidata.items():
                if k[1] == ':instance':
                    if len(v):
                        if type(v[0]) == penman.surface.Alignment:
                            var.append(k[0])
            tmp_idx, tmp_set = [], set()
            for e in penman_g.edges():
                if e.role in labels_dict[k_label]:
                    tmp_set.add(e.source)
                    tmp_set.add(e.target)
            for n in tmp_set:
                if n in var:
                    tmp_idx.append(var.index(n))
            tmp_idx.sort()
            sen_id.append(tmp_idx)
        noisy_id[k_label] = sen_id

    
    test_dict = {}
    for k, v in noisy_id.items():
        count = 0
        for s in v:
            count += len(s)
        test_dict[k] = count
    print(test_dict)

    return noisy_id

def load_glove(args, sentences, data_div='', dataset='ptb'):
    data_path = './tmp/glove_'+args.task+data_div+'.npz'
    if os.path.exists(data_path):
        return np.load(data_path)

    savez_dict = {}        
    embeddings_dict = {}
    with open('./tmp/glove/glove.42B.300d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    if dataset == 'ptb':
        for s in range(len(sentences)):
            word_emb = []
            for w in sentences[s]:
                if w.lower() in embeddings_dict:
                    word_emb.append(np.expand_dims(embeddings_dict[w.lower()], axis=0))
                else:
                    word_emb.append(np.expand_dims(embeddings_dict[','], axis=0))
            savez_dict['s'+str(s)] = np.concatenate(word_emb)
    else:
        for s in range(len(sentences)):
            word_emb = []
            # parse
            penman_g = penman.decode(sentences[s])
            sen = penman_g.metadata.get('tok').split(' ')
            wid = []
            var = []  # k=word id; v=variable
            for k, v in penman_g.epidata.items():
                if k[1] == ':instance':
                    if len(v):
                        if type(v[0]) == penman.surface.Alignment:
                            wid.append(v[0].indices[0])
                            var.append(k[0])
            c_s = []
            for w in sen:
                c_w = clean_string(w)
                if len(c_w) == 0: c_w = ','
                c_s.append(c_w)
            for w in c_s:
                if w.lower() in embeddings_dict:
                    word_emb.append(np.expand_dims(embeddings_dict[w.lower()], axis=0))
                else:
                    word_emb.append(np.expand_dims(embeddings_dict[','], axis=0))
            if len(wid) == 0: wid = [0]
            savez_dict['s'+str(s)] = np.concatenate([word_emb[i] for i in wid])
    np.savez('./tmp/glove_'+args.task+data_div+'.npz', **savez_dict)

    return np.load(data_path)


def load_elmo(args, sentences, data_div='', dataset='ptb'):
    data_path = './tmp/elmo_'+args.task+data_div+'.npz'
    if os.path.exists(data_path):
        return np.load(data_path)
    else:
        import nlu
        elmo_model = nlu.load('elmo')

    savez_dict = {}
    if dataset == 'ptb':
        word_set = set()
        for s in range(len(sentences)):
            for w in sentences[s]:
                if w.lower() not in word_set: word_set.add(w.lower())
        embeddings_dict = {}
        print('1. start to calculate ELMo embeddings...')
        for w in tqdm(word_set):
            output = elmo_model.predict(w)
            embeddings_dict[w] = output.to_numpy()[0,1]
        for s in range(len(sentences)):
            word_emb = []
            for w in sentences[s]:
                if w.lower() in embeddings_dict:
                    word_emb.append(np.expand_dims(embeddings_dict[w.lower()], axis=0))
                else:
                    word_emb.append(np.expand_dims(embeddings_dict[','], axis=0))
            savez_dict['s'+str(s)] = np.concatenate(word_emb)
    else:
        word_set = set()
        for s in range(len(sentences)):
            penman_g = penman.decode(sentences[s])
            sen = penman_g.metadata.get('tok').split(' ')
            for w in sen:
                if w.lower() not in word_set: word_set.add(w.lower())
        embeddings_dict = {}
        print('1. start to calculate ELMo embeddings...')
        for w in tqdm(word_set):
            output = elmo_model.predict(w)
            embeddings_dict[w] = output.to_numpy()[0,1]
        for s in range(len(sentences)):
            word_emb = []
            # parse
            penman_g = penman.decode(sentences[s])
            sen = penman_g.metadata.get('tok').split(' ')
            wid = []
            var = []  # k=word id; v=variable
            for k, v in penman_g.epidata.items():
                if k[1] == ':instance':
                    if len(v):
                        if type(v[0]) == penman.surface.Alignment:
                            wid.append(v[0].indices[0])
                            var.append(k[0])
            c_s = []
            for w in sen:
                c_w = clean_string(w)
                if len(c_w) == 0: c_w = ','
                c_s.append(c_w)
            for w in c_s:
                if w.lower() in embeddings_dict:
                    word_emb.append(np.expand_dims(embeddings_dict[w.lower()], axis=0))
                else:
                    word_emb.append(np.expand_dims(embeddings_dict[','], axis=0))
            if len(wid) == 0: wid = [0]
            savez_dict['s'+str(s)] = np.concatenate([word_emb[i] for i in wid])
    np.savez(data_path, **savez_dict)

    return np.load(data_path)


def load_elmos(args, sentences, data_div='', dataset='ptb'):
    data_path = './tmp/elmo0_'+args.task+data_div+'.npz'
    data_paths = []
    for i in range(3):
        data_paths.append('./tmp/elmo'+str(i)+'_'+args.task+data_div+'.npz')

    if os.path.exists(data_path):
        return data_paths
    else:
        import nlu
        pipe = nlu.load('elmo')

    layers_name = ['lstm_outputs1', 'lstm_outputs2', 'word_emb']
    savez_dict = {}
    if dataset == 'ptb':
        print('1. start to calculate ELMo embeddings...')
        for n in range(len(layers_name)):
            for s in tqdm(range(len(sentences))):
                pipe['elmo'].setPoolingLayer(layers_name[n])
                outputs = pipe.predict(sentences[s])
                output = outputs.to_numpy()[:,1:]
                if len(sentences[s]) <= 1:
                    savez_dict['s'+str(s)] = output[0]
                    continue
                output_vectors = np.empty((output.shape[0], output[1,0].shape[0]))
                for i in range(output_vectors.shape[0]):
                    output_vectors[i,:] = output[i,0]
                if len(sentences[s]) != output_vectors.shape[0]:
                    print('Error! failed to get whole word', s, sentences[s], output_vectors)
                savez_dict['s'+str(s)] = output_vectors
            np.savez(data_paths[n], **savez_dict)
    else:
        print('1. start to calculate ELMo embeddings...')
        for n in range(len(layers_name)):
            for s in tqdm(range(len(sentences))):
                # parse
                penman_g = penman.decode(sentences[s])
                sen = penman_g.metadata.get('tok').split(' ')
                wid = []
                var = []  # k=word id; v=variable
                for k, v in penman_g.epidata.items():
                    if k[1] == ':instance':
                        if len(v):
                            if type(v[0]) == penman.surface.Alignment:
                                wid.append(v[0].indices[0])
                                var.append(k[0])
                c_s = []
                for w in sen:
                    c_w = clean_string(w)
                    if len(c_w) == 0: c_w = ','
                    c_s.append(c_w)

                outputs = pipe.predict(c_s)
                output = outputs.to_numpy()[:,1:]
                if len(c_s) <= 1:
                    savez_dict['s'+str(s)] = output[0]
                    continue
                output_vectors = np.empty((output.shape[0], output[1,0].shape[0]))
                for i in range(output_vectors.shape[0]):
                    output_vectors[i,:] = output[i,0]
                if len(c_s) != output_vectors.shape[0]:
                    print('Error! failed to get whole word', s, c_s, outputs.to_numpy()[:,1:])
                if len(wid) == 0: wid = [0]
                savez_dict['s'+str(s)] = output_vectors[wid]
            np.savez(data_paths[n], **savez_dict)

    return data_paths


