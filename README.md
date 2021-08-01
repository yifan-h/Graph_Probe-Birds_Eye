## Bird's Eye: Probing for Linguistic Graph Structures with a Simple Information-Theoretic Approach

#### Authors: [Yifan Hou](https://yifan-h.github.io/), [Mrinmaya Sachan](http://www.mrinmaya.io/)

### Overview

NLP has a rich history of representing our prior understanding of language in the form of graphs. Recent work on analyzing contextualized text representations has focused on hand-designed probe models to understand how and to what extent do these representations encode a particular linguistic phenomenon. However, due to the inter-dependence of various phenomena and randomness of training probe models, detecting how these representations encode the rich information in these linguistic graphs remains a challenging problem. 

In this paper, we propose a new information-theoretic probe, Bird's Eye, which is a fairly simple probe method for detecting if and how these representations encode the information in these linguistic graphs. Instead of using classifier performance, our probe takes an information-theoretic view of probing and estimates the mutual information between the linguistic graph embedded in a continuous space and the contextualized word representations. Furthermore, we also propose an approach to use our probe to investigate localized linguistic information in the linguistic graphs using perturbation analysis. We call this probing setup Worm's Eye. Using these probes, we analyze BERT models on their ability to encode a syntactic and a semantic graph structure, and find that these models encode to some degree both syntactic as well as semantic information; albeit syntactic information to a greater extent. 

Please see the [paper](https://arxiv.org/abs/2105.02629) for more details. 

*Note:* If you make use of this code, or the probing model in your work, please cite the following paper *(formal bibliograph will be updated soon)*:

	@inproceedings{DBLP:conf/acl/HouS20,
	  author    = {Yifan Hou and Mrinmaya Sachan},
	  editor    = {Chengqing Zong and Fei Xia and Wenjie Li and Roberto Navigli},
	  title     = {Bird's Eye: Probing for Linguistic Graph Structures with a Simple Information-Theoretic Approach},
	  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
		       Linguistics and the 11th International Joint Conference on Natural
		       Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
		       Event, August 1-6, 2021},
	  pages     = {1844--1859},
	  publisher = {Association for Computational Linguistics},
	  year      = {2021},
	  url       = {https://doi.org/10.18653/v1/2021.acl-long.145},
	  doi       = {10.18653/v1/2021.acl-long.145},
	  timestamp = {Fri, 30 Jul 2021 16:41:20 +0200},
	  biburl    = {https://dblp.org/rec/conf/acl/HouS20.bib},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	}


### Requirements

[PyTorch](https://pytorch.org/get-started/locally/) (1.8.1) should be installed based on your system. For other libraries, recent versions of: tqdm, networkx, numpy, sklearn, gensim, transformers, penman, and conllu are required. You can install those required packages using the following command:

	$ pip install -r requirements.txt

### How to run

You can refer to comments and hyperparameters in *./src/main.py* for functions to reproduce all experiments in the paper. 

Simply run *./src/main.py*, then the *BERT-base-uncased model* is probed with *Penn Treebank* sample data in a layer-wise manner:

	$ python ./src/main.py

#### Input Format

You can refer to the sample data for the information about input format.

* syntax tree: Penn Treebank -- conllx format, parsed by CoNLL-U Parser (conllu).
* semantic graph: AMR Bank -- PENMAN notation format, parsed by PENMAN graph notation library (penman)

#### Usage of *Bird's Eye*

You can preprocess your data to the same formats as the sample data (syntax or semantics). Then run the code directly to get probing results. Difference models in [transformers](https://huggingface.co/transformers/) can be selected flexibly by changing several hyperparameters.

### Academic Paper

[**ACL 2021**] **Bird's Eye: Probing for Linguistic Graph Structures with a Simple Information-Theoretic Approach**, Yifan Hou, Mrinmaya Sachan
