'''
Training algorithms for ERGMs in supervised WordNet setting
Release note: only keeping macro mode here
'''
import enum
import dynet as dy
import numpy as np
import copy 
from tqdm import tqdm
import scipy
from bisect import bisect
from random import shuffle

from io_utils import timeprint
from multigraph_utils import targets
from math_utils import softmaxify
from consts import SYMMETRIC_RELATIONS

__author__ = "Yuval Pinter, 2018"

MARGIN = 1.0

# For easy use of switching between experiments change the below string to:
# "negative" for negative sampling experiment
# "importance" for importance sampling experiment
# "hierarchical" for hierarchical softmax experiment
samplingType = "hierarchical"

def macro_node_iteration(opts, multi_graph, assoc_cache,
                         trainer, log_file, synsets, rel, src_i, use_assoc):
    """
    One node-relation iteration in a macro-level pass over the multigraph
    :param opts: parameter dictionary from calling model
    :param multi_graph: trained data structure
    :param assoc_cache: cache for association model
    :param trainer: dynet training module
    :param log_file: log file location
    :param synsets: synset name dictionary for reporting
    :param rel: relation type for iteration
    :param src_i: source node ID for iteration
    :param use_assoc: use association score model
    :return: state of cache after iteration
    """
            
    g = multi_graph.graphs[rel]
    N = multi_graph.vocab_size
    
    # set up iteration
    if opts.debug:
        dy.renew_cg(immediate_compute = True, check_validity = True)
    else:
        dy.renew_cg()

    # keep existing score for all deltas
    multi_graph.rescore()
    score_with_all = multi_graph.dy_score

    # report progress
    perform_verbosity_steps = opts.v > 1 or (opts.v > 0 and src_i > 0 and src_i % 10 == 0)
    if perform_verbosity_steps:
        timeprint('iterating on node {}, {}, current score = {:.6f}'\
                  .format(src_i, synsets[src_i], score_with_all.scalar_value()))


    # true targets scoring

    true_targets = targets(g, src_i)

    if len(true_targets) == 0:
        # don't perform negative sampling without true targets
        return assoc_cache

    # compute log likelihood on targets
    # each used to be multiplied by multi_graph.a_scale
    target_assoc_scores = {t: multi_graph.word_assoc_score(src_i, t, rel) for t in true_targets}
    if opts.no_assoc_bp:
        # turn into values to detach from computation graph
        target_assoc_scores = {t: t_as.value() for t, t_as in list(target_assoc_scores.items())}
    target_scores = {t: score_with_all + t_as for t, t_as in list(target_assoc_scores.items())}


    # false targets scoring - importance sampling

    ### NEGATIVE SAMPLING
    #----------------------------------------------------------------------
    if samplingType == "negative":
        # compute softmax over all false targets based on bilinear scores
        if use_assoc:
            assoc_sc = multi_graph.score_from_source_cache(assoc_cache, src_i)
            neg_assocs = {j: s for j, s in enumerate(assoc_sc) if j not in true_targets and j != src_i}
        else:
            neg_assocs = {j: 1.0 for j in range(N) if j not in true_targets and j != src_i}
        neg_probs = softmaxify(neg_assocs)

        # TODO see if searchsorted can work here too (issue in dynet repo)
        neg_samples = {t: [dy.np.random.choice(range(len(neg_assocs)), p=neg_probs)\
                        for _ in range(opts.neg_samp)]\
                    for t in true_targets} # sample without return?
    #----------------------------------------------------------------------

    ### IMPORTANCE SAMPLING
    # note that the final sample values are called neg_samples for convenienve/for it to be
    # compatible with the rest of the code, however the samples are not negative samples
    #----------------------------------------------------------------------
    elif samplingType == "importance":
        # compute softmax over all false targets based on bilinear scores
        if use_assoc:
            assoc_sc = multi_graph.score_from_source_cache(assoc_cache, src_i)
            neg_assocs = {j: s for j, s in enumerate(assoc_sc) if j not in true_targets and j != src_i}
        else:
            neg_assocs = {j: 1.0 for j in range(N) if j not in true_targets and j != src_i}
        neg_probs = softmaxify(neg_assocs)
        
        # Set hyper parameter n to approximate a distribution for importance sampling
        n = 100000
        samples = []
        for i in range(n):
            x_samp = dy.np.random.choice(range(len(neg_assocs)), p=neg_probs)
            samples += [x_samp]
        # see report for citation to mikolov where gaussian is used
        new_mean = np.mean(samples)
        new_sd = np.std(samples)
        def importanceFind(x_i):
            qVal = scipy.stats.norm.pdf(x_i,new_mean,new_sd)
            pVal = neg_probs[bisect(range(len(neg_assocs)),x_i)]
            return x_i * (pVal/qVal)
        neg_samples = {t: [importanceFind(dy.np.random.normal(new_mean, new_sd))\
                        for _ in range(opts.neg_samp)]\
                    for t in true_targets}
    #----------------------------------------------------------------------

    ### HIERARCHICAL SOFTMAX
    # note that the final sample values are called neg_samples for convenienve/for it to be
    # compatible with the rest of the code, however the samples are not negative samples
    #----------------------------------------------------------------------
    elif samplingType == "hierarchical":
        # compute hierarchical softmax 
        # note that we can do this over all false targets based on bilinear scores similar to before
        if use_assoc:
            assoc_sc = multi_graph.score_from_source_cache(assoc_cache, src_i)
            neg_assocs = {j: s for j, s in enumerate(assoc_sc) if j not in true_targets and j != src_i}
        else:
            neg_assocs = {j: 1.0 for j in range(N) if j not in true_targets and j != src_i}
        # useful binary tree functions for hierarchical softmax
        def hiersoftmaxify(neg_assocs,treeCheck):
            # Use this to create the binary tree for given predictor sample
            # if tree does not already exist
            def create_tree(neg_assocs):
                shuffle(neg_assocs)
                temp_treeHolder = []
                if (len(neg_assocs) > 2):
                    for idx in range(0, len(neg_assocs), 2):
                        if len(neg_assocs) - idx - 1 > 0:
                            temp_treeHolder.append([neg_assocs[idx], neg_assocs[idx+1]])
                        else:
                            temp_treeHolder.append(neg_assocs[idx])
                else:
                    print("Warning: Inproper size for Hierarchical Softmax")
                return temp_treeHolder
            if treeCheck:
                newTree = create_tree(neg_assocs)
            else:
                return softmaxify(neg_assocs)
            # create function to recursively count number of nodes in binary tree
            def count_nodes(binaryTree):
                totalCount = 0
                for subTree in binaryTree:
                    totalCount += 1
                    totalCount += count_nodes(subTree)
                return totalCount
            # create function to easily traverse binary tree by getting all subtrees
            def all_subTrees(binaryTree):
                newSub = []
                for subTree in binaryTree:
                    newSub += subTree
                return newSub
            # create function to calculate path between root word and all leaf words
            def calculate_path(binaryTree):
                if(len(binaryTree) == 1):
                    return ([binaryTree],1)
                for (counter,subTree) in enumerate(binaryTree):
                    oldPath,oldCounter = calculate_path(subTree)
                    return (oldPath + [subTree],oldCounter + counter)
            totalPath, totalValue = calculate_path(newTree)
            totalNodes = count_nodes(newTree)
            subtrees = all_subTrees(newTree)
            subTreeCounter = 0
            for newSubTree in subtrees:
                subTreeCounter += len(newSubTree)
            denominator = np.exp(subTreeCounter) + np.exp(totalNodes)
            return np.exp(totalValue)/denominator
        neg_probs = hiersoftmaxify(neg_assocs, False)
        neg_samples = {t: [dy.np.random.choice(range(len(neg_assocs)), p=neg_probs)\
                        for _ in range(opts.neg_samp)]\
                    for t in true_targets}
    #----------------------------------------------------------------------
    else:
        print("Invalid type of sampling experiment: please read documentation at top of file!")

    # for reporting
    if perform_verbosity_steps:
        neg_sample_idcs = []
        for negs in list(neg_samples.values()):
            neg_sample_idcs.extend([list(neg_assocs.keys())[j] for j in negs])
    
    # compute neg log likelihood on negative samples
    margins = []
    for t in true_targets:
        t_score = target_scores[t]
        negs = [list(neg_assocs.keys())[j] for j in neg_samples[t]]
        # each used to be multiplied by multi_graph.a_scale
        neg_assoc_scores = [multi_graph.word_assoc_score(src_i, j, rel) for j in negs]
        if opts.no_assoc_bp:
            # turn into values to detach from computation graph
            neg_assoc_scores = [s.value() for s in neg_assoc_scores]
        # prepare graph for pass
        multi_graph.remove_edge(src_i, t, rel, permanent=True)
        t_cache = (copy.deepcopy(multi_graph.cache), copy.deepcopy(multi_graph.feature_vals))
        for jas, j, origj in zip(neg_assoc_scores, negs, neg_samples[t]):
            q_norm = 1.0 / neg_probs[origj]
            g_score = multi_graph.add_edge(src_i, j, rel, caches=t_cache, report_feat_diff=opts.v > 1)
            margins.append(dy.rectify(g_score + jas + MARGIN - t_score) * q_norm)
            log_file.write('{}\t{}\t{}\t{}\t{:.2e}\t{:.2e}\t{:.2e}\n'\
                         .format(rel, src_i, t, j, t_score.scalar_value(),
                                 g_score.scalar_value(), jas if type(jas) == float else jas.value()))
        # revert graph for next margin iteration
        multi_graph.add_edge(src_i, t, rel, permanent=True)
    node_loss = dy.esum(margins)
    
    # backprop and recompute score
    if perform_verbosity_steps:
        timeprint('selected nodes {} with probabilities {}'\
                  .format(neg_sample_idcs, ['{:.2e}'.format(neg_probs[n]) for n in neg_samples]))
        timeprint('overall {} loss = {:.6f}'\
                  .format('margin' if opts.margin_loss else 'neg log', node_loss.scalar_value()))

        # record state for later reporting
        pre_weights = multi_graph.ergm_weights.as_array()
        pre_assoc = multi_graph.word_assoc_weights[rel].as_array()

    # add regularization
    if multi_graph.regularize > 0.0:
        node_loss += multi_graph.regularize * dy.l2_norm(dy.parameter(multi_graph.ergm_weights))

    # perform actual learning
    node_loss.backward()
    trainer.update()

    if perform_verbosity_steps:
        post_weights = multi_graph.ergm_weights.as_array()
        post_assoc = multi_graph.word_assoc_weights[rel].as_array()
        w_diff = post_weights - pre_weights
        a_diff = post_assoc - pre_assoc
        timeprint('changed weights = {}'.format(len(w_diff.nonzero()[0])))
        timeprint('changed pre_assoc = {}, norm {}'\
                  .format(len(a_diff.nonzero()[0]), np.linalg.norm(a_diff)))

    # recompute assoc_cache columns for src_i and participating targets
    if use_assoc and not opts.no_assoc_bp:
        # TODO normalize embeddings?
        return multi_graph.source_ranker_cache(rel)
    return assoc_cache


def macro_loops(opts, ep_idx, multi_graph, trainer, log_file, synsets, use_assoc=True):
    """
    Passing over graph node by node, relation by relation.
    Single update returned, based on importance sampling from entire graph.
    :param opts: parameter dictionary from calling model
    :param ep_idx: epoch index
    :param multi_graph: trained data structure
    :param trainer: dynet training module
    :param log_file: log file location
    :param synsets: synset name dictionary for reporting
    :param use_assoc: include association component in scores
    :return: node-iteration scores
    """
    iteration_scores = []
    iteration_scores.append(multi_graph.score)

    N = multi_graph.vocab_size
    timeprint('caching original graph features')
    
    # report
    if opts.v > 0:
        timeprint('starting epoch {}'.format(ep_idx))
        
    if not opts.rand_all:
        # iterate over relations
        graphs_order = list(multi_graph.graphs.keys())
        if opts.rand_nodes:
            dy.np.random.shuffle(graphs_order)
        for rel in graphs_order:
            # report
            if opts.v > 0:
                timeprint('starting loop over {}'.format(rel))
            
            if opts.skip_symmetrics and rel in SYMMETRIC_RELATIONS:
                timeprint('skipping symmetric relation {}'.format(rel))
                continue
                
            if rel == 'co_hypernym':
                timeprint('skipping auxiliary co_hypernym relation')
                continue

            # compute target-wide association cache (no backprop)
            if use_assoc:
                assoc_cache = multi_graph.source_ranker_cache(rel)
            else:
                assoc_cache = np.zeros((multi_graph.word_assoc_weights[rel].shape()[0], multi_graph.embeddings.shape()[1]))
            timeprint('calculated association cache for {}'.format(rel))

            # iterate over nodes:
            node_order = list(range(N))
            if opts.rand_nodes:
                dy.np.random.shuffle(node_order)
            for src_i in tqdm(node_order):
                assoc_cache = macro_node_iteration(opts, multi_graph, assoc_cache,
                                                   trainer, log_file, synsets, rel, src_i, use_assoc)

            multi_graph.rescore()
            # total score = sum over all nodes
            iteration_scores.append(multi_graph.score)
    else:
        # iterate randomly over <rel, node>-s iid
        # rand_nodes implied
        all_rels = list(multi_graph.graphs.keys())
        if opts.skip_symmetrics:
            rels = [r for r in all_rels if r not in SYMMETRIC_RELATIONS]
        else:
            rels = all_rels
            
        if 'co_hypernym' in rels:
            rels.remove('co_hypernym')
            
        if use_assoc:
            assoc_caches = {rel: multi_graph.source_ranker_cache(rel) for rel in rels}
        else:
            assoc_caches = {rel: np.zeros((multi_graph.word_assoc_weights[rel].shape()[0], multi_graph.embeddings.shape()[1])) for rel in rels}
                        
        relnode_order = list(range(N * len(rels)))
        dy.np.random.shuffle(relnode_order)
        for idx in tqdm(relnode_order):
            rel = rels[idx % len(rels)]
            src_i = idx % N
            assoc_caches[rel] = macro_node_iteration(opts, multi_graph, assoc_caches[rel],
                                                     trainer, log_file, synsets, rel, src_i, use_assoc)
        
        # only happens once in this setup
        multi_graph.rescore()
        # total score = sum over all nodes
        iteration_scores.append(multi_graph.score)

    return iteration_scores
    