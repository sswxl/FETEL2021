import torch
import numpy as np
import time
from typing import List
from Biye2021.models.feteldeep import FETELStack
from Biye2021.models.fetentvecutils import ELDirectEntityVec
from Biye2021.modelexp import exputils
from Biye2021.modelexp.exputils import ModelSample, anchor_samples_to_model_samples, model_samples_from_json
import logging
from Biye2021.utils import datautils, utils


def __get_l2_person_type_ids(type_vocab):
    person_type_ids = list()
    for i, t in enumerate(type_vocab):
        if t.startswith('/person') and t != '/person':
            person_type_ids.append(i)
    return person_type_ids


def __get_entity_vecs_for_samples(el_entityvec: ELDirectEntityVec, samples: List[ModelSample], noel_pred_results,
                                  filter_by_pop=False, person_type_id=None, person_l2_type_ids=None, type_vocab=None):
    mstrs = [s.mention_str for s in samples]
    prev_pred_labels = None
    if noel_pred_results is not None:
        prev_pred_labels = [noel_pred_results[s.mention_id] for s in samples]
    return el_entityvec.get_entity_vecs(
        mstrs, prev_pred_labels, filter_by_pop=filter_by_pop, person_type_id=person_type_id,
        person_l2_type_ids=person_l2_type_ids, type_vocab=type_vocab)


def __get_entity_vecs_for_mentions(el_entityvec: ELDirectEntityVec, mentions, noel_pred_results, n_types,
                                   filter_by_pop=False):
    all_entity_type_vecs = -np.ones((len(mentions), n_types), np.float32)
    all_el_sgns = np.zeros(len(mentions), np.float32)
    all_probs = np.zeros(len(mentions), np.float32)
    mention_id_to_idxs = {m['mention_id']: i for i, m in enumerate(mentions)}
    doc_mentions_dict = utils.json_objs_to_kvlistdict(mentions, 'file_id') # file_id:16
    for doc_id, doc_mentions in doc_mentions_dict.items():
        prev_pred_labels = [noel_pred_results[m['mention_id']] for m in doc_mentions]
        mstrs = [m['str'] for m in doc_mentions]
        entity_type_vecs, el_sgns, probs = el_entityvec.get_entity_vecs(mstrs, prev_pred_labels,
                                                                        filter_by_pop=filter_by_pop)
        # print(entity_type_vecs.shape)
        for m, vec, el_sgn, prob_vec in zip(doc_mentions, entity_type_vecs, el_sgns, probs):
            idx = mention_id_to_idxs[m['mention_id']]
            # print(vec.shape)
            all_entity_type_vecs[idx] = vec
            all_el_sgns[idx] = el_sgn
            all_probs[idx] = prob_vec
    return all_entity_type_vecs, all_el_sgns, all_probs


def train_fetel(device, gres: exputils.GlobalRes, el_entityvec: ELDirectEntityVec, train_samples_pkl,
                dev_samples_pkl, test_mentions_file, test_sents_file, test_noel_preds_file, type_embed_dim,
                context_lstm_hidden_dim, learning_rate, batch_size, n_iter, dropout, rand_per, per_penalty,
                use_mlp=False, pred_mlp_hdim=None, save_model_file=None, nil_rate=0.5,
                single_type_path=False, stack_lstm=False, concat_lstm=False, test_results_file=None,dev_results_file = None):
    logging.info('test_results_file={}'.format(test_results_file))
    logging.info('dev_results_file={}'.format(dev_results_file))
    logging.info(
        'type_embed_dim={} cxt_lstm_hidden_dim={} pmlp_hdim={} nil_rate={} single_type_path={}'.format(
            type_embed_dim, context_lstm_hidden_dim, pred_mlp_hdim, nil_rate, single_type_path))
    logging.info('rand_per={} per_pen={}'.format(rand_per, per_penalty))
    logging.info('stack_lstm={} cat_lstm={}'.format(stack_lstm, concat_lstm))

    if stack_lstm:
        model = FETELStack(
            device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer, context_lstm_hidden_dim,
            type_embed_dim=type_embed_dim, dropout=dropout, use_mlp=use_mlp, mlp_hidden_dim=pred_mlp_hdim,
            concat_lstm=concat_lstm)
    else:
        model = None
    if device.type == 'cuda':
        model = model.cuda(device.index)

    train_samples = datautils.load_pickle_data(train_samples_pkl)#(4932861,7)
    # mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids
    dev_samples = datautils.load_pickle_data(dev_samples_pkl)  #(2000, 7)

    # datautils.save_pickle_data(train_samples[:1000], 'E:/Pycoding/biye/Biye2021/data/fetel-data/results/train.pkl')
    # datautils.save_pickle_data(dev_samples[:200], 'E:/Pycoding/biye/Biye2021/data/fetel-data/results/dev.pkl')

    #mention_token_id=3
    # parent_type_ids_dict={0: [], 1: [54], 2: [], 3: [63], 4: [],......           128
    dev_samples = anchor_samples_to_model_samples(dev_samples, gres.mention_token_id, gres.parent_type_ids_dict)
    # mention_id, mention_str, mstr_token_seq, context_token_seq, mention_token_idx, labels
    lr_gamma = 0.7
    eval_batch_size = 16
    logging.info('{}'.format(model.__class__.__name__))
    dev_true_labels_dict = {s.mention_id: [gres.type_vocab[l] for l in s.labels] for s in dev_samples}
    dev_entity_vecs, dev_el_sgns, dev_el_probs = __get_entity_vecs_for_samples(el_entityvec, dev_samples, None)

    test_samples = model_samples_from_json(gres.token_id_dict, gres.unknown_token_id, gres.mention_token_id,
                                           gres.type_id_dict, test_mentions_file, test_sents_file)
    test_noel_pred_results = datautils.read_pred_results_file(test_noel_preds_file) # test_true_labels_dict
    # mention,labels
    test_mentions = datautils.read_json_objs(test_mentions_file)
    test_entity_vecs, test_el_sgns, test_el_probs = __get_entity_vecs_for_mentions( # 因为测试的时候已经知道实体提及和标签了
        el_entityvec, test_mentions, test_noel_pred_results, gres.n_types)

    test_true_labels_dict = {m['mention_id']: m['labels'] for m in test_mentions} if (
            'labels' in next(iter(test_mentions))) else None

    n_batches = (len(train_samples) + batch_size - 1) // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batches, gamma=lr_gamma)
    losses = list()
    best_dev_acc = -1
    logging.info('{} steps, {} steps per iter, lr_decay={}, start training ...'.format(
        n_iter * n_batches, n_batches, lr_gamma))
    step = 0
    n_steps = n_iter * n_batches
    while step < n_steps:
        batch_idx = step % n_batches
        batch_beg, batch_end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(train_samples))
        batch_samples = anchor_samples_to_model_samples(
            train_samples[batch_beg:batch_end], gres.mention_token_id, gres.parent_type_ids_dict)
        entity_vecs, el_sgns, el_probs = __get_entity_vecs_for_samples(el_entityvec, batch_samples, None, True)#在训练的时候这个实体类型是已经有的。不需要链接
        #16,128  16   16
        use_entity_vecs = True
        model.train()

        (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, y_true
         ) = exputils.get_mstr_cxt_label_batch_input(model.device, gres.n_types, batch_samples)

        el_probs = torch.tensor(el_probs, dtype=torch.float32, device=model.device)
        entity_vecs = torch.tensor(entity_vecs, dtype=torch.float32, device=model.device)
        t = model(context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs)

        loss = model.cross_entropy(t,y_true)#16,128
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
        optimizer.step()
        scheduler.step()
        losses.append(loss.data.cpu().numpy().item())
        step += 1
        if step % 1 == 0:
            # logging.info('i={} l={:.4f}'.format(step + 1, sum(losses)))
            acc_v, pacc_v, _, _, dev_results = eval_fetel(
                gres, model, dev_samples, dev_entity_vecs, dev_el_probs, eval_batch_size,
                use_entity_vecs=use_entity_vecs, single_type_path=single_type_path,
                true_labels_dict=dev_true_labels_dict)
            acc_t, _, maf1, mif1, test_results = eval_fetel(
                gres, model, test_samples, test_entity_vecs, test_el_probs, eval_batch_size,
                use_entity_vecs=use_entity_vecs, single_type_path=single_type_path,
                true_labels_dict=test_true_labels_dict)

            best_tag = '*' if acc_v > best_dev_acc else ''
            logging.info(
                'i={} l={:.4f} accv={:.4f} paccv={:.4f} acct={:.4f} maf1={:.4f} mif1={:.4f}{}'.format(
                    step, sum(losses), acc_v, pacc_v, acc_t, maf1, mif1, best_tag))
            # if acc_v > best_dev_acc and save_model_file:
            #     torch.save(model.state_dict(), save_model_file)
            #     logging.info('model saved to {}'.format(save_model_file))
            #
            # if dev_results_file is not None and acc_v > best_dev_acc:
            #     datautils.save_json_objs(dev_results, dev_results_file)
            #     logging.info('dev reuslts saved {}'.format(dev_results_file))
            # if results_file is not None and acc_v > best_dev_acc:
            #     datautils.save_json_objs(test_results, results_file)
            #     logging.info('test reuslts saved {}'.format(results_file))
            # datautils.save_json_objs(dev_results, dev_results_file)
            # logging.info('dev reuslts saved {}'.format(dev_results_file))
            # datautils.save_json_objs(test_results, test_results_file)
            # logging.info('test reuslts saved {}'.format(test_results_file))
            # if acc_v > best_dev_acc:
            #     best_dev_acc = acc_v
            # losses = list()
            # with open("temp.txt", 'w') as f:
            #     f.write(str(losses))


def eval_fetel(gres: exputils.GlobalRes, model, samples: List[ModelSample], entity_vecs, el_probs, batch_size=32,
               use_entity_vecs=True, single_type_path=False, true_labels_dict=None):
    model.eval()
    n_batches = (len(samples) + batch_size - 1) // batch_size
    pred_labels_dict = dict()
    result_objs = list()
    for i in range(n_batches):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]
        (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs
         ) = exputils.get_mstr_cxt_batch_input(batch_samples)
        entity_vecs_batch, el_probs_batch = None, None
        if use_entity_vecs:
            # entity_vecs, el_sgns = __get_entity_vecs_for_samples(el_entityvec, batch_samples, noel_pred_results)
            entity_vecs_batch = torch.tensor(entity_vecs[batch_beg:batch_end], dtype=torch.float32,
                                             device=model.device)
            # el_sgns_batch = torch.tensor(el_sgns[batch_beg:batch_end], dtype=torch.float32, device=model.device)
            el_probs_batch = torch.tensor(el_probs[batch_beg:batch_end], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs,
                           entity_vecs_batch, el_probs_batch)
            # logits = torch.argmax(p, dim=1)

        if single_type_path:
            preds = model.inference(logits)
        else:
            preds = model.inference_full(logits, extra_label_thres=0.5)
        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(batch_samples, preds, logits.data.cpu().numpy())):
            labels = utils.get_full_types([gres.type_vocab[tid] for tid in type_ids_pred])
            pred_labels_dict[sample.mention_id] = labels
            result_objs.append({'mention_id': sample.mention_id, 'labels': labels})

    strict_acc, partial_acc, maf1, mif1 = 0, 0, 0, 0
    if true_labels_dict is not None:
        strict_acc = utils.strict_acc(true_labels_dict, pred_labels_dict)
        partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
        maf1 = utils.macrof1(true_labels_dict, pred_labels_dict)
        mif1 = utils.microf1(true_labels_dict, pred_labels_dict)
    return strict_acc, partial_acc, maf1, mif1, result_objs
