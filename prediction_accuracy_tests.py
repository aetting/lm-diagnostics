import re
import os
import argparse
import random
import copy
import numpy as np
from proc_datasets import *
import scipy
import scipy.stats
from io import open
import itertools
from collections import Counter

def test_cprag_acc(dataset_ref,word_preds,k=5):
    tot_score = []
    correct = []
    used = []
    for i,pr in enumerate(word_preds):
        sent = dataset_ref[i]['sent']
        score = 0
        if sent in used: continue
        used.append(sent)
        for candidate in pr:
            if candidate.strip() in dataset_ref[i]['exp']:
                score = 1
        if score == 1:
            ctup = (sent,dataset_ref[i]['exp'],pr)
            correct.append(ctup)
        tot_score.append(score)
    tot_acc = get_acc(tot_score)
    report = '\nPrediction accuracies:\n'
    report += 'EXPECTED TARGET in TOP %s predictions: %s (%s/%s)\n\n'%(k,tot_acc,sum(tot_score),len(tot_score))
    report += 'ITEMS PREDICTED CORRECTLY:\n\n'
    for sent,exp,pred in correct:
        report += sent.encode('utf-8')
        report += '\n   EXPECTED: %s | PREDICTED: %s\n'%(','.join(exp).encode('utf-8'),','.join(pred).encode('utf-8'))
    return report

def test_role_acc(dataset_ref,word_preds,k=5):
    dataset_dict,clozelist = dataset_ref
    tot_score = []
    correct = []
    correct_by_quartile = {'q1_corr':[],'q2_corr':[],'q3_corr':[],'q4_corr':[]}
    avgcloze = np.average(clozelist)
    q1 = np.percentile(clozelist,25)
    q3 = np.percentile(clozelist,75)
    q2 = np.percentile(clozelist,50)
    q4 = max(clozelist)
    for i,pred in enumerate(word_preds):
        sent = dataset_dict[i]['sent']
        score = 0
        itm = dataset_dict[i]['item']
        cond = dataset_dict[i]['cond']
        for candidate in pred:
            if candidate.strip() in [e.split()[0] for e in dataset_dict[i]['exp']]:
                score = 1
        if score == 1:
            ctup = (sent,dataset_dict[i]['exp'],pred,dataset_dict[i]['maxcloze'])
            correct.append(ctup)
        tot_score.append(score)
        if dataset_dict[i]['maxcloze'] <= q1:
            correct_by_quartile['q1_corr'].append(score)
        elif dataset_dict[i]['maxcloze'] <= q2:
            correct_by_quartile['q2_corr'].append(score)
        elif dataset_dict[i]['maxcloze'] <= q3:
            correct_by_quartile['q3_corr'].append(score)
        else:
            correct_by_quartile['q4_corr'].append(score)
    report = '\nPrediction accuracies:\n'
    report += 'EXPECTED WORD in top %s predictions: %s (%s/%s)\n'%(k,get_acc(tot_score),sum(tot_score),len(tot_score))
    report += '  for cloze quartile 1: %s (%s upper, %s items)\n'%(get_acc(correct_by_quartile['q1_corr']),q1,len(correct_by_quartile['q1_corr']))
    report += '  for cloze quartile 2: %s (%s upper, %s items)\n'%(get_acc(correct_by_quartile['q2_corr']),q2,len(correct_by_quartile['q2_corr']))
    report += '  for cloze quartile 3: %s (%s upper, %s items)\n'%(get_acc(correct_by_quartile['q3_corr']),q3,len(correct_by_quartile['q3_corr']))
    report += '  for cloze quartile 4: %s (%s upper, %s items)\n\n'%(get_acc(correct_by_quartile['q4_corr']),q4,len(correct_by_quartile['q4_corr']))

    report += 'ITEMS PREDICTED CORRECTLY:\n\n'
    for sent,exp,pred,mc in correct:
        report += '%s\n   EXPECTED: %s | PREDICTED: %s\n'%(sent,','.join(exp),','.join(pred))
    return report

def test_neg_acc(dataset_ref,word_preds,k=5):
    correct = []
    tot_score = []
    used = []
    for i,pred in enumerate(word_preds):
        score = 0
        sent = dataset_ref[i]['sent']
        if sent in used: continue
        used.append(sent)
        for candidate in pred:
            if candidate.strip() in dataset_ref[i]['exp']:
                score = 1
        if score == 1 and dataset_ref[i]['cond'] == 'TA':
            ctup = (sent,dataset_ref[i]['exp'],pred,dataset_ref[i]['cond'])
            correct.append(ctup)
        if dataset_ref[i]['cond'] == 'TA':
            tot_score.append(score)
    report = "\nPrediction 'accuracy' (affirmative contexts only):\n"
    report += 'TRUE COMPLETION in top %s predictions: %s (%s/%s)\n\n'%(k,get_acc(tot_score),sum(tot_score),len(tot_score))

    report += 'ITEMS PREDICTED CORRECTLY:\n\n'
    for sent,exp,pred,cond in correct:
        report += '%s\n   EXPECTED: %s | PREDICTED: %s\n'%(sent,','.join(exp),','.join(pred))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preddir",default=None, type=str)
    parser.add_argument("--resultsdir",default=None, type=str)
    parser.add_argument("--models", nargs="+", type=str)
    parser.add_argument("--k_values", nargs="+", type=int)
    parser.add_argument("--cprag_stim", default=None, type=str)
    parser.add_argument("--role_stim", default=None, type=str)
    parser.add_argument("--negsimp_stim", default=None, type=str)
    parser.add_argument("--negnat_stim", default=None, type=str)
    args = parser.parse_args()

    testlist = [
    (args.cprag_stim, test_cprag_acc,'cprag',process_cprag),
    (args.role_stim, test_role_acc,'role',process_role),
    (args.negsimp_stim, test_neg_acc,'negsimp',process_negsimp),
    (args.negnat_stim, test_neg_acc,'negnat',process_negnat)
    ]

    for stimfile,acc_test,testname,process_func in testlist:
        if not stimfile: continue
        inputlist,_,dataset_ref = process_func(stimfile,mask_tok=False)
        with open(args.resultsdir+'/prediction_accuracy-%s.txt'%testname,'wb') as out:
            for modelname in args.models:
                out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
                word_preds_full = []
                with open(os.path.join(args.preddir,'modelpreds-%s-%s'%(testname,modelname))) as predfile:
                    for line in predfile: word_preds_full.append(line.strip().split())
                for k in args.k_values:
                    out.write('\n--\nk = %s\n--\n'%k)
                    word_preds = [p[:k] for p in word_preds_full]
                    report = acc_test(dataset_ref,word_preds,k=k)
                    out.write(report)
