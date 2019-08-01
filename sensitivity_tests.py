import re
import os
import argparse
import random
import copy
import numpy as np
from proc_datasets import *
from io import open


def convert_to_experiment_grouping(datadict):
    conddict = {}
    for k in datadict:
        it = datadict[k]['item']
        co = datadict[k]['cond']
        if it not in conddict:
            conddict[it] = {}
        conddict[it][co] = {}
        for keycat in ['sent','tgt','tgtcloze','constraint','licensing','maxcloze','tgtprob','toppreds','topprobs']:
            if keycat in datadict[k]:
                conddict[it][co][keycat] = datadict[k][keycat]
    return conddict

def cprag_sensitivity_test(dataset_ref,target_probs):
    for i,prob in enumerate(target_probs):
        dataset_ref[i]['tgtprob'] = prob
    conddict = convert_to_experiment_grouping(dataset_ref)
    thresh = 0.01
    exp_top = {'H':[],'L':[]}
    exp_top_thresh = {'H':[],'L':[]}
    allprobs = {'H':[],'L':[]}
    for it in conddict:
        exp_prob,wc_prob,bc_prob = [conddict[it][cont]['tgtprob'] for cont in ['exp','wc','bc']]
        cons = conddict[it]['exp']['constraint']
        allprobs[cons].append((exp_prob,wc_prob,bc_prob))
        if (exp_prob > wc_prob) and (exp_prob > bc_prob):
            exp_top[cons].append(1)
            if abs(exp_prob - wc_prob) > thresh and abs(exp_prob - bc_prob) > thresh:
                exp_top_thresh[cons].append(1)
            else:
                exp_top_thresh[cons].append(0)
        else:
            exp_top[cons].append(0)
            exp_top_thresh[cons].append(0)

    report = '\nSensitivity results:\n\n'

    report += 'Expected word more probable than two inappropriate targets: %s (%s/%s)\n'%(get_acc(exp_top['H']+exp_top['L']),sum(exp_top['H']+exp_top['L']),len(exp_top['H']+exp_top['L']))
    report += '  for high-constraint items: %s (%s/%s)\n'%(get_acc(exp_top['H']),sum(exp_top['H']),len(exp_top['H']))
    report += '  for low-constraint items: %s (%s/%s)\n\n'%(get_acc(exp_top['L']),sum(exp_top['L']),len(exp_top['L']))
    report += 'Expected word more probable than two inappropriate targets -- difference threshold %s: %s (%s/%s)\n'%(thresh,get_acc(exp_top_thresh['H']+exp_top_thresh['L']),sum(exp_top_thresh['H']+exp_top_thresh['L']),len(exp_top_thresh['H']+exp_top_thresh['L']))
    report += '  for high-constraint items: %s (%s/%s)\n'%(get_acc(exp_top_thresh['H']),sum(exp_top_thresh['H']),len(exp_top_thresh['H']))
    report += '  for low-constraint items: %s (%s/%s)\n'%(get_acc(exp_top_thresh['L']),sum(exp_top_thresh['L']),len(exp_top_thresh['L']))

    return report

def role_sensitivity_test(dataset_ref,target_probs):
    dataset_dict,clozelist = dataset_ref
    for i,prob in enumerate(target_probs):
        dataset_dict[i]['tgtprob'] = prob
    conddict = convert_to_experiment_grouping(dataset_dict)
    thresh = 0.01
    good_top = []
    good_top_thresh = []
    probpairs = []
    clozepairs = []
    for it in conddict:
        a_prob,b_prob = (conddict[it]['a']['tgtprob'],conddict[it]['b']['tgtprob'])
        if (a_prob > b_prob):
            good_top.append(1)
        else:
            good_top.append(0)
        if (a_prob > b_prob) and (abs(a_prob - b_prob) > thresh):
            good_top_thresh.append(1)
        else:
            good_top_thresh.append(0)
        probpairs.append((a_prob,b_prob))
        clozepairs.append((conddict[it]['a']['tgtcloze'],conddict[it]['b']['tgtcloze']))
    probdiffs = [e[0] - e[1] for e in probpairs]
    clozediffs = [e[0] - e[1] for e in clozepairs]

    report = '\nSensitivity results:\n\n'

    report += 'Good completion more probable than role reversal: %s (%s/%s)\n\n'%(get_acc(good_top),sum(good_top),len(good_top))
    report += 'Good completion more probable than role reversal -- difference threshold %s: %s (%s/%s)\n\n'%(thresh,get_acc(good_top_thresh),sum(good_top_thresh),len(good_top_thresh))
    report += 'AVG PROB DIFF: %s\n'%np.average(probdiffs)
    report += 'AVG CLOZE DIFF: %s\n'%np.average(clozediffs)

    return report

def neg_sensitivity_test(dataset_ref,target_probs):
    for i,prob in enumerate(target_probs):
        dataset_ref[i]['tgtprob'] = prob
    conddict = convert_to_experiment_grouping(dataset_ref)
    thresh = 0.01
    pattern = []
    same = []
    preftrue = {'aff':[],'neg':[]}
    preftrue_l = {'aff':[],'neg':[]}
    preftrue_u = {'aff':[],'neg':[]}
    preftrue_thresh = {'aff':[],'neg':[]}
    lic = None
    for it in conddict:
        if 'licensing' in conddict[it]['TA']:
            lic = conddict[it]['TA']['licensing']
        for true_cond,false_cond,pol in [('TA','FA','aff'),('TN','FN','neg')]:
            true_prob,false_prob = (conddict[it][true_cond]['tgtprob'],conddict[it][false_cond]['tgtprob'])
            if true_prob > false_prob:
                score = 1
            else:
                score = 0
            preftrue[pol].append(score)
            if lic:
                if lic == 'Y':
                    preftrue_l[pol].append(score)
                elif lic == 'N':
                    preftrue_u[pol].append(score)
            if (true_prob > false_prob) and (abs(true_prob - false_prob) > thresh):
                preftrue_thresh[pol].append(1)
            else:
                preftrue_thresh[pol].append(0)


    report = '\nSensitivity results:\n\n'
    report += 'True completion more probable than false: %s (%s/%s)\n'%(get_acc(preftrue['aff'] + preftrue['neg']),sum(preftrue['aff'] + preftrue['neg']),len(preftrue['aff'] + preftrue['neg']))
    report += '   in affirmative contexts: %s (%s/%s)\n'%(get_acc(preftrue['aff']),sum(preftrue['aff']),len(preftrue['aff']))
    report += '   in negative contexts: %s (%s/%s)\n'%(get_acc(preftrue['neg']),sum(preftrue['neg']),len(preftrue['neg']))
    report += 'True completion more probable than false -- difference threshold %s: %s (%s/%s)\n'%(thresh,get_acc(preftrue_thresh['aff']+preftrue_thresh['neg']),sum(preftrue_thresh['aff']+preftrue_thresh['neg']),len(preftrue_thresh['aff']+preftrue_thresh['neg']))
    report += '   in affirmative contexts: %s (%s/%s)\n'%(get_acc(preftrue_thresh['aff']),sum(preftrue_thresh['aff']),len(preftrue_thresh['aff']))
    report += '   in negative contexts: %s (%s/%s)\n\n'%(get_acc(preftrue_thresh['neg']),sum(preftrue_thresh['neg']),len(preftrue_thresh['neg']))
    if lic:
        report += 'True completion more probable in NATURAL sentences: %s (%s/%s)\n'%(get_acc(preftrue_l['aff'] + preftrue_l['neg']),sum(preftrue_l['aff'] + preftrue_l['neg']),len(preftrue_l['aff'] + preftrue_l['neg']))
        report += '   in affirmative contexts: %s (%s/%s)\n'%(get_acc(preftrue_l['aff']),sum(preftrue_l['aff']),len(preftrue_l['aff']))
        report += '   in negative contexts: %s (%s/%s)\n'%(get_acc(preftrue_l['neg']),sum(preftrue_l['neg']),len(preftrue_l['neg']))
        report += 'True completion more probable in LESS NATURAL sentences: %s (%s/%s)\n'%(get_acc(preftrue_u['aff'] + preftrue_u['neg']),sum(preftrue_u['aff'] + preftrue_u['neg']),len(preftrue_u['aff'] + preftrue_u['neg']))
        report += '   in affirmative contexts: %s (%s/%s)\n'%(get_acc(preftrue_u['aff']),sum(preftrue_u['aff']),len(preftrue_u['aff']))
        report += '   in negative contexts: %s (%s/%s)\n'%(get_acc(preftrue_u['neg']),sum(preftrue_u['neg']),len(preftrue_u['neg']))
    report += '\n\n'

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preddir",default=None, type=str)
    parser.add_argument("resultsdir",default=None, type=str)
    parser.add_argument("--models", nargs="+", type=str)
    parser.add_argument("--cprag_stim", default=None, type=str)
    parser.add_argument("--role_stim", default=None, type=str)
    parser.add_argument("--negsimp_stim", default=None, type=str)
    parser.add_argument("--negnat_stim", default=None, type=str)
    args = parser.parse_args()

    testlist = [
    (args.cprag_stim, cprag_sensitivity_test,'cprag',process_cprag),
    (args.role_stim, role_sensitivity_test,'role',process_role),
    (args.negsimp_stim, neg_sensitivity_test,'negsimp',process_negsimp),
    (args.negnat_stim, neg_sensitivity_test,'negnat',process_negnat)
    ]

    for stimfile,sens_test,testname,process_func in testlist:
        if not stimfile: continue
        inputlist,_,dataset_ref = process_func(stimfile,mask_tok=False)
        with open(args.resultsdir+'/sensitivity-%s.txt'%testname,'wb') as out:
            for modelname in args.models:
                out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
                target_probs = []
                with open(os.path.join(args.preddir,'modeltgtprobs-%s-%s'%(testname,modelname))) as probfile:
                    for line in probfile: target_probs.append(float(line.strip()))
                report = sens_test(dataset_ref,target_probs)
                out.write(report)
