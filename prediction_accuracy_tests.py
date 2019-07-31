import re
import os
import argparse
import random
import copy
import numpy as np
import access_model as tp
import scipy
import scipy.stats
from io import open
from collections import Counter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

def get_acc(scorelist):
    if len(scorelist) == 0:
        acc = 0
    else:
        acc = float(sum(scorelist))/len(scorelist)
    return acc

def test_role_acc(hldict,inputlist,tgtlist,model,tokenizer,setting,fklog,k=5,bert=True):
    tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    tot_score = []
    by_constraint_score = {}
    by_constraint_score['H'] = []
    by_constraint_score['L'] = []
    correct = []
    used = []
    for i,pr in enumerate(tok_preds):
        sent = hldict[i]['sent'][setting]
        hldict[i]['toppreds'] = pr
        hldict[i]['tgtprob'] = tok_probs[i]
        hldict[i]['topprobs'] = top_probs[i]
        score = 0
        if sent in used: continue
        used.append(sent)
        for subpr in pr:
            for candidate in subpr:
                if candidate.strip() in hldict[i]['exp']:
                    score = 1
        if score == 1:
            ctup = (hldict[i]['sent'][setting],hldict[i]['exp'],pr,hldict[i]['constraint'])
            if ctup not in correct:
                correct.append(ctup)
        tot_score.append(score)
        by_constraint_score[hldict[i]['constraint']].append(score)
    conddict = make_conddict(hldict)
    n4report = sim_fk_N400(conddict,fklog,setting,k=k,bert=bert)
    tot_acc = get_acc(tot_score)
    report = '\nPrediction accuracies:\n'
    report += 'EXP TGT in TOP %s preds: %s (%s/%s)\n'%(k,tot_acc,sum(tot_score),len(tot_score))
    report += 'in TOP %s for H: %s\n'%(k,get_acc(by_constraint_score['H']))
    report += 'in TOP %s for L: %s\n'%(k,get_acc(by_constraint_score['L']))
    return report,n4report,correct,tot_acc,oov_list

def test_role_acc(clozedict,inputlist,tgtlist,clozelist,model,tokenizer,rrlog,k=5,bert=True,scat=None):
    tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    tot_score = []
    correct = []
    correct_by_quartile = {'q1_corr':[],'q2_corr':[],'q3_corr':[],'q4_corr':[]}
    # q1_corr = []
    # q2_corr = []
    # q3_corr = []
    # q4_corr = []
    avgcloze = np.average(clozelist)
    q1 = np.percentile(clozelist,25)
    q3 = np.percentile(clozelist,75)
    q2 = np.percentile(clozelist,50)
    q4 = max(clozelist)
    predcounts = Counter()
    for i,pred in enumerate(tok_preds):
        clozedict[i]['toppreds'] = pred
        clozedict[i]['tgtprob'] = tok_probs[i]
        clozedict[i]['topprobs'] = top_probs[i]
        score = 0
        itm = clozedict[i]['item']
        cond = clozedict[i]['cond']
        for subpr in pred:
            predcounts.update(subpr)
            for candidate in subpr:
                if candidate.strip() in [e.split()[0] for e in clozedict[i]['exp']]:
                    score = 1
        if score == 1:
            ctup = (clozedict[i]['sent'],clozedict[i]['exp'],pred,clozedict[i]['maxcloze'])
            if ctup not in correct:
                correct.append(ctup)
        tot_score.append(score)
        if clozedict[i]['maxcloze'] <= q1:
            correct_by_quartile['q1_corr'].append(score)
        elif clozedict[i]['maxcloze'] <= q2:
            correct_by_quartile['q2_corr'].append(score)
        elif clozedict[i]['maxcloze'] <= q3:
            correct_by_quartile['q3_corr'].append(score)
        else:
            correct_by_quartile['q4_corr'].append(score)
    conddict = make_conddict(clozedict)
    n4report = sim_rr_N400(conddict,rrlog,scat=scat,k=k,bert=bert)
    report = '\nPrediction accuracies:\n'
    report += 'TGT in top %s preds: %s (%s/%s)\n'%(k,get_acc(tot_score),sum(tot_score),len(tot_score))
    report += 'TGT in top %s for Q1: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q1_corr']),q1,len(correct_by_quartile['q1_corr']))
    report += 'TGT in top %s for Q2: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q2_corr']),q2,len(correct_by_quartile['q2_corr']))
    report += 'TGT in top %s for Q3: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q3_corr']),q3,len(correct_by_quartile['q3_corr']))
    report += 'TGT in top %s for Q4: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q4_corr']),q4,len(correct_by_quartile['q4_corr']))
    report += 'AVG CLOZE: %s\n'%avgcloze
    report += 'MED CLOZE: %s\n'%q2
    return report,n4report,correct,predcounts,oov_list

def test_nkf_acc(nkfdict,inputlist,tgtlist,model,tokenizer,nkflog,k=5,bert=True):
    correct = []
    tot_score = []
    tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    for i,pred in enumerate(tok_preds):
        nkfdict[i]['toppreds'] = pred
        nkfdict[i]['tgtprob'] = tok_probs[i]
        nkfdict[i]['topprobs'] = top_probs[i]
        score = 0
        for subpr in pred:
            rawclozevals = []
            for candidate in subpr:
                if candidate.strip() in nkfdict[i]['exp']:
                    score = 1
        if score == 1:
            ctup = (nkfdict[i]['sent'],nkfdict[i]['exp'],pred,nkfdict[i]['cond'])
            if ctup not in correct:
                correct.append(ctup)
        if nkfdict[i]['cond'] == 'TA':
            tot_score.append(score)
    conddict = make_conddict(nkfdict)
    n4report = sim_nkf_N400(conddict,nkflog,k=k,bert=bert)
    report = "\nPrediction 'accuracy':\n"
    report += 'TRUE TGT in top %s preds: %s (%s/%s)\n'%(k,get_acc(tot_score),sum(tot_score),len(tot_score))

    return report,n4report,correct,oov_list
