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

def make_conddict(clozedict):
    conddict = {}
    for k in clozedict:
        it = clozedict[k]['item']
        co = clozedict[k]['cond']
        if it not in conddict:
            conddict[it] = {}
        conddict[it][co] = {}
        conddict[it][co]['sent'] = clozedict[k]['sent']
        conddict[it][co]['tgt'] = clozedict[k]['tgt']
        if 'tgtcloze' in clozedict[k]:
            conddict[it][co]['tgtcloze'] = clozedict[k]['tgtcloze']
        if 'constraint' in clozedict[k]:
            conddict[it][co]['constraint'] = clozedict[k]['constraint']
        if 'licensing' in clozedict[k]:
            conddict[it][co]['licensing'] = clozedict[k]['licensing']
        if 'maxcloze' in clozedict[k]:
            conddict[it][co]['maxcloze'] = clozedict[k]['maxcloze']
        conddict[it][co]['tgtprob'] = clozedict[k]['tgtprob']
        conddict[it][co]['toppreds'] = clozedict[k]['toppreds']
        conddict[it][co]['topprobs'] = clozedict[k]['topprobs']
    return conddict

def sim_cprag_N400(conddict,logfile,setting,k=5,bert=True):
    thresh = 0.01
    exp_top = {'H':[],'L':[]}
    exp_top_thresh = {'H':[],'L':[]}
    wc_boost = {'H':[],'L':[]}
    allprobs = {'H':[],'L':[]}
    for it in conddict:
        exp_prob,wc_prob,bc_prob = [conddict[it][cont]['tgtprob'][0] for cont in ['exp','wc','bc']]
        logfile.write(conddict[it]['exp']['sent'][setting].encode('utf-8'))
        logfile.write(' ' + '/'.join([conddict[it][cont]['tgt'] for cont in ['exp','wc','bc']]) + '\n')
        logfile.write('TGT probs: %s\n'%[exp_prob,wc_prob,bc_prob])
        logfile.write('PREDICTED: %s\n'%conddict[it]['exp']['toppreds'])
        logfile.write(str(conddict[it]['exp']['topprobs']) + '\n')
        cons = conddict[it]['exp']['constraint']
        logfile.write(cons + '\n')
        allprobs[cons].append((exp_prob,wc_prob,bc_prob))
        if (exp_prob > wc_prob) and (exp_prob > bc_prob):
            exp_top[cons].append(1)
            logfile.write('EXP TOP\n')
            if abs(exp_prob - wc_prob) > thresh and abs(exp_prob - bc_prob) > thresh:
                exp_top_thresh[cons].append(1)
            else:
                exp_top_thresh[cons].append(0)
        else:
            exp_top[cons].append(0)
            exp_top_thresh[cons].append(0)
        if (wc_prob > bc_prob) and abs(wc_prob - bc_prob) > thresh:
            wc_boost[cons].append(1)
            logfile.write('WC BOOST\n')
        else:
            wc_boost[cons].append(0)
        logfile.write('----\n\n\n')

    report = '\nTGT probability patterning:\n'
    report += 'AVG PROB HIGH EXP: %s\n'%np.average([e[0] for e in allprobs['H']])
    report += 'AVG PROB HIGH WC: %s\n'%np.average([e[1] for e in allprobs['H']])
    report += 'AVG PROB HIGH BC: %s\n'%np.average([e[2] for e in allprobs['H']])
    report += 'AVG PROB LOW EXP: %s\n'%np.average([e[0] for e in allprobs['L']])
    report += 'AVG PROB LOW WC: %s\n'%np.average([e[1] for e in allprobs['L']])
    report += 'AVG PROB LOW BC: %s\n'%np.average([e[2] for e in allprobs['L']])
    report += 'EXP TOP: %s\n'%get_acc(exp_top['H']+exp_top['L'])
    report += 'EXP TOP HIGH: %s\n'%get_acc(exp_top['H'])
    report += 'EXP TOP LOW: %s\n'%get_acc(exp_top['L'])
    report += 'EXP TOP w/ THRESH %s: %s\n'%(thresh,get_acc(exp_top_thresh['H']+exp_top_thresh['L']))
    report += 'EXP TOP HIGH w/ THRESH %s: %s\n'%(thresh,get_acc(exp_top_thresh['H']))
    report += 'EXP TOP LOW w/ THRESH %s: %s\n'%(thresh,get_acc(exp_top_thresh['L']))
    report += 'WC BOOST HIGH: %s\n'%get_acc(wc_boost['H'])
    report += 'WC BOOST LOW: %s\n'%get_acc(wc_boost['L'])

    return report

def sim_role_N400(conddict,logfile,scat=None,k=5,bert=True):
    thresh = .01
    pattern = []
    pattern_thresh = []
    same = []
    probpairs = []
    clozepairs = []
    for it in conddict:
        a_prob,b_prob = (conddict[it]['a']['tgtprob'][0],conddict[it]['b']['tgtprob'][0])
        logfile.write(conddict[it]['a']['sent'] + '   ' + conddict[it]['a']['tgt'] + '\n')
        logfile.write(conddict[it]['b']['sent'] + '   ' + conddict[it]['b']['tgt'] + '\n')
        logfile.write('TGT probs: %s\n'%[a_prob,b_prob])
        logfile.write('TGT cloze: %s\n'%[conddict[it]['a']['tgtcloze'],conddict[it]['b']['tgtcloze']])
        logfile.write('PREDICTED: %s'%conddict[it]['a']['toppreds'] + '\n')
        logfile.write(str(conddict[it]['a']['topprobs']) + '\n')
        logfile.write('PREDICTED: %s'%conddict[it]['b']['toppreds'] + '\n')
        logfile.write(str(conddict[it]['b']['topprobs']) + '\n')
        if (a_prob > b_prob):
            pattern.append(1)
            logfile.write('PATTERN\n')
        else:
            pattern.append(0)
        if (a_prob > b_prob) and (abs(a_prob - b_prob) > thresh):
            pattern_thresh.append(1)
            logfile.write('PATTERN THRESH\n')
        else:
            pattern_thresh.append(0)
        if (abs(a_prob - b_prob) < thresh):
            same.append(1)
            logfile.write('NO DIFF\n')
        else:
            same.append(0)
        logfile.write('----\n\n\n')
        probpairs.append((a_prob,b_prob))
        clozepairs.append((conddict[it]['a']['tgtcloze'],conddict[it]['b']['tgtcloze']))
    probdiffs = [e[0] - e[1] for e in probpairs]
    clozediffs = [e[0] - e[1] for e in clozepairs]

    report = '\nTarget probs vs cloze:\n'
    if sum(probdiffs) > 0:
        report += 'PEARSON: %s\n'%scipy.stats.pearsonr(probdiffs,clozediffs)[0]
        report += 'SPEARMAN: %s\n'%scipy.stats.spearmanr(probdiffs,clozediffs)[0]
    else:
        report += 'PEARSON: ---\n'
        report += 'SPEARMAN: ---\n'
    report += 'GOOD TGT HIGHER: %s\n'%get_acc(pattern)
    report += 'GOOD TGT HIGHER BY %s: %s\n'%(thresh,get_acc(pattern_thresh))
    report += 'DIFF BELOW THRESH %s: %s\n'%(thresh,get_acc(same))
    report += 'AVG PROB DIFF: %s\n'%np.average(probdiffs)
    report += 'AVG CLOZE DIFF: %s\n'%np.average(clozediffs)

    if scat:
        plot_rr_prcl(clozediffs,probdiffs,scat)

    return report

def sim_nkf_N400(conddict,logfile,k=5,bert=True):
    pattern = []
    same = []
    preftrue = {'aff':[],'neg':[]}
    preftrue_l = {'aff':[],'neg':[]}
    preftrue_u = {'aff':[],'neg':[]}
    preftrue_thresh = {'aff':[],'neg':[]}
    lic = None
    thresh = .01
    for it in conddict:
        if 'licensing' in conddict[it]['TA']:
            lic = conddict[it]['TA']['licensing']
        for true_cond,false_cond,pol in [('TA','FA','aff'),('TN','FN','neg')]:
            true_prob,false_prob = (conddict[it][true_cond]['tgtprob'][0],conddict[it][false_cond]['tgtprob'][0])
            logfile.write(str(conddict[it][true_cond]['sent'] + ' ' + conddict[it][true_cond]['tgt']) + '\n')
            logfile.write(str(conddict[it][false_cond]['sent'] + ' ' + conddict[it][false_cond]['tgt']) + '\n')
            logfile.write(u'TGT probs: %s\n'%[true_prob,false_prob])
            logfile.write(u'PREDICTED: %s'%conddict[it][true_cond]['toppreds'] + '\n')
            logfile.write(str(conddict[it][true_cond]['topprobs']) + '\n')
            logfile.write(u'PREDICTED: %s'%conddict[it][false_cond]['toppreds'] + '\n')
            logfile.write(str(conddict[it][false_cond]['topprobs']) + '\n')
            logfile.write(u'---\n\n\n')
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
                else:
                    print('LICENSING ERROR')
            if (true_prob > false_prob) and (abs(true_prob - false_prob) > thresh):
                preftrue_thresh[pol].append(1)
            else:
                preftrue_thresh[pol].append(0)


    report = '\nPreference for true vs false sentences:\n'
    report += 'PREF TRUE: %s\n'%get_acc(preftrue['aff'] + preftrue['neg'])
    report += 'AFF: %s\n'%get_acc(preftrue['aff'])
    report += 'NEG: %s\n'%get_acc(preftrue['neg'])
    report += 'PREF TRUE AFF THRESH %s: %s (%s/%s)\n'%(thresh,get_acc(preftrue_thresh['aff']),sum(preftrue_thresh['aff']),len(preftrue_thresh['aff']))
    report += 'PREF TRUE NEG THRESH %s: %s (%s/%s)\n'%(thresh,get_acc(preftrue_thresh['neg']),sum(preftrue_thresh['neg']),len(preftrue_thresh['neg']))
    if lic:
        report += 'PREF TRUE LICENSED: %s\n'%get_acc(preftrue_l['aff'] + preftrue_l['neg'])
        report += 'AFF: %s\n'%get_acc(preftrue_l['aff'])
        report += 'NEG: %s\n'%get_acc(preftrue_l['neg'])
        report += 'PREF TRUE UNLICENSED: %s\n'%get_acc(preftrue_u['aff'] + preftrue_u['neg'])
        report += 'AFF: %s\n'%get_acc(preftrue_u['aff'])
        report += 'NEG: %s\n'%get_acc(preftrue_u['neg'])
    report += '\n\n'

    return report
