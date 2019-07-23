import re
import os
import argparse
import random
import copy
import numpy as np
import test_pyt as tp
import scipy
import scipy.stats
from io import open
from collections import Counter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

def process_fk(infile):
    hldict = {}
    inputlist = []
    inputlist_shuf = []
    inputlist_nw = []
    inputlist_shufnw = []
    tgtlist = []
    cleancsv = []
    with open(infile) as f:
        i = 0
        for line in f:
            if len(line.strip('\t')) < 2: continue
            it,sent1,sent2,exp,wc,bc,constraint = line.strip().split('\t')
            if it == 'item': continue
            sent1l = re.sub('\.','',sent1).split()
            sent2l = sent2.split()
            random.shuffle(sent1l)
            sent2l = sent2l[-2:]
            origsent = ' '.join([sent1,sent2])
            shuf = ' '.join(sent1l) + '. ' + sent2
            nw = sent1 + ' ' + ' '.join(sent2l)
            shufnw = ' '.join(sent1l) + '. ' + ' '.join(sent2l)

            for tgt,cond in ((exp,'exp'),(wc,'wc'),(bc,'bc')):
                hldict[i] = {}
                inputlist.append(origsent + ' [MASK]')
                inputlist_shuf.append(shuf + ' [MASK]')
                inputlist_nw.append(nw + ' [MASK]')
                inputlist_shufnw.append(shufnw + ' [MASK]')
                tgtlist.append(tgt)
                hldict[i]['constraint'] = constraint
                hldict[i]['tgt'] = tgt
                hldict[i]['sent'] = {}
                hldict[i]['sent']['orig'] = origsent
                hldict[i]['sent']['shuf'] = shuf
                hldict[i]['sent']['nw'] = nw
                hldict[i]['sent']['shufnw'] = shufnw
                hldict[i]['item'] = it
                hldict[i]['exp'] = [exp]
                hldict[i]['cond'] = cond
                i += 1
    return hldict,inputlist,inputlist_shuf,inputlist_nw,inputlist_shufnw,tgtlist

def process_rr(csvfile,gen_obj=False,gen_subj=False):
    inputlist = []
    tgtlist = []
    clozedict = {}
    clozelist = []
    i = 0
    with open(csvfile,'rU') as f:
        for line in f:
            linesplit = line.strip().split('\t')
            item = linesplit[0]
            if item == 'item' or len(linesplit) < 1:
                continue
            itemnum,condition = item.split('-')
            sent = linesplit[1]
            exp = linesplit[2].split('|')
            maxcloze = float(linesplit[3])
            tgt = linesplit[4].strip().split()[0]
            tgtcloze = float(linesplit[5])
            tgtcloze_strict = float(linesplit[6])
            if gen_obj:
                sent = re.sub('which .* the','which one the',sent)
            if gen_subj:
                sent = re.sub('which (.*) the .* had','which \g<1> the other had',sent)
            masked_sent = sent + ' [MASK]'
            clozedict[i] = {}
            if masked_sent in inputlist:
                for item in clozedict:
                    if clozedict[item]['sent'] == masked_sent:
                        clozedict[i]['maxcloze'] = maxcloze
                        break
            else:
                clozedict[i]['maxcloze'] = maxcloze
            inputlist.append(masked_sent)
            tgtlist.append(tgt)
            clozedict[i]['sent'] = masked_sent
            clozedict[i]['tgt'] = tgt
            clozedict[i]['cond'] = condition
            clozedict[i]['item'] = itemnum
            clozedict[i]['tgtcloze'] = tgtcloze
            clozedict[i]['tgtcloze_strict'] = tgtcloze_strict
            clozedict[i]['exp'] = exp
            clozelist.append(maxcloze)
            i += 1
    return clozedict,inputlist,tgtlist,clozelist

def process_fischler(infile):
    nkdict = {}
    inputlist = []
    tgtlist = []
    it = 0
    i = 0
    csvclean = []
    with open(infile,'rU') as f:
        for line in f:
            it,affsent,negsent,afftgt,negtgt = [e.strip() for e in line.strip().split('\t')]
            affsent = re.sub(' \(.+\)','',affsent)
            negsent = re.sub(' \(.+\)','',negsent)
            if it == 'item': continue
            for sent,tgt,cond in [(affsent,afftgt,'TA'),(negsent,afftgt,'FN'),(affsent,negtgt,'FA'),(negsent,negtgt,'TN')]:
                nkdict[i] = {}
                if re.match('[aeiou]',tgt): det = 'an'
                else: det = 'a'
                masked_sent = ' '.join([sent,det,'[MASK]'])
                nkdict[i]['sent'] = masked_sent
                nkdict[i]['tgt'] = tgt
                nkdict[i]['item'] = it
                nkdict[i]['cond'] = cond
                if cond in ('TA','FA'):
                    nkdict[i]['exp'] = [afftgt]
                else:
                    nkdict[i]['exp'] = [negtgt]
                inputlist.append(masked_sent)
                tgtlist.append(tgt)
                i += 1
    return inputlist,nkdict,tgtlist

def process_nk(infile):
    nkdict = {}
    inputlist = []
    tgtlist = []
    it = 0
    i = 0
    with open(infile,'rU') as f:
        for line in f:
            it,affsent,negsent,afftgt,negtgt,lic = [e.strip() for e in line.strip().split('\t')]
            if it == 'item': continue
            for sent,tgt,cond in [(affsent,afftgt,'TA'),(negsent,afftgt,'FN'),(affsent,negtgt,'FA'),(negsent,negtgt,'TN')]:
                nkdict[i] = {}
                masked_sent = sent + ' [MASK]'
                nkdict[i]['sent'] = masked_sent
                nkdict[i]['tgt'] = tgt
                nkdict[i]['item'] = it
                nkdict[i]['cond'] = cond
                nkdict[i]['licensing'] = lic
                if cond in ('TA','FA'):
                    nkdict[i]['exp'] = [afftgt]
                else:
                    nkdict[i]['exp'] = [negtgt]
                inputlist.append(masked_sent)
                tgtlist.append(tgt)
                i += 1
    return inputlist,nkdict,tgtlist

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


def get_acc(scorelist):
    if len(scorelist) == 0:
        acc = 0
    else:
        acc = float(sum(scorelist))/len(scorelist)
    return acc

def test_fk_acc(hldict,inputlist,tgtlist,model,tokenizer,setting,fklog,k=5,bert=True):
    tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    tot_score = []
    by_constraint_score = {}
    by_constraint_score['H'] = []
    by_constraint_score['L'] = []
    correct = []
    for i,pr in enumerate(tok_preds):
        hldict[i]['toppreds'] = pr
        hldict[i]['tgtprob'] = tok_probs[i]
        hldict[i]['topprobs'] = top_probs[i]
        score = 0
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
    report += 'EXP TGT in TOP %s preds: %s\n'%(k,tot_acc)
    report += 'in TOP %s for H: %s\n'%(k,get_acc(by_constraint_score['H']))
    report += 'in TOP %s for L: %s\n'%(k,get_acc(by_constraint_score['L']))
    return report,n4report,correct,tot_acc,oov_list

def sim_fk_N400(conddict,logfile,setting,k=5,bert=True):
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

def test_rr_acc(clozedict,inputlist,tgtlist,clozelist,model,tokenizer,rrlog,k=5,bert=True,scat=None):
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
    report += 'TGT in top %s preds: %s\n'%(k,get_acc(tot_score))
    report += 'TGT in top %s for Q1: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q1_corr']),q1,len(correct_by_quartile['q1_corr']))
    report += 'TGT in top %s for Q2: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q2_corr']),q2,len(correct_by_quartile['q2_corr']))
    report += 'TGT in top %s for Q3: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q3_corr']),q3,len(correct_by_quartile['q3_corr']))
    report += 'TGT in top %s for Q4: %s (%s upper, %s items)\n'%(k,get_acc(correct_by_quartile['q4_corr']),q4,len(correct_by_quartile['q4_corr']))
    report += 'AVG CLOZE: %s\n'%avgcloze
    report += 'MED CLOZE: %s\n'%q2
    return report,n4report,correct,predcounts,oov_list

def sim_rr_N400(conddict,logfile,scat=None,k=5,bert=True):
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
    avgprobdiff = np.average([e for e in probdiffs if e > 0])
    avgclozediff = np.average([e for e in clozediffs])

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
    report += 'AVG PROB DIFF (when good higher): %s\n'%avgprobdiff
    report += 'AVG CLOZE DIFF (for same items): %s\n'%avgclozediff

    if scat:
        plot_rr_prcl(clozediffs,probdiffs,scat)

    return report

def plot_rr_prcl(clozediffs,probdiffs,scat):
    plt.scatter(clozediffs,probdiffs)
    plt.ylabel('Model probability differences',fontsize='x-large')
    plt.yticks(fontsize='large')
    plt.xlabel('Cloze differences',fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.ylim(-.2,.7)
    plt.savefig(scat)
    plt.clf()

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
    report += 'TRUE TGT in top %s preds: %s\n'%(k,get_acc(tot_score))

    return report,n4report,correct,oov_list

def sim_nkf_N400(conddict,logfile,k=5,bert=True):
    pattern = []
    same = []
    preftrue = {'aff':[],'neg':[]}
    preftrue_l = {'aff':[],'neg':[]}
    preftrue_u = {'aff':[],'neg':[]}
    lic = None
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

    report = '\nPreference for true vs false sentences:\n'
    report += 'PREF TRUE: %s\n'%get_acc(preftrue['aff'] + preftrue['neg'])
    report += 'AFF: %s\n'%get_acc(preftrue['aff'])
    report += 'NEG: %s\n'%get_acc(preftrue['neg'])
    if lic:
        report += 'PREF TRUE LICENSED: %s\n'%get_acc(preftrue_l['aff'] + preftrue_l['neg'])
        report += 'AFF: %s\n'%get_acc(preftrue_l['aff'])
        report += 'NEG: %s\n'%get_acc(preftrue_l['neg'])
        report += 'PREF TRUE UNLICENSED: %s\n'%get_acc(preftrue_u['aff'] + preftrue_u['neg'])
        report += 'AFF: %s\n'%get_acc(preftrue_u['aff'])
        report += 'NEG: %s\n'%get_acc(preftrue_u['neg'])
    report += '\n\n'

    return report

def run_fk_all(args,out,models,logcode,klist,hldict,inputlist,tgtlist,bert=True):
    outstring = ''
    outstring += '\n\n***\nSETTING: %s\n***\n\n'%logcode
    print(logcode)
    acclist = []
    acclist_names = []
    for modelname,model,tokenizer in models:
        outstring += '\n\n***\nMODEL: %s\n***\n'%modelname
        print(modelname)
        reports = []
        for k in klist:
            with open(os.path.join(args.resultsdir,'FK-%s_predlog_%s-%s'%(logcode,modelname,k)),'wb') as fklog:
                report,n4report,corr,acc,oov_list = test_fk_acc(hldict,inputlist,tgtlist,model,tokenizer,logcode,fklog,k=k,bert=bert)
                acclist.append(acc)
                acclist_names.append(modelname + '-%s'%k)
                for crritem in corr:
                    fklog.write(str(crritem) + '\n')
                reports.append((report,n4report,k))
        for acc,n4,k in reports:
            outstring +='\nFED/KUT k=%s acc\n'%k
            outstring += acc
        outstring +='\nFED/KUT N400\n'
        outstring += n4
        outstring +='\n----\n\n'
    return acclist,acclist_names,outstring


def run_rr_all(args,out,models,logcode,klist,clozedict,inputlist,tgtlist,clozelist,bert=True):
    out.write('\n\n***\nSETTING: %s\n***\n\n'%logcode)
    for modelname,model,tokenizer in models:
        out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
        reports = []
        for k in klist:
            with open(os.path.join(args.resultsdir,'RR-%s_predlog_%s-%s'%(logcode,modelname,k)),'wb') as rrlog:
                print('CHOW k=%s'%k)
                # n4report = sim_rr_N400(clozedict,inputlist,tgtlist,model,tokenizer,rrlog,scat=os.path.join(args.resultsdir,'prcl-%s'%modelname),k=k,bert=bert)
                report,n4report,corr,prcounts,oov_list = test_rr_acc(clozedict,inputlist,tgtlist,clozelist,model,tokenizer,rrlog,k=k,bert=bert)
                for crritem in corr:
                    rrlog.write(str(crritem) + '\n')
                rrlog.write('\n'+ str(prcounts))
                reports.append((report,n4report,k))
        for acc,n4,k in reports:
            out.write('\nCHOW k=%s acc\n'%k)
            out.write(acc)
        out.write('\nCHOW N400\n')
        out.write(n4)
        out.write('\n----\n\n')

    return report,n4report


def run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,dataname,logcode,bert=True):

    for modelname,model,tokenizer in models:
        out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
        print(modelname)
        reports = []
        for k in klist:
            with open(args.resultsdir+'/%s_predlog_%s-%s'%(logcode,modelname,k),'wb') as nkflog:
                report,n4report,corr,oov_list = test_nkf_acc(negdict,inputlist,tgtlist,model,tokenizer,nkflog,k=k,bert=bert)
                # n4report = sim_nkf_N400(fsdict,inputlist,tgtlist,model,tokenizer,fslog,k=k,bert=bert)
                for crritem in corr:
                    nkflog.write(str(crritem) + '\n')
                reports.append((report,n4report,k))
        for acc,n4,k in reports:
            out.write('\n%s k=%s acc\n'%(dataname,k))
            out.write(acc)
        out.write('\n%s N400\n'%dataname)
        out.write(n4)
        out.write('\n----\n\n')

def run_weight_mixing():
    ftcode = args.ftcode
    model2,tokenizer2 = tp.mix_weights(args.pretraineddir,args.finetuneddir)

    models = [
    ('pretrained',model1,tokenizer1),
    ('mixed-%s'%ftcode,model2,tokenizer2),
    ]

#runs all three datasets WITH all additional perturbations tried in the paper
def run_aux_tests(args,models,klist,bert=True):
    acclists_shuf = []
    acclists_shufnw = []
    acclists = []
    with open(args.resultsdir+'/results-fk.txt','wb') as out:
        hldict,inputlist,_,inputlist_nw,_,tgtlist = process_fk(args.fk_stim)
        _,_,outstring = run_fk_all(args,out,models,'orig',klist,hldict,inputlist,tgtlist,bert=bert)
        out.write(outstring)
        _,_,outstring = run_fk_all(args,out,models,'nw',klist,hldict,inputlist_nw,tgtlist,bert=bert)
        out.write(outstring)
        for i in range(100):
            _,_,inputlist_shuf,_,inputlist_shufnw,_ = process_fk(args.fk_stim)
            acclist,acclist_names_shuf,_ = run_fk_all(args,out,models,'shuf',klist,hldict,inputlist_shuf,tgtlist,bert=bert)
            acclists_shuf.append(acclist)
            acclist,acclist_names_shufnw,_ = run_fk_all(args,out,models,'shufnw',klist,hldict,inputlist_shufnw,tgtlist,bert=bert)
            acclists_shufnw.append(acclist)
        out.write('\n\nSHUF ACCLISTS')
        out.write(str(acclist_names_shuf) + '\n')
        out.write(str(acclists_shuf))
        out.write('\n\nSHUFNW ACCLISTS')
        out.write(str(acclist_names_shufnw) + '\n')
        out.write(str(acclists_shufnw))

        out.write('\n\nSHUF ACCURACIES\n')
        accs_by_modk = zip(*acclists_shuf)
        i = 0
        for modelname,_,_ in models:
            for k in klist:
                this_accs = accs_by_modk[i]
                out.write('%s k=%s: %s pm %s\n'%(modelname,k,np.average(this_accs),np.std(this_accs)))
                i += 1
        out.write('\n\nSHUF-NW ACCURACIES\n')
        accs_by_modk = zip(*acclists_shufnw)
        i = 0
        for modelname,_,_ in models:
            for k in klist:
                this_accs = accs_by_modk[i]
                out.write('%s k=%s: %s pm %s\n'%(modelname,k,np.average(this_accs),np.std(this_accs)))
                i += 1

    with open(args.resultsdir+'/results-rr.txt','wb') as out:
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=False,gen_subj=False)
        run_rr_all(args,out,models,'orig',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=True,gen_subj=False)
        run_rr_all(args,out,models,'obj',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=False,gen_subj=True)
        run_rr_all(args,out,models,'subj',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=True,gen_subj=True)
        run_rr_all(args,out,models,'obsub',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)

    with open(args.resultsdir+'/results-neg.txt','wb') as out:
        inputlist,negdict,tgtlist = process_fischler(args.fisch_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,'FISCHLER','FS',bert=bert)
        inputlist,negdict,tgtlist = process_nk(args.nk_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,'NIEUWLAND','NK',bert=bert)

#runs all three datasets without any perturbations from paper
def run_three_orig(args,models,klist,bert=True):
    with open(args.resultsdir+'/results-neg.txt','wb') as out:
        inputlist,negdict,tgtlist = process_fischler(args.fisch_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,'FISCHLER','FS',bert=bert)
        inputlist,negdict,tgtlist = process_nk(args.nk_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,'NIEUWLAND','NK',bert=bert)

    with open(args.resultsdir+'/results-rr.txt','wb') as out:
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=False,gen_subj=False)
        run_rr_all(args,out,models,'orig',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)

    with open(args.resultsdir+'/results-fk.txt','wb') as out:
        hldict,inputlist,_,_,_,tgtlist = process_fk(args.fk_stim)
        _,_,outstring = run_fk_all(args,out,models,'orig',klist,hldict,inputlist,tgtlist,bert=bert)
        out.write(outstring)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fk_stim", default=None, type=str)
    # parser.add_argument("--rr_stim", default=None, type=str, nargs='+')
    parser.add_argument("--rr_stim", default=None, type=str)
    # parser.add_argument("--rr_raw", default=None, type=str)
    parser.add_argument("--fisch_stim", default=None, type=str)
    parser.add_argument("--nk_stim", default=None, type=str)
    parser.add_argument("--manual",default=None, type=str)
    parser.add_argument("--resultsdir",default=None, type=str)
    parser.add_argument("--bertbase",default=None, type=str)
    parser.add_argument("--bertlarge",default=None, type=str)
    parser.add_argument("--finetuneddir",default=None, type=str)
    parser.add_argument("--ftcode",default=None, type=str)
    args = parser.parse_args()


    # main(args)

    # model,tokenizer = tp.load_modelGPT()
    # bert = False

    print('LOADING MODELS')
    bert_base,tokenizer_base = tp.load_model(args.bertbase)
    bert_large,tokenizer_large = tp.load_model(args.bertlarge)


    klist = [1,5]

    models = [('bert-base-uncased',bert_base,tokenizer_base),('bert-large-uncased',bert_large,tokenizer_large)]
    # models = [('bert-base-uncased',bert_base,tokenizer_base)]
    # models = []

    print('RUNNING EXPERIMENTS')
    # run_three_orig(args,models,klist,bert=True)
    run_aux_tests(args,models,klist,bert=True)
