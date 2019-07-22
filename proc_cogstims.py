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

def process_fk(infile,tokenizer):
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
            #m = re.match('\(([H|L])\) (.+)\s(.*)\/(.*)\/(.*)\.',line.strip())
            #m2 = re.match('\([H|L]\) (.+)\.(.+)\s.*\/.*\/.*\.',line.strip())
            it,sent1,sent2,exp,wc,bc,constraint = line.strip().split('\t')
            if it == 'item': continue
            #exp,wc,bc = m.group(3,4,5)
            #sent1,sent2 = m2.group(1,2)
            #remove period from s1 for shuffling purposes
            sent1l = re.sub('\.','',sent1).split()
            sent2l = sent2.split()
            #origs1 = copy.copy(sent1)
            #origs2 = copy.copy(sent2)
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
                try:
                    tokenizer.convert_tokens_to_ids([tgt])
                except:
                    print('OOV: %s'%tgt)
            # cleancsv.append(','.join([str(it),str(origsent),exp,wc,bc,m.group(1)]))
    # with open(infile+'-clean.csv','wb') as f:
    #     f.write('item,context,expected,within_category,between_category,constraint')
    #     for line in cleancsv:
    #         print(line)
    #         f.write(line + '\n')
    # written = []
    # with open(re.sub('(\.txt)|(\.csv)','',infile)+'-1perline.txt','w') as f:
    #     for it in inputlist:
    #         if it in written:
    #             continue
    #         written.append(it)
    #         f.write(it + ' . \n')
    return hldict,inputlist,inputlist_shuf,inputlist_nw,inputlist_shufnw,tgtlist

# def process_rr_rawcloze(clozefile):
#     clozedict = {}
#     itemlist = []
#     with open(clozefile,'rU') as f:
#         for line in f:
#             linesplit = line.strip().split(',')
#             itm = linesplit[0].strip()
#             if itm == 'item': continue
#             if itm not in itemlist: itemlist.append(itm)
#             cond = linesplit[1].strip()
#             sent = linesplit[2].strip()
#             completions = [e.strip() for e in linesplit[3:]]
#             if itm not in clozedict:
#                 clozedict[itm] = {}
#             clozedict[itm][cond] = {}
#             clozedict[itm][cond]['sent'] = sent
#             tot_resps = float(len(completions))
#             compl_counts = Counter(completions)
#             compl_cloze = {k:compl_counts[k]/tot_resps for k in compl_counts}
#             sorted_cloze = sorted(compl_cloze.items(),key=lambda x:x[1])
#             m = max([c for w,c in sorted_cloze])
#             top_preds = [(w,c) for w,c in sorted_cloze if c == m]
#             clozedict[itm][cond]['cloze'] = compl_cloze
#             clozedict[itm][cond]['top'] = top_preds
#
#     return clozedict

def process_rr(csvfile,tokenizer,gen_obj=False,gen_subj=False):
    inputlist = []
    tgtlist = []
    clozedict = {}
    clozelist = []
    i = 0
    # cleancsv = []
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
            # exp = [w for w,c in rawcloze[itemnum][condition]['top']]
            inputlist.append(masked_sent)
            tgtlist.append(tgt)
            clozedict[i]['sent'] = masked_sent
            clozedict[i]['tgt'] = tgt
            clozedict[i]['cond'] = condition
            clozedict[i]['item'] = itemnum
            clozedict[i]['tgtcloze'] = tgtcloze
            clozedict[i]['tgtcloze_strict'] = tgtcloze_strict
            # clozedict[i]['fuzexp'] = fuzexp
            clozedict[i]['exp'] = exp
            clozelist.append(maxcloze)
            i += 1
            try:
                tokenizer.convert_tokens_to_ids([tgt])
            except:
                print('OOV: %s'%tgt)
            for e in exp:
                try:
                    tokenizer.convert_tokens_to_ids([e.split()[0]])
                except:
                    print('OOV: %s -- %s'%(e,exp))
            # try:
            #     cleantgtcloze = rawcloze[itemnum][condition]['cloze'][tgt]
            # except:
            #     cleantgtcloze = 0.
            # cleanmaxcloze = rawcloze[itemnum][condition]['cloze'][exp[0]]
    #         exp = [e for e in exp if not re.match('.+\(pass\)',e) and not re.match('been',e)]
    #         print(exp)
    #         cleancsv.append('\t'.join(['%s-%s'%(itemnum,condition),sent,'|'.join(exp),str(maxcloze),tgt,str(tgtcloze),str(tgtcloze_strict)]))
    # with open(csvfile+'-clean.tsv','wb') as f:
    #     # f.write('\t'.join(['item','context','expected','exp_cloze','target','tgt_cloze','tgt_cloze(strict)']) + '\n')
    #     for line in cleancsv:
    #         f.write(line + '\n')
    return clozedict,inputlist,tgtlist,clozelist

def process_fischler(infile,tokenizer):
    nkdict = {}
    inputlist = []
    tgtlist = []
    it = 0
    i = 0
    csvclean = []
    with open(infile,'rU') as f:
        for line in f:
            # ta,fn,fa,tn,aff_tgt,neg_tgt,_ = [e.strip() for e in line.strip().split(',')]
            it,affsent,negsent,afftgt,negtgt = [e.strip() for e in line.strip().split('\t')]
            affsent = re.sub(' \(.+\)','',affsent)
            negsent = re.sub(' \(.+\)','',negsent)
            # for cond,condsent in [('TA',ta),('TN',tn),('FA',fa),('FN',fn)]:
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
                    # affsent = ' '.join(sent)
                else:
                    nkdict[i]['exp'] = [negtgt]
                    # negsent = ' '.join(sent)
                inputlist.append(masked_sent)
                tgtlist.append(tgt)
                i += 1
            for t in afftgt,negtgt:
                try:
                    tokenizer.convert_tokens_to_ids([t])
                except:
                    print('OOV: %s'%t)
    #         affsent = affsent.split()
    #         affsent.pop()
    #         affsent = ' '.join(affsent) + ' [a|an]'
    #         negsent = negsent.split()
    #         negsent.pop()
    #         negsent = ' '.join(negsent) + ' [a|an]'
    #         csvclean.append('\t'.join([str(it),affsent,negsent,afftgt,negtgt]))
    # with open(infile+'-clean.tsv','wb') as f:
    #     f.write('\t'.join(['item','context_pos','context_neg','target_pos','target_neg']) + '\n')
    #     for line in csvclean:
    #         f.write(line + '\n')
    return inputlist,nkdict,tgtlist

def process_nk(infile,tokenizer):
    nkdict = {}
    inputlist = []
    tgtlist = []
    it = 0
    i = 0
    # csvclean = []
    with open(infile,'rU') as f:
        for line in f:
            # c,ta,tn,fa,fn,aff_tgt,neg_tgt = [e.strip() for e in line.strip().split('&&')]
            it,affsent,negsent,afftgt,negtgt,lic = [e.strip() for e in line.strip().split('\t')]
            # for cond,condsent in [('TA',ta),('TN',tn),('FA',fa),('FN',fn)]:
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
                    # affsent = ' '.join(sent)
                else:
                    nkdict[i]['exp'] = [negtgt]
                    # negsent = ' '.join(sent)
                inputlist.append(masked_sent)
                tgtlist.append(tgt)
                i += 1
            for t in afftgt,negtgt:
                try:
                    tokenizer.convert_tokens_to_ids([t])
                except:
                    print('OOV: %s'%t)
    # with open(infile+'-clean.tsv','wb') as f:
    #     f.write('\t'.join(['item','context_pos','context_neg','target_pos','target_neg','licensing']) + '\n')
    #     for line in csvclean:
    #         f.write(line + '\n')
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
        if len(set(hldict[i]['exp']) & set(oov_list)) > 0:
            print('\n\nSKIPPING THIS')
            print(hldict[i]['exp'])
            continue
        score = 0
        # logfile.write(hldict[i]['sent'])
        # logfile.write(str(hldict[i]['exp']) + '\n')
        for subpr in pr:
            # logfile.write(str(subpr) + '\n')
            for candidate in subpr:
                if candidate.strip() in hldict[i]['exp']:
                    score = 1
        if score == 1:
            ctup = (hldict[i]['sent'][setting],hldict[i]['exp'],pr,hldict[i]['constraint'])
            if ctup not in correct:
                correct.append(ctup)
        tot_score.append(score)
        by_constraint_score[hldict[i]['constraint']].append(score)
        # logfile.write(str(top_probs[i])+ '\n')
        # logfile.write('---\n')
    n4report = sim_fk_N400(hldict,tok_preds,top_probs,tok_probs,fklog,setting,k=k,bert=bert)
    tot_acc = get_acc(tot_score)
    report = '\nPrediction accuracies:\n'
    report += 'EXP TGT in TOP %s preds: %s\n'%(k,tot_acc)
    report += 'in TOP %s for H: %s\n'%(k,get_acc(by_constraint_score['H']))
    report += 'in TOP %s for L: %s\n'%(k,get_acc(by_constraint_score['L']))
    return report,n4report,correct,tot_acc,oov_list

def sim_fk_N400(hldict,tok_preds,top_probs,tok_probs,logfile,setting,k=5,bert=True):
    # for s in inputlist:
    #     print(s)
    # tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    # tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    for i,prob in enumerate(tok_probs):
        hldict[i]['tgtprob'] = prob
        hldict[i]['toppreds'] = tok_preds[i]
        hldict[i]['topprobs'] = top_probs[i]
    conddict = make_conddict(hldict)
    probtrips = []
    stimtrips = []
    constrips = []
    toppreds = []
    topprobs = []
    for it in conddict:
        probtrips.append((conddict[it]['exp']['tgtprob'][0],conddict[it]['wc']['tgtprob'][0],conddict[it]['bc']['tgtprob'][0]))
        constrips.append(conddict[it]['exp']['constraint'])
        toppreds.append(conddict[it]['exp']['toppreds'])
        topprobs.append(conddict[it]['exp']['topprobs'])
        stimtrips.append(conddict[it]['exp']['sent'][setting] + ' ' + '/'.join([conddict[it]['exp']['tgt'],conddict[it]['wc']['tgt'],conddict[it]['bc']['tgt']]))
    thresh = 0.01
    exp_top = {}
    exp_top['H'] = []
    exp_top['L'] = []
    exp_top_thresh = {}
    exp_top_thresh['H'] = []
    exp_top_thresh['L'] = []
    wc_boost = {}
    wc_boost['H'] = []
    wc_boost['L'] = []
    allprobs = {}
    allprobs['H'] = []
    allprobs['L'] = []
    for i,pair in enumerate(probtrips):
        invoc = (pair[0] and pair[1] and pair[2])
        logfile.write(u' '.join(stimtrips[i]).encode('utf-8') + '\n')
        logfile.write('TGT probs: %s\n'%list(pair))
        logfile.write('PREDICTED: %s\n'%toppreds[i])
        logfile.write(str(topprobs[i]) + '\n')
        logfile.write(constrips[i]+ '\n')
        cons = constrips[i]
        allprobs[cons].append(pair)
        if (pair[0] > pair[1]) and (pair[0] > pair[2]):
            exp_top[cons].append(1)
            logfile.write('EXP TOP\n')
            if abs(pair[0] - pair[1]) > thresh and abs(pair[0] - pair[2]) > thresh:
                exp_top_thresh[cons].append(1)
            else:
                exp_top_thresh[cons].append(0)
        else:
            exp_top[cons].append(0)
            exp_top_thresh[cons].append(0)
        if (pair[1] > pair[2]) and abs(pair[1] - pair[2]) > thresh:
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
    q1_corr = []
    q2_corr = []
    q3_corr = []
    q4_corr = []
    avgcloze = np.average(clozelist)
    q1 = np.percentile(clozelist,25)
    q3 = np.percentile(clozelist,75)
    q2 = np.percentile(clozelist,50)
    q4 = max(clozelist)
    predcounts = Counter()
    for i,pred in enumerate(tok_preds):
        score = 0
        print(clozedict[i]['sent'])
        itm = clozedict[i]['item']
        cond = clozedict[i]['cond']
        # print(rawclozedict[itm][cond]['top'])
        for subpr in pred:
            predcounts.update(subpr)
            print(subpr)
            print(clozedict[i]['exp'])
            # rawclozevals = []
            for candidate in subpr:
                # if rawclozedict:
                #     if candidate.strip() in rawclozedict[itm][cond]['cloze']:
                #         rawclozevals.append(rawclozedict[itm][cond]['cloze'][candidate.strip()])
                #     else:
                #         rawclozevals.append(0.)
                if candidate.strip() in [e.split()[0] for e in clozedict[i]['exp']]:
                    score = 1
                    print('WINNER\n')
        #     print(rawclozevals)
        # print(top_probs[i])
        # print('---')
        if score == 1:
            ctup = (clozedict[i]['sent'],clozedict[i]['exp'],pred,clozedict[i]['maxcloze'])
            if ctup not in correct:
                correct.append(ctup)
        tot_score.append(score)
        # clozethresh = .4
        if clozedict[i]['maxcloze'] <= q1:
            q1_corr.append(score)
        elif clozedict[i]['maxcloze'] <= q2:
            q2_corr.append(score)
        elif clozedict[i]['maxcloze'] <= q3:
            q3_corr.append(score)
        else:
            q4_corr.append(score)

    n4report = sim_rr_N400(clozedict,tok_preds,top_probs,tok_probs,rrlog,scat=scat,k=k,bert=bert)
    report = '\nPrediction accuracies:\n'
    report += 'TGT in top %s preds: %s\n'%(k,get_acc(tot_score))
    # report += 'Tgt in top %s for HC: %s (%s thresh, %s items)\n'%(k,get_acc(hc_corr),clozethresh,len(hc_corr))
    # report += 'Tgt in top %s for LC: %s (%s thresh, %s items)\n'%(k,get_acc(lc_corr),clozethresh,len(lc_corr))
    report += 'TGT in top %s for Q1: %s (%s upper, %s items)\n'%(k,get_acc(q1_corr),q1,len(q1_corr))
    report += 'TGT in top %s for Q2: %s (%s upper, %s items)\n'%(k,get_acc(q2_corr),q2,len(q2_corr))
    report += 'TGT in top %s for Q3: %s (%s upper, %s items)\n'%(k,get_acc(q3_corr),q3,len(q3_corr))
    report += 'TGT in top %s for Q4: %s (%s upper, %s items)\n'%(k,get_acc(q4_corr),q4,len(q4_corr))
    report += 'AVG CLOZE: %s\n'%avgcloze
    report += 'MED CLOZE: %s\n'%q2
    return report,n4report,correct,predcounts,oov_list

def sim_rr_N400(clozedict,tok_preds,top_probs,tok_probs,logfile,scat=None,k=5,bert=True):
    # for s in inputlist:
    #     print(s)
    # tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    # tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    for i,prob in enumerate(tok_probs):
        clozedict[i]['tgtprob'] = prob
        clozedict[i]['toppreds'] = tok_preds[i]
        clozedict[i]['topprobs'] = top_probs[i]
    conddict = make_conddict(clozedict)
    probpairs = []
    stimpairs = []
    clozepairs = []
    toppreds = []
    topprobs = []
    for it in conddict:
        probpairs.append((conddict[it]['a']['tgtprob'][0],conddict[it]['b']['tgtprob'][0]))
        clozepairs.append((conddict[it]['a']['tgtcloze'],conddict[it]['b']['tgtcloze']))
        toppreds.append((conddict[it]['a']['toppreds'],conddict[it]['b']['toppreds']))
        topprobs.append((conddict[it]['a']['topprobs'],conddict[it]['b']['topprobs']))
        stimpairs.append((conddict[it]['a']['sent'] + ' ' + conddict[it]['a']['tgt'],conddict[it]['b']['sent'] + ' ' + conddict[it]['b']['tgt']))
        if 'c' in conddict[it]:
            probpairs.append((conddict[it]['d']['tgtprob'][0],conddict[it]['c']['tgtprob'][0]))
            clozepairs.append((conddict[it]['d']['tgtcloze'],conddict[it]['c']['tgtcloze']))
            toppreds.append((conddict[it]['d']['toppreds'],conddict[it]['c']['toppreds']))
            topprobs.append((conddict[it]['d']['topprobs'],conddict[it]['c']['topprobs']))
            stimpairs.append((conddict[it]['d']['sent'] + ' ' + conddict[it]['d']['tgt'],conddict[it]['c']['sent'] + ' ' + conddict[it]['c']['tgt']))
    pattern_thresh = []
    pattern = []
    same = []
    thresh = .01
    for i,pair in enumerate(probpairs):
        logfile.write(str(stimpairs[i][0]) + '\n')
        logfile.write(str(stimpairs[i][1]) + '\n')
        logfile.write('TGT probs: %s\n'%list(pair))
        logfile.write('TGT cloze: %s\n'%list(clozepairs[i]))
        logfile.write('PREDICTED: %s'%toppreds[i][0] + '\n')
        logfile.write(str(topprobs[i][0]) + '\n')
        logfile.write('PREDICTED: %s'%toppreds[i][1] + '\n')
        logfile.write(str(topprobs[i][1]) + '\n')
        if (pair[0] and pair[1]) and (pair[0] > pair[1]) and (abs(pair[0] - pair[1]) > thresh):
            pattern_thresh.append(1)
            logfile.write('PATTERN THRESH\n')
        else:
            pattern_thresh.append(0)
        if (pair[0] and pair[1]) and (pair[0] > pair[1]):
            pattern.append(1)
            logfile.write('PATTERN\n')
        else:
            pattern.append(0)
        if (pair[0] and pair[1]) and (abs(pair[0] - pair[1]) < thresh):
            same.append(1)
            logfile.write('NO DIFF\n')
        else:
            same.append(0)
        logfile.write('----\n\n\n')
    # probdiffs = [e[0] - e[1] for i,e in enumerate(probpairs) if e[0] and e[1]]
    # clozediffs = [e[0] - e[1] for i,e in enumerate(clozepairs) if probpairs[i][0] and probpairs[i][1]]
    probdiffs = [e[0] - e[1] for i,e in enumerate(probpairs)]
    clozediffs = [e[0] - e[1] for i,e in enumerate(clozepairs)]
    if scat:
        plt.scatter(clozediffs,probdiffs)
        plt.ylabel('Model probability differences',fontsize='x-large')
        plt.yticks(fontsize='large')
        plt.xlabel('Cloze differences',fontsize='x-large')
        plt.xticks(fontsize='large')
        plt.ylim(-.2,.7)
        plt.savefig(scat)
        plt.clf()
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

    return report

def test_nkf_acc(nkfdict,inputlist,tgtlist,model,tokenizer,nkflog,k=5,bert=True):
    correct = []
    tot_score = []
    tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    for i,pred in enumerate(tok_preds):
        score = 0
        # print(nkfdict[i]['sent'])
        # print(nkfdict[i]['exp'])
        for subpr in pred:
            # print(subpr)
            rawclozevals = []
            for candidate in subpr:
                if candidate.strip() in nkfdict[i]['exp']:
                    score = 1
        # print(top_probs[i])
        # print('---')
        if score == 1:
            ctup = (nkfdict[i]['sent'],nkfdict[i]['exp'],pred,nkfdict[i]['cond'])
            if ctup not in correct:
                correct.append(ctup)
        if nkfdict[i]['cond'] == 'TA':
            tot_score.append(score)
            if not score:
                print('\n\nWRONG')
                print(nkfdict[i]['sent'])
                print(nkfdict[i]['exp'])
                print(pred)

    n4report = sim_nkf_N400(nkfdict,tok_preds,top_probs,tok_probs,nkflog,k=k,bert=bert)
    report = "\nPrediction 'accuracy':\n"
    report += 'TRUE TGT in top %s preds: %s\n'%(k,get_acc(tot_score))


    return report,n4report,correct,oov_list

def sim_nkf_N400(clozedict,tok_preds,top_probs,tok_probs,logfile,k=5,bert=True):
    # tok_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    # tok_probs,oov_list = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)
    for i,prob in enumerate(tok_probs):
        clozedict[i]['tgtprob'] = prob
        clozedict[i]['toppreds'] = tok_preds[i]
        clozedict[i]['topprobs'] = top_probs[i]
    conddict = make_conddict(clozedict)
    probpairs = []
    stimpairs = []
    licensing = []
    lictoggle = False
    toppreds = []
    topprobs = []
    for it in conddict:
        probpairs.append([(conddict[it]['TA']['tgtprob'][0],conddict[it]['FA']['tgtprob'][0]),(conddict[it]['TN']['tgtprob'][0],conddict[it]['FN']['tgtprob'][0])])
        toppreds.append([(conddict[it]['TA']['toppreds'],conddict[it]['FA']['toppreds']),(conddict[it]['TN']['toppreds'],conddict[it]['FN']['toppreds'])])
        topprobs.append([(conddict[it]['TA']['topprobs'],conddict[it]['FA']['topprobs']),(conddict[it]['TN']['topprobs'],conddict[it]['FN']['topprobs'])])
        stimpairs.append([(conddict[it]['TA']['sent'] + ' ' + conddict[it]['TA']['tgt'],conddict[it]['FA']['sent'] + ' ' + conddict[it]['FA']['tgt']),(conddict[it]['TN']['sent'] + ' ' + conddict[it]['TN']['tgt'],conddict[it]['FN']['sent'] + ' ' + conddict[it]['FN']['tgt'])])
        if 'licensing' in conddict[it]['TA']:
            licensing.append(conddict[it]['TA']['licensing'])
            lictoggle = True
    pattern = []
    same = []
    thresh = 0
    preftrue = {}
    preftrue[0] = []
    preftrue[1] = []
    preftrue_l = {}
    preftrue_l[0] = []
    preftrue_l[1] = []
    preftrue_u = {}
    preftrue_u[0] = []
    preftrue_u[1] = []
    for i,pair in enumerate(probpairs):
        if lictoggle:
            lic = licensing[i]
        for j,subpair in enumerate(pair):
            logfile.write(str(stimpairs[i][j][0]) + '\n')
            logfile.write(str(stimpairs[i][j][1]) + '\n')
            logfile.write(u'TGT probs: %s\n'%list(subpair))
            logfile.write(u'PREDICTED: %s'%toppreds[i][j][0] + '\n')
            logfile.write(str(topprobs[i][j][0]) + '\n')
            logfile.write(u'PREDICTED: %s'%toppreds[i][j][1] + '\n')
            logfile.write(str(topprobs[i][j][1]) + '\n')
            logfile.write(u'---\n\n\n')
            if subpair[0] > subpair[1]:
                score = 1
            else:
                score = 0
            preftrue[j].append(score)
            if lictoggle:
                if lic == 'Y':
                    preftrue_l[j].append(score)
                elif lic == 'N':
                    preftrue_u[j].append(score)
                else:
                    print('WRONG')

    # probdiffs = [e[0] - e[1] for i,e in enumerate(probpairs) if e[0] and e[1]]
    report = '\nPreference for true vs false sentences:\n'
    report += 'PREF TRUE: %s\n'%get_acc(preftrue[0] + preftrue[1])
    report += 'AFF: %s\n'%get_acc(preftrue[0])
    report += 'NEG: %s\n'%get_acc(preftrue[1])
    if lictoggle:
        report += 'PREF TRUE LICENSED: %s\n'%get_acc(preftrue_l[0] + preftrue_l[1])
        report += 'AFF: %s\n'%get_acc(preftrue_l[0])
        report += 'NEG: %s\n'%get_acc(preftrue_l[1])
        report += 'PREF TRUE UNLICENSED: %s\n'%get_acc(preftrue_u[0] + preftrue_u[1])
        report += 'AFF: %s\n'%get_acc(preftrue_u[0])
        report += 'NEG: %s\n'%get_acc(preftrue_u[1])
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
                # n4report = sim_fk_N400(hldict,inputlist,tgtlist,model,tokenizer,fklog,logcode,k=k,bert=bert)
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
        # outstring +='OOV\n'
        # for w in oov_list:
        #     outstring += w + '\n'
    return acclist,acclist_names,outstring


def run_rr_all(args,out,models,logcode,klist,clozedict,inputlist,tgtlist,clozelist,bert=True,gen_obj=False,gen_subj=False):
    out.write('\n\n***\nSETTING: %s\n***\n\n'%logcode)
    # rawclozedict = process_rr_rawcloze(args.rr_raw)
    for modelname,model,tokenizer in models:
        out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
        print(modelname)
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

        # out.write('OOV\n')
        # for w in oov_list:
        #     out.write(w + '\n')
    return report,n4report


def run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,bert=True):

    for modelname,model,tokenizer in models:
        out.write('\n\n***\nMODEL: %s\n***\n'%modelname)
        print(modelname)
        reports = []
        for k in klist:
            with open(args.resultsdir+'/FS_predlog_%s-%s'%(modelname,k),'wb') as fslog:
                print('FISCHLER k=%s\n'%k)
                report,n4report,corr,oov_list = test_nkf_acc(negdict,inputlist,tgtlist,model,tokenizer,fslog,k=k,bert=bert)
                # n4report = sim_nkf_N400(fsdict,inputlist,tgtlist,model,tokenizer,fslog,k=k,bert=bert)
                for crritem in corr:
                    fslog.write(str(crritem) + '\n')
                reports.append((report,n4report,k))
        for acc,n4,k in reports:
            out.write('\nFISCHLER k=%s acc\n'%k)
            out.write(acc)
        out.write('\nFISCHLER N400\n')
        out.write(n4)
        out.write('\n----\n\n')
        # out.write('OOV\n')
        # for w in oov_list:
        #     out.write(w + '\n')

        # reports = []
        # for k in klist:
        #     with open(args.resultsdir+'/NK_predlog_%s-%s'%(modelname,k),'wb') as nklog:
        #         print('NIEUWLAND k=%s\n'%k)
        #         report,n4report,corr,oov_list = test_nkf_acc(nkdict,inputlist_nk,tgtlist_nk,model,tokenizer,nklog,k=k,bert=bert)
        #         # n4report = sim_nkf_N400(nkdict,inputlist,tgtlist,model,tokenizer,nklog,bert=bert)
        #         for crritem in corr:
        #             nklog.write(str(crritem) + '\n')
        #         reports.append((report,n4report,k))
        # for acc,n4,k in reports:
        #     out.write('NIEUWLAND k=%s acc\n'%k)
        #     out.write(acc)
        # out.write('\nNIEUWLAND N400\n')
        # out.write(n4)
        # out.write('\n----\n\n')
        # # out.write('OOV\n')
        # # for w in oov_list:
        # #     out.write(w + '\n')

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
        for i in range(3):
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

    # with open(args.resultsdir+'/results-rr.txt','wb') as out:
    #     run_rr_all(args,out,models,'orig',klist,bert=bert,gen_obj=False,gen_subj=False)
    #     run_rr_all(args,out,models,'obj',klist,bert=bert,gen_obj=True,gen_subj=False)
    #     run_rr_all(args,out,models,'subj',klist,bert=bert,gen_obj=False,gen_subj=True)
    #     run_rr_all(args,out,models,'obsub',klist,bert=bert,gen_obj=True,gen_subj=True)
    #
    # with open(args.resultsdir+'/results-neg.txt','wb') as out:
    #     run_neg_all(args,out,models,klist,bert=bert)

#runs all three datasets without any perturbations from paper
def run_three_orig(args,models,klist,bert=True):
    with open(args.resultsdir+'/results-neg.txt','wb') as out:
        inputlist,negdict,tgtlist = process_fischler(args.fisch_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,bert=bert)
        inputlist,negdict,tgtlist = process_nk(args.nk_stim)
        run_neg_all(args,out,models,klist,inputlist,negdict,tgtlist,bert=bert)

    with open(args.resultsdir+'/results-rr.txt','wb') as out:
        clozedict,inputlist,tgtlist,clozelist = process_rr(args.rr_stim,gen_obj=False,gen_subj=False)
        # run_rr_all(args,out,models,'orig',klist,clozedict,inputlist,tgtlist,clozelist,bert=bert)

    with open(args.resultsdir+'/results-fk.txt','wb') as out:
        hldict,inputlist,_,_,_,tgtlist = process_fk(args.fk_stim)
    #     _,_,outstring = run_fk_all(args,out,models,'orig',klist,hldict,inputlist,tgtlist,bert=bert)
    #     out.write(outstring)


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
    # bert_large,tokenizer_large = tp.load_model(args.bertlarge)


    klist = [1,5]

    # models = [('bert-base-uncased',bert_base,tokenizer_base),('bert-large-uncased',bert_large,tokenizer_large)]
    # models = [('bert-base-uncased',bert_base,tokenizer_base)]
    models = []

    print('RUNNING EXPERIMENTS')
    run_three_orig(args,models,klist,bert=True)
    # run_aux_tests(args,models,klist,bert=True)

    # modelnames = ['bert-base-uncased','bert-large-uncased']
    # models = []
    # for mn in modelnames:
    #     print('LOADING %s'%mn)
    #     model,tokenizer = tp.load_model(mn)
    #     inputlist = []
    #     with open(args.manual) as manual_inputs:
    #         for line in manual_inputs:
    #             inputlist.append(line.strip())
    #     tok_preds,tok_probs = tp.get_predictions(inputlist,model,tokenizer,bert=True)
        # for i,sent in enumerate(inputlist):
        #     print(sent)
        #     print(tok_preds[i])
        #     print(tok_probs[i])
        #     print('---')
    # inputlist = [
    # 'the camper reported which girl the bear had [MASK] .',
    # 'the restaurant owner forgot which waitress the customer had [MASK] .'
    # ]
    # tp.get_attention(inputlist,model1,tokenizer1,bert=True)
    # tok_preds,tok_probs = tp.get_predictions(inputlist,model1,tokenizer1,bert=True)
    # for i,sent in enumerate(inputlist):
    #     print(sent)
    #     print(tok_preds[i])
    #     print(tok_probs[i])
    #     print('---')
