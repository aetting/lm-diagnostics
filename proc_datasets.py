import re
import argparse
from io import open

def process_cprag(tsvfile):
    hldict = {}
    inputlist = []
    inputlist_shuf = []
    inputlist_nw = []
    inputlist_shufnw = []
    tgtlist = []
    cleancsv = []
    with open(tsvfile) as f:
        i = 0
        for line in f:
            if len(line.strip('\t')) < 2: continue
            it,sent1,sent2,exp,wc,bc,constraint = line.strip().split('\t')
            if it == 'item': continue
            context = ' '.join([sent1,sent2])
            for tgt,cond in ((exp,'exp'),(wc,'wc'),(bc,'bc')):
                hldict[i] = {}
                inputlist.append(context + ' [MASK]')
                tgtlist.append(tgt)
                hldict[i]['constraint'] = constraint
                hldict[i]['tgt'] = tgt
                hldict[i]['sent'] = {}
                hldict[i]['sent'] = context
                hldict[i]['item'] = it
                hldict[i]['exp'] = [exp]
                hldict[i]['cond'] = cond
                i += 1
    return inputlist,tgtlist,hldict

def process_role(tsvfile,gen_obj=False,gen_subj=False):
    inputlist = []
    tgtlist = []
    clozedict = {}
    clozelist = []
    i = 0
    with open(tsvfile,'rU') as f:
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
    return inputlist,tgtlist,(clozedict,clozelist)

def process_negsimp(tsvfile):
    nkdict = {}
    inputlist = []
    tgtlist = []
    i = 0
    csvclean = []
    with open(tsvfile,'rU') as f:
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
    return inputlist,tgtlist,nkdict

def process_negnat(tsvfile):
    nkdict = {}
    inputlist = []
    tgtlist = []
    i = 0
    with open(tsvfile,'rU') as f:
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
    return inputlist,tgtlist,nkdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_inputs",default=None, type=str)
    parser.add_argument("--cprag_stim", default=None, type=str)
    parser.add_argument("--role_stim", default=None, type=str)
    parser.add_argument("--negsimp_stim", default=None, type=str)
    parser.add_argument("--negnat_stim", default=None, type=str)
    args = parser.parse_args()

    stimlist = [
    (args.cprag_stim, process_cprag,'cprag'),
    (args.role_stim, process_role,'role'),
    (args.negsimp_stim, process_negsimp,'negsimp'),
    (args.negnat_stim, process_negnat,'negnat')
    ]

    for stimfile,process_func,out_pref in stimlist:
        if not stimfile: continue
        inputlist,tgtlist,_ = process_func(stimfile)
        with open(args.model_inputs+'/%s-contextlist'%out_pref,'wb') as out:
            out.write('\n'.join([c.encode('utf-8') for c in inputlist]))
        with open(args.model_inputs+'/%s-completiontlist'%out_pref,'wb') as out:
            out.write('\n'.join(tgtlist))

    # inputlist,tgtlist,_ = process_role(args.role_stim)
    # with open(args.model_inputs+'/role-contextlist','wb') as out:
    #     out.write('\n'.join(inputlist))
    # with open(args.model_inputs+'/role-completiontlist','wb') as out:
    #     out.write('\n'.join(targetlist))
    #
    # inputlist,tgtlist,_ = process_negsimp(args.negsimp_stim)
    # with open(args.model_inputs+'/negsimp-contextlist','wb') as out:
    #     out.write('\n'.join(inputlist))
    # with open(args.model_inputs+'/negsimp-completiontlist','wb') as out:
    #     out.write('\n'.join(targetlist))
    #
    # inputlist,tgtlist,_ = process_negnat(args.negnat_stim)
    # with open(args.model_inputs+'/negnat-contextlist','wb') as out:
    #     out.write('\n'.join(inputlist))
    # with open(args.model_inputs+'/negnat-completiontlist','wb') as out:
    #     out.write('\n'.join(targetlist))
