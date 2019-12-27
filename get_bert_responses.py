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



def get_model_responses(inputlist,tgtlist,modeliname,model,tokenizer,k=5,bert=True):
    top_preds,top_probs = tp.get_predictions(inputlist,model,tokenizer,k=k,bert=bert)
    tgt_probs = tp.get_probabilities(inputlist,tgtlist,model,tokenizer,bert=bert)

    return top_preds,top_probs,tgt_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir", default=None, type=str)
    parser.add_argument("--bertbase",default=None, type=str)
    parser.add_argument("--bertlarge",default=None, type=str)
    args = parser.parse_args()

    testlist = ['cprag','role', 'negsimp','negnat']

    print('LOADING MODELS')
    bert_base,tokenizer_base = tp.load_model(args.bertbase)
    bert_large,tokenizer_large = tp.load_model(args.bertlarge)


    k = 5

    models = [('bert-base-uncased',bert_base,tokenizer_base),('bert-large-uncased',bert_large,tokenizer_large)]

    for testname in testlist:
        inputlist = []
        tgtlist = []
        with open(os.path.join(args.inputdir,testname+'-contextlist')) as cont:
            for line in cont: inputlist.append(line.strip())
        with open(os.path.join(args.inputdir,testname+'-targetlist')) as comp:
            for line in comp: tgtlist.append(line.strip())

        for modelname,model,tokenizer in models:
            top_preds,top_probs,tgt_probs = get_model_responses(inputlist,tgtlist,modelname,model,tokenizer,k=k)

            with open(args.inputdir+'/modelpreds-%s-%s'%(testname,modelname),'w') as pred_out:
                for i,preds in enumerate(top_preds):
                    pred_out.write(' '.join(preds))
                    pred_out.write('\n')

            with open(args.inputdir+'/modeltgtprobs-%s-%s'%(testname,modelname),'w') as prob_out:
                for i,prob in enumerate(tgt_probs):
                    prob_out.write(str(prob))
                    prob_out.write('\n')
