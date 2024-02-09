#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Copied and adapted from compare-mt by Neulab @ CMU.
"""
from Levenshtein import distance as lev
import math
from collections import Counter


class PyBleuScorer(object):
    
    """
    A scorer that calculates BLEU score.
    """
    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25), case_insensitive=False, filter_indent=False):
        self.weights = weights
        self.case_insensitive = case_insensitive
        self.filter_indent = filter_indent
    
    def accuracy_core(self, ref, out):
        """
        Score a corpus using accuracy
        Args:
            ref: A reference corpus
            out: An output corpus
        Returns:
            A tuple containing a value for the accuracy score and a value for item accuracy score
        """      
        if self.case_insensitive:
            ref = lower(ref)
            out = lower(out)
        if self.filter_indent:
            ref = filter_indent(ref)
            out = filter_indent(out)

        ref = tokenize(ref)
        out = tokenize(out)
        total = 0.
        correct = 0.
        item_correct = 0.
        for r, o in zip(ref, out):
            total += 1
            if r == o:
                correct += 1  
            if set(r) == set(o):
                item_correct += 1      
        return  item_correct / total, correct / total
    
    def following_acc(self, ref, out, n):
        """
        Score a corpus using accuracy
        Args:
            ref: A reference corpus
            out: An output corpus
        Returns:
            A tuple containing a value for the accuracy score and a value for item accuracy score
        """      
        if self.case_insensitive:
            ref = lower(ref)
            out = lower(out)

        if self.filter_indent:
            ref = filter_indent(ref)
            out = filter_indent(out)

        ref = tokenize(ref)
        out = tokenize(out)
        total = 0.
        correct = 0.
        for r, o in zip(ref, out):
            total += 1
            if r[n:] == o[n:]:
                correct += 1  
        return  correct / total
    
    def editsim_core(self, ref, out):
        """
        Score a corpus using edit similarity score
        Args:
            ref: A reference corpus
            out: An output corpus
        Returns:
            A value for the edit similarity score
        """      
        if self.case_insensitive:
            ref = lower(ref)
            out = lower(out)
        
        if self.filter_indent:
            ref = filter_indent(ref)
            out = filter_indent(out)

        ref = tokenize(ref)
        out = tokenize(out)

        total = 0.
        edit_sim_score = 0.
        for r, o in zip(ref, out):
            total += 1
            edit_sim_score += self.compute_edit_sim(r, o)
            
        return edit_sim_score / total

    def compute_edit_sim(self, ref, out):
        ref = ' '.join(ref)
        out = ' '.join(out)
        Ldis = lev(ref, out)
        return 1 - Ldis/(len(ref) + len(out))

    def score(self, ref, out):
        import sacrebleu
        """
        Score a corpus using BLEU score
        Args:
            ref: A reference corpus
            out: An output corpus
        Returns:
            A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
        """
        if self.case_insensitive:
            ref = lower(ref)
            out = lower(out)

        if self.filter_indent:
            ref = filter_indent(ref)
            out = filter_indent(out)

        # ===========
        #return sacrebleu.corpus_bleu(out, [ref], tokenize="none").score
        return sacrebleu.corpus_bleu(out, [ref]).score
        # ==========
        # ref = tokenize(ref)
        # out = tokenize(out)
        # ref_len = 0
        # out_len = 0
        # num_prec = Counter()
        # denom_prec = Counter()
        # for r, o in zip(ref, out):
        #     ref_len += len(r)
        #     out_len += len(o)
        #     for n in range(1, len(self.weights) + 1):
        #         num, denom = self.precision(r, o, n)
        #         num_prec[n] += num
        #         denom_prec[n] += denom

        # if num_prec[1] == 0:
        #     return 0.0
        
        # prec = 0
        # for i, w in enumerate(self.weights, start=1):
        #     p = num_prec[i] / denom_prec[i] if denom_prec[i] != 0 else 0
        #     p = math.log(p) if p > 0 else 0
        #     prec += p * w 
        
        # bp = min(1, math.exp(1 - ref_len/out_len)) if out_len != 0 else 0
        
        # return 100.0 * bp * math.exp(prec)

    
    def precision(self, ref, out, n):
        """
        Caculate n-gram precision 
        Args:
            ref: A reference sentence
            out: An output sentence
        Returns:
            Numerator and denominator of the precision
        """
        ref_ngram = sent_ngrams_list(ref, n)
        ref_cnt = Counter(ref_ngram)
        out_ngram = sent_ngrams_list(out, n)
        out_cnt = Counter(out_ngram)

        num = 0
        denom = 0
        for ngram, o_cnt in out_cnt.items():
            num += min(o_cnt, ref_cnt[ngram])
            denom += o_cnt
        denom = max(1, denom)

        return num, denom
   

def lower(inp):
    return inp.lower() if type(inp) == str else [lower(x).replace('<<unk>>','<unk>') for x in inp]

def filter_indent(inp):
    return inp.replace('<indent>','').strip() if type(inp) == str else [x.replace('<indent>','').strip() for x in inp]

def tokenize(corpus):
    return [list(sent.strip().split()) for sent in corpus]


def sent_ngrams_list(words, n):
    """
    Create a list with all the n-grams in a sentence
    Arguments:
    words: A list of strings representing a sentence
    n: The ngram length to consider
    Returns:
    A list of n-grams in the sentence
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

