from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
from overrides import overrides

from collections import namedtuple
from collections import defaultdict

class F1_abc(object):
    def __init__(self):
        self.A = 0
        self.B = 0
        self.C = 0
        self.ABC = namedtuple('ABC', ['A', 'B', 'C'])
        self.rel_detail = defaultdict(self.ABC(A=0, B=0, C=0))

    def reset(self) -> None:
        self.A = 0
        self.B = 0
        self.C = 0
        self.rel_detail.clear()

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        result = self.calc(self.A, self.B, self.C)
        return result

    @staticmethod
    def calc(A, B, C):
        p = A / B if B > 0 else 0.
        r = A / C if C > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p+r) > 0 else 0.
        prf1 = {"precision": p, "recall": r, "fscore": f1}
        return prf1

    def get_metric_detail(self, reset: bool = False):
        if reset:
            self.reset()
        results = {}
        for k, v in self.rel_detail:
            results[k] = self.calc(v.A, v.B, v.C)
        return results

    def __call__(self, predictions,
                 gold_labels):
        raise NotImplementedError


class F1_triplet(F1_abc):

    @overrides
    def __call__(self, predictions: List[List[Dict[str, str]]],
                 gold_labels: List[List[Dict[str, str]]]):

        for g, p in zip(gold_labels, predictions):
            try:
                g_set = set('_'.join((gg['object'], gg['predicate'],
                                    gg['subject'])) for gg in g)
                p_set = set('_'.join((pp['object'], pp['predicate'],
                                    pp['subject'])) for pp in p)
            except:
                g_set = set('_'.join((''.join(gg['object']), gg['predicate'],
                                    ''.join(gg['subject']))) for gg in g)
                p_set = set('_'.join((''.join(pp['object']), pp['predicate'],
                                    ''.join(pp['subject']))) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)

            # for rel detail
            g_set_rel, p_set_rel = {}, {}
            try:
                g_set_tmp = [(gg['predicate'], '_'.join((gg['object'], gg['predicate'], gg['subject']))) for gg in g]
                p_set_tmp = [(pp['predicate'], '_'.join((pp['object'], pp['predicate'], pp['subject']))) for pp in p]
            except:
                g_set_tmp = [(gg['predicate'], '_'.join((''.join(gg['object']), gg['predicate'], ''.join(gg['subject'])))) for gg in g]
                p_set_tmp = [(pp['predicate'], '_'.join((''.join(pp['object']), pp['predicate'], ''.join(pp['subject'])))) for pp in p]

            for elem in g_set_tmp:
                if elem[0] in g_set_rel:
                    g_set_rel[elem[0]] |= set(elem[1])
                else:
                    g_set_rel[elem[0]] = set(elem[1])
            for elem in p_set_tmp:
                if elem[0] in p_set_rel:
                    p_set_rel[elem[0]] |= set(elem[1])
                else:
                    p_set_rel[elem[0]] = set(elem[1])

            rel = set(g_set_tmp.keys()) | set(p_set_tmp.keys())
            for k in rel:
                self.rel_detail[k] = self.ABC(0, 0, 0)

            for k in rel:
                vg, vp = g_set_tmp.get(k, set([])), p_set_tmp.get(k, set([]))
                self.rel_detail[k].A += len(vg & vp)
                self.rel_detail[k].B + len(vg & vp)
                self.rel_detail[k].C + len(vg & vp)


class F1_ner(F1_abc):

    @overrides
    def __call__(self, predictions: List[List[str]], gold_labels: List[List[str]]):
        for g, p in zip(gold_labels, predictions):

            inter = sum(tok_g == tok_p and tok_g in ('B', 'I')
                        for tok_g, tok_p in zip(g, p))
            bi_g = sum(tok_g in ('B', 'I') for tok_g in g)
            bi_p = sum(tok_p in ('B', 'I') for tok_p in p)

            self.A += inter
            self.B += bi_p
            self.C += bi_g
