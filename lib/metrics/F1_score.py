from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
from overrides import overrides

# maybe wrong~, eval for 'char'

class F1_abc(object):
    def __init__(self):
        self.A = 0
        self.B = 0
        self.C = 0

    def reset(self) -> None:
        self.A = 0
        self.B = 0
        self.C = 0

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        p = self.A / self.B if self.B > 0 else 0.
        r = self.A / self.C if self.C > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p+r) > 0 else 0.
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

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
