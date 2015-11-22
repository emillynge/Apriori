import bisect
from itertools import chain, combinations
from collections import Sequence, Iterator


class AprioriSet(tuple):
    __slots__ = []

    def union(self, other_set: tuple, collect_set: set):
        """
        union between two AprioriSets of size k, that adds result into a collection if union yields a set of size k+1
        :param other_set: AprioriSet of size k
        :param collect_set: AprioriSet of size k
        :return:
        """
        lo = 0
        hi = bisect.bisect_right(self, other_set[-1])
        new_i = None
        for i in other_set:
            lo = bisect.bisect_left(self, i, lo=lo, hi=hi)
            if lo == len(self) or self[lo] != i:
                if new_i is not None:
                    return  # break function as this is the second new item to be found
                new_i = i

        if new_i is not None:
            collect_set.add(AprioriSet(chain(self[:lo], (new_i,), self[lo:])))


class AprioriCollection(object):
    __slots__ = ['lists', 'k', 'size', 'counts', 'in_lists']

    def __init__(self, k):
        self.k = k
        self.lists = tuple(list() for _ in range(k))
        self.size = 0
        self.counts = list()
        self.in_lists = set()

    def _index(self, find_set: AprioriSet):
        lo = 0
        hi = self.size - 1

        for k, (l, i) in enumerate(zip(self.lists, find_set)):
            lo = bisect.bisect_left(l, i, lo=lo, hi=hi)      # for i, find lower index

            # if this point is occupied by a different i, the set is not in collection
            if l[lo] != i:
                raise KeyError('set not in collection')
            elif k == self.k - 1:
                return lo   # no more lists to look through. this is the index of out item

            # We cannot yet determine if set is in collection get a high bound and move to next i
            hi = bisect.bisect_right(l, i, lo=lo, hi=hi)
        else:
            return lo # the loop ended without finding a different set => this set already exists in this collection

    def add(self, new_set: AprioriSet):
        if self.size == 0:
            self._insert_set(new_set, 0)
            return

        lo = 0
        hi = self.size

        for l, i in zip(self.lists, new_set):
            lo = bisect.bisect_left(l, i, lo=lo, hi=hi)      # for i, find lower index for insertion
            if lo == self.size or l[lo] != i:                              # if this point is occupied by a different set, do insertion here
                break

            # We cannot yet determine if insertion point is different. get a high bound and move to next i
            hi = bisect.bisect_right(l, i, lo=lo, hi=hi)
        else:
            return  # the loop ended without finding a different set => this set already exists in this collection

        self._insert_set(new_set, lo)

    def _insert_set(self, new_set, idx):
        # We found an insertion point with different i. This set is not in collection and should be inserted
        for l, i in zip(self.lists, new_set):
            l.insert(idx, i)
        self.size += 1
        self.in_lists.update(iter(new_set))

    def merge(self, other_collection):
        assert isinstance(other_collection, AprioriCollection)
        self.lists = list(zip(*self._merge_generator(self.lists, other_collection.lists)))
        self.size = len(self.lists[0])
        self.in_lists.update(other_collection.in_lists)
        return self

    def _merge_generator(self, lists1: list, lists2:list):
        gen1 = zip(*lists1)
        gen2 = zip(*lists2)
        it1 = next(gen1)
        it2 = next(gen2)

        pop_side = None # pop from it1: True, from it2, from both: None
        while True:
            for el1, el2 in zip(it1, it2):
                if el1 < el2:
                    yield it1
                    pop_side = True
                    break

                if el2 < el1:
                    yield it2
                    pop_side = False
                    break
            else:   # no break => it1 and it2 is the same item
                yield it1
                pop_side = None

            try:
                if pop_side:    # all these may raise StopIteration
                    it1 = next(gen1)
                elif pop_side is False:
                    it2 = next(gen2)
                else:
                    it1, it2 = next(gen1), next(gen2)
            except StopIteration:
                break
        # one of the iterators will be empty
        for it1 in gen1:
            yield it1

        for it2 in gen2:
            yield it2








    def reset_counts(self):
        self.counts = [0 for _ in range(self.size)]

    def count(self, count_set: AprioriSet):
        try:
            self.counts[self._index(count_set)] += 1
        except KeyError:
            pass

    # def _recur_bounds(self, i, hi, lo, j):
    #     l = self.lists[j]
    #     inner_lo = bisect.bisect_left(l, i, lo=lo, hi=hi)
    #
    #     # if i is in list, find hi bound using newfound lo
    #     if l[inner_lo] == i:
    #         inner_hi = bisect.bisect_right(l, i, lo=inner_lo, hi=hi)
    #
    #     # if i is NOT in list, try the next one with same bounds
    #     else:

    def filter_infrequent(self, basket_set: Sequence):
        """
        yield items in basket that is contained in lists
        :param basket_set: sequence of items
        :return:
        """
        for i in basket_set:
            if i in self.in_lists:
                yield i

    def count_basket(self, basket_set):
        """
        Add basket items sets of size k to counts
        :param basket_set: SORTED sequence
        :return:
        """
        for item_set in combinations(self.filter_infrequent(basket_set), self.k):
            self.count(item_set)

    def frequent_items_collection(self, min_support, n_baskets):
        return self.from_sorted_items_iter_w_count(self.frequent_items(min_support, n_baskets), self.k)

    def frequent_items(self, min_support, n_baskets):
        for item, count in zip(zip(*self.lists), self.counts):
            if count / n_baskets > min_support:
                yield chain((count,), item)

    def to_partitioned_tuples(self, p):
        """
        output p tuples that can be used to create subcollections of this collection
        :param p:
        :return:
        """
        psz = int(self.size / p) # partition size
        i = 0
        for j in range(p - 1):
            yield self.k, tuple(l[i:i+psz] for l in self.lists), self.counts[i:i+psz], psz, self.in_lists
            i += psz
        yield self.k, tuple(l[i:] for l in self.lists), self.counts[i:], self.size-i, self.in_lists

    def to_tuple(self):
        """
        output tuple that can be used to recreate this collection
        :return:
        """
        return self.k, self.lists, self.counts, self.size, self.in_lists

    @classmethod
    def from_tuple(cls, tup_collection: tuple):
        """
        build a new collection from a tuple generated by .to_tuple()
        :param tup_collection: a tuple generated by .to_tuple
        :return: AprioriCollection
        """
        nc = cls(tup_collection[0])
        nc.k, nc.lists, nc.counts, nc.size, nc.in_lists = tup_collection
        return nc

    @classmethod
    def from_tuples(cls, tuples: Iterator):
        """
        build a new collection from tuples generated by .to_partioned_tuples(p)
        The tuples MUST be passed in same order as they were yielded
        :param tuples: a sequence of tuples a tuple generated by .to_partioned_tuples
        :return: AprioriCollection
        """
        nc = None
        for k, lists, counts, size, in_lists in tuples:
            if nc is None:
                nc = cls(k)
            for i, l in enumerate(lists):
                nc.lists[i].extend(l)
            nc.counts.extend(counts)
            nc.size += size
            nc.in_lists.update(in_lists)
        return nc

    @classmethod
    def from_lists(cls, lists: tuple, k):
        nc = cls(k)
        nc.lists = lists
        nc.init_from_list()
        return nc

    @classmethod
    def from_sorted_items_iter(cls, items: Iterator, k):
        nc = cls(k)
        nc.lists = tuple(zip(*items))
        nc.init_from_list()
        return nc

    @classmethod
    def from_sorted_items_iter_w_count(cls, items: Iterator, k):
        nc = cls(k)
        #items = tuple(items)
        count, *nc.lists = zip(*items)
        nc.counts = count
        nc.init_from_list(reset_counts=False)
        return nc

    def init_from_list(self, reset_counts=True):
        self.size = len(self.lists[0])
        if reset_counts:
            self.reset_counts()
        self.build_in_lists()

    def build_in_lists(self):
        self.in_lists = set(chain(*self.lists))

    def to_item_sets(self):
        for item_set in zip(*self.lists):
            yield AprioriSet(item_set)



