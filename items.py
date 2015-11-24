import bisect
from itertools import chain, combinations
from collections import Sequence, Iterator, OrderedDict, UserList, namedtuple
from operator import itemgetter
import memory_profiler
import heapq
import json
from abc import ABCMeta


class AprioriSet(tuple):
    __slots__ = []

    def union(self, other_set: tuple, collect_set):
        """
        union between two AprioriSets of size k, that adds result into a collection if union yields a set of size k+1
        :param other_set: AprioriSet of size k
        :param collect_set: AprioriSet of size k
        :return:
        """
        el_diff = None
        for idx, (el1, el2) in enumerate(zip(self, other_set)):
            if el1 != el2:
                if el_diff is not None:
                    return
                el_diff = idx, el2

        if el_diff is not None:
            collect_set.append(AprioriSet(chain(self[:el_diff[0]], (el_diff[1],), self[el_diff[0]:])))

    def difference(self, subset):
        it1, it2 = iter(self), iter(subset)
        el1, el2 = next(it1), next(it2)

        while True:
            try:
                if el1 < el2:
                    yield el1
                    el1 = next(it1)
                else:
                    el2 = next(it2)
                    el1 = next(it1)
            except StopIteration:
                yield from it1
                return

    def subsets(self):
        for k in range(1, len(self)):
            yield from combinations(self, k)

class AprioriCollection(object):
    __slots__ = ['set_list', 'k', 'size', 'counts', 'in_lists']

    def __init__(self, k):
        self.k = k
        self.set_list = list()
        self.size = 0
        self.counts = list()
        self.in_lists = set()

    def _index(self, find_set: AprioriSet):
        return bisect.bisect_left(self.set_list, find_set)

    def append(self, new_set: AprioriSet):
        self.set_list.append(new_set)
        self.size += 1
        #self.in_lists.update(iter(new_set))

    def add(self, new_set: AprioriSet):
        idx = self._index(new_set)
        if idx == self.size or self.set_list[idx] != new_set:
            self._insert_set(new_set, idx)

    def _insert_set(self, new_set: AprioriSet, idx):
        self.set_list.insert(idx, new_set)
        self.size += 1
        self.in_lists.update(iter(new_set))

    def merge(self, other_collection):
        assert isinstance(other_collection, AprioriCollection)
        self.set_list = list(self._merge_generator(self.set_list, other_collection.set_list))
        self.size = len(self.set_list)
        #self.in_lists.update(other_collection.in_lists)
        return self

    def _merge_generator(self, lists1: list, lists2:list):
        gen1 = iter(lists1)
        gen2 = iter(lists2)
        try:
            it1 = next(gen1)
            it2 = next(gen2)
            while True:
                try:
                    if it1 < it2:    # all these may raise StopIteration
                        yield it1
                        it1 = next(gen1)

                    elif it2 < it1:
                        yield it2
                        it2 = next(gen2)

                    else:
                        yield it1
                        it1, it2 = next(gen1), next(gen2)
                except StopIteration:
                    break
        except StopIteration:
            pass

        # one of the iterators will be empty
        yield from gen1
        yield from gen2

    def reset_counts(self):
        self.counts = [0 for _ in range(self.size)]

    def get_count(self, item):
        idx = bisect.bisect_left(self.set_list, item)
        if self.size == idx or self.set_list[idx] != item:
            raise KeyError('set: {0!r} not in this collection {1}'.format(item, (self.k, self.size, self.set_list[:10])))
        return self.counts[idx]

    def count(self, count_set: AprioriSet):
        idx = self._index(count_set)
        if idx != self.size and self.set_list[idx] == count_set:
            self.counts[idx] += 1

    def filter_infrequent(self, basket_set: Sequence, interval=None):
        """
        yield items in basket that is contained in lists
        :param basket_set: sequence of items
        :return:
        """
        if interval:
            in_lists = set(chain(*self.set_list[interval[0]:interval[1]]))
        else:
            in_lists = self.in_lists

        for i in basket_set:
            if i in in_lists:
                yield i

    def count_basket2(self, basket_set):
        """
        Add basket items sets of size k to counts
        :param basket_set: SORTED sequence
        :return:
        Note that the item_sets generated wil be in sorted order.
        Thus the index of the last found item_set is a lower bound on 
        the index for the item_set in the current iteration
        """


        filtered_basket = list(self.filter_infrequent(basket_set))
        last_set = AprioriSet(filtered_basket[-self.k:])
        max_hi = bisect.bisect_left(self.set_list, last_set, hi=(self.size - 1))
        idx = 0
        prev_idx = 0
        window = 20

        def get_idx(item_set, window, lo):
            hi = min(lo + window, max_hi)
            return bisect.bisect_right(self.set_list, item_set, lo=idx, hi=hi)

        for item_set in combinations(filtered_basket, self.k):
            idx = get_idx(item_set, window, idx)
            while self.set_list[idx] < item_set:
                window *= 2
                idx = get_idx(item_set, window, idx)

            if self.set_list[idx-1] == item_set:
                self.counts[idx-1] += 1
            window, prev_idx = (idx - prev_idx + window * 3) // 4, idx

    def frequent_items_collection(self, min_support, n_baskets):
        return self.from_sorted_items_iter_w_count(self.frequent_items(min_support, n_baskets), self.k)

    def frequent_items(self, min_support, n_baskets):
        for item, count in zip(self.set_list, self.counts):
            if count / n_baskets > min_support:
                yield count, item

    def to_partitioned_tuples(self, p):
        """
        output p tuples that can be used to create subcollections of this collection
        :param p:
        :return:
        """
        psz = int(self.size / p) # partition size
        i = 0
        for j in range(p - 1):
            yield self.k, self.set_list[i:i+psz], self.counts[i:i+psz], psz, self.in_lists
            i += psz
        yield self.k, self.set_list[i:], self.counts[i:], self.size-i, self.in_lists

    def to_tuple(self):
        """
        output tuple that can be used to recreate this collection
        :return:
        """
        return self.k, self.set_list, self.counts, self.size, self.in_lists

    def to_json_tuple(self):
        return self.k, self.set_list, self.counts, self.size, list(self.in_lists)

    @classmethod
    def from_tuple(cls, tup_collection: tuple):
        """
        build a new collection from a tuple generated by .to_tuple()
        :param tup_collection: a tuple generated by .to_tuple
        :return: AprioriCollection
        """
        nc = cls(tup_collection[0])
        nc.k, nc.set_list, nc.counts, nc.size, nc.in_lists = tup_collection
        #nc.set_list = [AprioriSet(item) for item in set_list]
        return nc

    @classmethod
    def from_json_tuple(cls, tup_collection: tuple):
        """
        build a new collection from a tuple generated by .to_tuple()
        :param tup_collection: a tuple generated by .to_tuple
        :return: AprioriCollection
        """
        nc = cls(tup_collection[0])
        nc.k, nc.set_list, nc.counts, nc.size, nc.in_lists = tup_collection
        nc.set_list = [tuple(item) for item in nc.set_list]
        nc.in_lists = set(nc.in_lists)
        return nc

    @classmethod
    def from_collection_iter(cls, collections: Iterator):
        nc = next(collections)
        for next_nc in collections:
            assert isinstance(next_nc, AprioriCollection)
            nc.set_list.extend(next_nc.set_list)
            nc.in_lists.update(next_nc.set_list)
            nc.size += next_nc.size
            nc.counts.extend(next_nc.counts)
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
        for k, set_list, counts, size, in_lists in tuples:
            if nc is None:
                nc = cls(k)
            nc.set_list.extend(set_list)
            nc.counts.extend(counts)
            nc.size += size
            nc.in_lists.update(in_lists)
        return nc

    def filter_from_counters(self, counter_gen, min_support, n_baskets):
        counts = list(chain(*counter_gen))
        support = [count / n_baskets >= min_support for count in counts]
        #print(support, len(support), self.size)
        self.set_list = [item for sup, item in zip(support, self.set_list) if sup]
        self.counts = [count for sup, count in zip(support, counts) if sup]
        self.size = len(self.set_list)
        self.in_lists = set()

    @classmethod
    def from_lists(cls, lists: list, k):
        nc = cls(k)
        nc.set_list = lists
        nc.init_from_list(reset_counts=False)
        return nc

    @classmethod
    def from_sorted_items_iter(cls, items: Iterator, k):
        nc = cls(k)
        nc.set_list = list(items)
        nc.init_from_list()
        return nc

    @classmethod
    def from_sorted_items_iter_w_count(cls, count_items: Iterator, k):
        nc = cls(k)
        count, set_list = zip(*count_items)
        nc.counts = list(count)
        nc.set_list = list(set_list)
        nc.init_from_list(reset_counts=False)
        return nc

    def init_from_list(self, reset_counts=True):
        self.size = len(self.set_list)
        if reset_counts:
            self.reset_counts()
        self.build_in_lists()

    def build_in_lists(self):
        self.in_lists = set(chain(*self.set_list))

    def to_item_sets(self):
        for item_set in self.set_list:
            yield AprioriSet(item_set)


class AprioriBasketsSubsets(list):
    __slots__ = ['gen']
    def __init__(self, basket: list, collect: AprioriCollection):
        self.gen = combinations(collect.filter_infrequent(basket), collect.k)
        super(AprioriBasketsSubsets, self).__init__(next(self.gen))
    
    def next(self):
        super(AprioriBasketsSubsets, self).__init__(next(self.gen))

class AprioriCounter(list):
    __slots__ = ['collect', 'min_support', 'start', 'end']

    def __init__(self, collection: AprioriCollection, start, end, basket_sets, min_support):
        self.min_support = min_support
        self.start, self.end = start, end
        super(AprioriCounter, self).__init__(self.count_baskets(collection, basket_sets))

    def count_baskets2(self, collect, basket_sets):
        """
        Add basket items sets of size k to counts
        :param basket_set: SORTED sequence
        :return:
        Note that the item_sets generated wil be in sorted order.
        Thus the index of the last found item_set is a lower bound on
        the index for the item_set in the current iteration
        """
        n_baskets = len(basket_sets)
        basket_generators = [AprioriBasketsSubsets(basket_set, collect) for basket_set in basket_sets]
        heapq.heapify(basket_generators)
        min_basket = heapq.heappop(basket_generators)
        
        def switch_basket(current_basket):
            # get next basket set from current generator. 
            # if generator is empty pop next basket
            # else first put new basket item on heap and the pop
            try:
                current_basket.next()
            except StopIteration:
                return heapq.heappop(basket_generators)
            return heapq.heappushpop(basket_generators, current_basket)
            
        try:
            for i, _item_set in enumerate(collect.set_list[self.start:self.end]):
                item_set = list(_item_set)
                if min_basket > item_set:
                    yield 0
                    continue
                    
                count = 0
                while min_basket < item_set:
                    min_basket.next()
                    while min_basket < item_set:
                        min_basket.next()

                    if min_basket == item_set:
                        count += 1
                        min_basket = switch_basket(min_basket)
                    else:
                        min_basket = heapq.heapreplace(basket_generators, min_basket)
                
                while min_basket == item_set:
                    count += 1
                    min_basket = switch_basket(min_basket)
                
                yield count
        except IndexError:
            yield count
            # No more basket sets left. the rest of the items has zero count
            for j in range(i + 1, self.end):
                yield 0          
                
    def count_baskets(self, collect, basket_sets):
        """
        Add basket items sets of size k to counts
        :param basket_set: SORTED sequence
        :return:
        Note that the item_sets generated wil be in sorted order.
        Thus the index of the last found item_set is a lower bound on
        the index for the item_set in the current iteration
        """
        n_baskets = len(basket_sets)
        basket_generators = list(combinations(collect.filter_infrequent(basket_set,
                                                                        interval=(self.start, self.end)),
                                              collect.k) for basket_set in basket_sets)
        baskets = [[gen, tuple(-1 for _ in range(collect.k))] for gen in basket_generators]

        for i, item_set in enumerate(collect.set_list[self.start:self.end]):
            rm_list = []
            count = 0
            for basket in baskets:
                try:
                    while item_set > basket[1]:
                        basket[1] = next(basket[0])

                    if item_set == basket[1]:
                        count += 1
                        basket[1] = next(basket[0])
                except StopIteration:
                    rm_list.append(basket)

            #if count / n_baskets > self.min_support:
            yield count

            for basket in rm_list:
                baskets.remove(basket)


class AprioriSession(OrderedDict):
    def __init__(self, transactions, collections, fp):
        self.transactions = transactions
        self.fp = fp
        super(AprioriSession, self).__init__(collections)

    def counted_collections(self):
        for collect in self.values():
            if collect.counts:
                yield collect

    @property
    def total_size(self):
        return sum(c.size for c in self.counted_collections())

    def _collection_tuples(self):
        for k, collect in self.items():
            yield k, collect.to_json_tuple()

    def save(self):
        if self.fp is not None:
            self.fp.seek(0)
            collections = dict(self._collection_tuples())
            json.dump({'collections': collections, 'transactions': self.transactions}, self.fp)
            self.fp.truncate()

    def last_collection(self) -> AprioriCollection:
        i = 1
        while i in self:
            i += 1
        return self[i - 1]

    def __setitem__(self, key, value):
        super(AprioriSession, self).__setitem__(key, value)
        self.save()

    @classmethod
    def from_fp(cls, fp_in, fp = None):
        data = json.load(fp_in)
        collections = [(int(k), AprioriCollection.from_json_tuple(collect_tuple)) for
                       k, collect_tuple in data.pop('collections').items()]
        collections.sort(key=itemgetter(0))
        transactions = data.pop('transactions')
        return cls(transactions, collections, fp)

    @classmethod
    def from_scratch(cls, transactions, item_set, fp=None):
        collections = [(1, item_set)]
        return cls(transactions, collections, fp)

    def __del__(self):
        if self.fp is not None:
            self.fp.close()
