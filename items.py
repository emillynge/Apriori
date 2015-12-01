import bisect
from itertools import chain, combinations
from collections import Sequence, Iterator, OrderedDict, UserList, deque
from operator import itemgetter
import heapq
import json
import sys
from progressbar import ProgressBar
from scipy.special import binom


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


class AprioriCollectionSubset(UserList):
    def __init__(self, sub_set_list, out_k):
        self.out_k = out_k
        super(AprioriCollectionSubset, self).__init__(sub_set_list)

    def recreate_item(self, prefix, small_ending, large_ending):
        return AprioriSet(chain(prefix[:self.out_k], (small_ending, large_ending), prefix[self.out_k:]))

    def find_cut_idx(self, i):
        item = self[i]
        for j in range(i, len(self)):
            if self[j] != item:
                return j
        return len(self)

    def separate_candidates(self, candidates):
        last_subset = [candidates.pop()]
        while candidates[-1] == last_subset[0]:
            last_subset.append(candidates.pop())
        return last_subset

    def iter_subset(self, start, end):
        gen = iter(self[start:end])
        for candidates in self._candidate_generator(gen):
            while candidates[0][:-1] != candidates[-1][:-1]:
                yield self.separate_candidates(candidates)
            yield candidates

    def _candidate_generator(self, gen):
        first_item = next(gen)
        while True:
            candidates = [first_item]
            item = next(gen)
            while item[-2] == first_item[-2]:
                candidates.append(item)
                item = next(gen)
            yield candidates
            first_item = item

        yield candidates


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

    def extend(self, new_sets):
        self.set_list.extend(new_sets)
        self.size = len(self.set_list)

    def add(self, new_set: AprioriSet):
        idx = self._index(new_set)
        if idx == self.size or self.set_list[idx] != new_set:
            self._insert_set(new_set, idx)

    def is_sorted(self):
        return self.set_list == sorted(self.set_list)

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

    def sub_set_lists(self):
        if self.k == 1:
            return [self.set_list]
        getters = [(out_k, itemgetter(*tuple(chain(range(out_k), range(out_k+1, self.k), (out_k,)))))
                   for out_k in range(self.k)]
        return [AprioriCollectionSubset(sorted([getter(item) for item in self.set_list]), out_k)
                for out_k, getter in ProgressBar(self.k, fd=sys.stdout)(getters)]

    @staticmethod
    def _merge_generator(lists1: list, lists2:list):
        gen1 = iter(lists1)
        gen2 = iter(lists2)

        it1, it2 = None, None
        try:
            it1 = next(gen1)
            it2 = next(gen2)
            while True:
                try:
                    if it1 < it2:    # all these may raise StopIteration
                        it1 = yield it1     # yield returns None to it1
                        it1 = next(gen1)

                    elif it2 < it1:
                        it2 = yield it2     # yield returns None to it2
                        it2 = next(gen2)

                    else:
                        it1 = yield it1     # yield returns None to it1
                        it2 = None
                        it1 = next(gen1)
                        it2 = next(gen2)

                except StopIteration:
                    break
        except StopIteration:
            pass

        if it1 is not None:
            yield it1
        elif it2 is not None:
            yield it2

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

    def filter_infrequent2(self, basket_sets: Sequence, interval=None):
        """
        basket sets with only the items that is contained in lists
        :param basket_set: sequence of items
        :return:
        """
        if interval:
            in_lists = set(chain(*self.set_list[interval[0]:interval[1]]))
        else:
            in_lists = self.in_lists

        return [[basket_element for basket_element in basket_set if basket_element in in_lists]
                for basket_set in basket_sets]

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


class AprioriCollectionMerger(object):
    def __init__(self, length):
        self.collections = list()
        self.l = length

    def merge(self, full_merge=False):
        # Merge last 2 collections in queue if last collection is larger than next to last
        while len(self.collections) > 1 and ((self.collections[-1].size >= self.collections[-2].size) or full_merge):
            self.collections.append(self.collections.pop().merge(self.collections.pop()))

    def append(self, collection):
        self.collections.append(collection)
        self.merge()

    def new_set(self, old_set: AprioriCollection=None) -> AprioriCollection:
        """
        put a local collection in merge queue and do as many merges as possible. return empry local collection
        :param old_set:
        :return:
        """
        if old_set is not None:
            self.collections.append(old_set)
            self.merge()
        return AprioriCollection(self.l)


class AprioriBasket(UserList):
    __slots__ = ['empty_list', 'remaining_items', 'k', 'hi', 'lo', 'gen']
    items_per_fill = 1500
    jumps = list()

    def __init__(self, basket_set, k, empty_list):
        self.empty_list = empty_list
        self.remaining_items = len(basket_set)
        self.k = k
        self.hi = self.items_per_fill
        self.lo = 0
        self.gen = combinations(basket_set, k)
        super().__init__(self.read(self.items_per_fill))
        self.append(tuple())    # insert an item that is guaranteed to be smallest in any comparison

    def read(self, n):
        """
        reads at most n items from combinations generator.
        yields no items if generator is exhausted
        :return:
        """
        i = 0
        for i, item in zip(range(n), self.gen):
            yield item

        if i != n - 1:
            self.gen = None
            self.hi = i

    def refill(self):
        super().__init__(self.read(self.items_per_fill))
        self.hi = len(self)
        self.append(tuple())

        self.lo = 0

    def spool(self, item, *args):
        # Check if we have exhausted current list
        while self[self.hi - 1] < item:
            # check if there are more lists to pull items from
            if not self.gen:
                self[0] = tuple()
                self.hi = 1
                self.empty_list.append(self)
                return False

            # refill list and check first item. We might have jumped "ahead" of the item
            self.refill()
            if self[0] > item:
                self.lo = 0
                return False

            # A good guess is that the item is the first one after a refill
            if self[0] == item:
                self.lo = 1
                return True


        # Try to find item among next 3 basket items (most jumps are small)
        i = 0
        while i < 3 and self[self.lo] < item:
            self.lo += 1
            i += 1

        if self[self.lo] >= item:
            if self[i] == item:
                return True
            return False

        # Jump is larger than 3, use bisection instead
        self.lo = bisect.bisect_left(self, item, lo=i, hi=(self.hi - 1))

        #if self.lo - _lo > 5:
        #   print('lo: {0}\thi: {2}\tjump: {1}'.format(self.lo, self.lo - _lo, self.hi))

        # check found item
        if self[self.lo] == item:
            AprioriBasket.jumps.append(self.lo - i)
            return True

        # if we are at last item we have to spool again
        #if self.hi <= self.lo + 1:
        #    return self.spool(item)

        # item is not in this basket
        return False

    def __contains__(self, item):
        if self[self.lo] == item:
            self.lo += 1
            return True

        if item < self[self.lo]:
            return False

        return self.spool(item)


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
        super(AprioriCounter, self).__init__(self.count_baskets3(collection, basket_sets))

    def count_baskets3(self, collect: AprioriCollection, baskets_sets: list):
        empty_list = list()
        baskets = [AprioriBasket(basket_set, collect.k, empty_list) for basket_set in
                   collect.filter_infrequent2(baskets_sets, interval=(self.start, self.end))]

        for i, item_set in enumerate(collect.set_list[self.start:self.end]):
            while empty_list:
                remove_basket = empty_list.pop()
                baskets.remove(remove_basket)
                if not baskets:
                    yield from (0 for _ in range(i, self.end))
                    return
            if i % 1000 == 0 and AprioriBasket.jumps:
                AprioriBasket.items_per_fill = max(3 * max(AprioriBasket.jumps), AprioriBasket.items_per_fill)
                AprioriBasket.jumps.clear()

            yield sum(item_set in basket for basket in baskets)


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

    @staticmethod
    def init_baskets(basket_generators):
        for gen in basket_generators:
            try:
                yield [gen, next(gen)]
            except StopIteration:
                pass

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
        baskets = list(self.init_baskets(basket_generators))

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

    def count_baskets4(self, collect, basket_sets):
        """
        Add basket items sets of size k to counts
        :param basket_set: SORTED sequence
        :return:
        Note that the item_sets generated wil be in sorted order.
        Thus the index of the last found item_set is a lower bound on
        the index for the item_set in the current iteration
        """
        n_baskets = len(basket_sets)
        basket_sets = list(deque(sorted(_set)) for _set in collect.filter_infrequent(basket_sets, interval=(self.start, self.end)))




        basket_generators = list(combinations(basket_set, collect.k) for basket_set in basket_sets)
        baskets = list(self.init_baskets(basket_generators))

        min_basket = min(basket[1] for basket in baskets)
        skips = 0
        prev_first_el = collect.set_list[self.start][0]
        for i, item_set in enumerate(collect.set_list[self.start:self.end]):
            #if item_set < min_basket:
            #    skips += 1
            #    continue
            if prev_first_el != item_set[0]:
                for basket_set, basket in zip(basket_sets, baskets):
                    assert isinstance(basket_set, deque)
                    if basket[1][0] == prev_first_el:
                        while basket_set.popleft() != prev_first_el:
                            pass
                        skips += 1
                        basket[0] = combinations(basket_set, collect.k)
                        basket[1] = next(basket[1])
                prev_first_el = item_set[0]

            rm_list = []
            count = 0
            for basket in baskets:
                try:
                    while item_set > basket[1]:
                        #skips += 1
                        basket[1] = next(basket[0])
                        #min_basket = min(min_basket, basket[1])

                    if item_set == basket[1]:
                        count += 1
                        basket[1] = next(basket[0])
                        min_basket = min(min_basket, basket[1])

                except StopIteration:
                    rm_list.append(basket)

            #if count / n_baskets > self.min_support:
            yield count

            for basket in rm_list:
                baskets.remove(basket)
        print(skips)

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
