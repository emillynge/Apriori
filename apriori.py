"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
from operator import itemgetter
from itertools import zip_longest
from collections import defaultdict, deque
from pandas import DataFrame
from optparse import OptionParser
from io import StringIO
import progressbar
import logging
import math
import time
import bisect
import sys
from itertools import chain, combinations
from multiprocessing import Process, Queue, queues
from .items import AprioriSet, AprioriCollection, AprioriCounter, AprioriSession, AprioriCollectionMerger, AprioriCollectionSubset
_print = print

class Logging(object):
    log_streams = None
    config = None
    @classmethod
    def print(cls, *args, **kwargs):
        if cls.log_streams is None:
            return _print(*args, **kwargs)

        if cls.config is None:
            handlers = list()
            for stream in cls.log_streams:
                if stream == 'stream':
                    handlers.append(logging.StreamHandler(sys.stdout))
                else:
                    handlers.append(logging.FileHandler(stream, mode='a'))

            logging.basicConfig(handlers=handlers, level=logging.INFO)
            cls.log_streams = handlers
            cls.config = True

        logging.info(*args, **kwargs)
print = Logging.print

class PBlogger(StringIO):
    real_logger = None

    def write(self, *args, **kwargs):
        pos = self.tell()
        super(PBlogger, self).write(*args, **kwargs)
        self.seek(pos)
        if self.real_logger is None:
            return sys.stderr.write(self.read())

        if 'strip' in self.real_logger:
            msg = self.read().strip('\r')
            if 'print' not in self.real_logger:
                msg += '\n'
        else:
            msg = self.read()

        if 'stdout' in self.real_logger:
            return sys.stdout.write(msg)

        if 'print' in self.real_logger:
            return print(msg)

        return sys.stderr.write(msg)


def prog_bar(maxval, message=''):
    class Message(progressbar.Widget):
        """Displays the current count."""
        __slots__ = ('message',)

        def __init__(self):
            self.message = message

        def __call__(self, msg):
            self.message = msg

        def update(self, pbar):
            return '\t' + self.message

    msg_widget = Message()
    pb = progressbar.ProgressBar(maxval=maxval, widgets=[progressbar.widgets.Percentage(), ' '
                                                                                             ' ',
                                                           progressbar.widgets.Timer(),
                                                           '   ',
                                                           progressbar.widgets.ETA(),
                                                           msg_widget], fd=PBlogger())
    return pb, msg_widget


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def returnItemsWithMinSupport(collect: AprioriCollection, transactions, min_support, n_workers):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    chunk_siz = max(5000, (collect.size // n_workers) // 25)

    MP = True

    if not MP:
        n_workers = 1

    intervals_q = Queue()
    counted_q = Queue()

    print('\n\tWorkers: {0}\n\tChunk size: {1}\n\tCollection size: {2}'.format(n_workers, chunk_siz, collect.size))
    def worker():
        while True:
            interv = intervals_q.get()
            if interv is StopIteration:
                counted_q.put(StopIteration)
                break
            start, end, i = interv
            counter = AprioriCounter(collect, start, end, transactions, min_support)
            counted_q.put((i, counter))

    print('init job queue')
    start = None
    n_chunks = 1
    for i, stop in enumerate(range(0, collect.size, chunk_siz)):
        if start is None:
            start = stop
            continue
        intervals_q.put((start, stop, i))
        n_chunks += 1
        start = stop
    intervals_q.put((start, collect.size, i + 1))

    for _ in range(n_workers):
        intervals_q.put(StopIteration)

    print('start workers')
    if MP:
        p = deque(Process(target=worker) for _ in range(n_workers))
        for proc in p:
            proc.start()
    else:
        worker()

    def get_results():
        sentinels = 0
        pb, msg = prog_bar(n_chunks + n_workers, 'counting baskets')
        result_cache = dict()

        def get_res():
            nonlocal sentinels
            try:
                result = counted_q.get(timeout=5)
            except queues.Empty:
                return

            if result is StopIteration:
                sentinels += 1
                return
            result_cache[result[0]] = result[1]

        i = 1
        pb.start()
        while sentinels < n_workers:
            pb.update(n_chunks - intervals_q.qsize() + n_workers)
            get_res()
            # print('##########################')
            # objgraph.show_growth()
            # msg('memory: {0}'.format(mem))
            while i in result_cache:
                res = result_cache.pop(i)
                if res:
                    yield res
                i += 1
        pb.finish()
        if MP:
            for proc in p:
                proc.join()

    collect.filter_from_counters(get_results(), min_support, len(transactions))


compl_time_ratios = list()


def merge_worker(merge_collections, msg):
    ii = 0
    count = 0
    for i in range(len(merge_collections)):
        while len(merge_collections[i]) > 1:
            # pop 2 collections from i and append result to i + 1
            merge_collections[i + 1].append(merge_collections[i].pop().merge(merge_collections[i].pop()))
            ii += 1
        count += len(merge_collections[i])
    msg('merged {0} times'.format(ii))
    return count

class Timer(object):
    def __init__(self):
        self.t0 = time.time()
        self.t_last = self.t0

    @property
    def elapsed(self):
        t = time.time()
        _ret = '{:2.2f} - {:2.2f}'.format(t - self.t_last, t - self.t0)
        self.t_last = t
        return _ret

def setGenerator(collection: AprioriCollection, n_workers):
    timer = Timer()
    length = collection.k + 1
    MP = True
    l = collection.size

    if length == 2:
        set_list = list(AprioriSet(item_set) for item_set in combinations((item[0] for item in collection.set_list), 2))
        return AprioriCollection.from_lists(set_list, 2)

    if not MP:
        n_workers = 1

    chunks = n_workers * length
    chunk_sz = l // n_workers - 1
    print('Generating sets of length {0} with {1} starting sets'.format(length, l))
    print('\tworkers: {}\n\tchunks: {}'.format(n_workers, chunks))

    print('Create sub lists... ' + timer.elapsed)
    sub_lists = collection.sub_set_lists()

    set_queue = Queue()
    filtered_sets_queue = Queue()
    def worker():
        local_sets = AprioriCollectionMerger(length)
        _local_set = local_sets.new_set()

        while True:
            g = set_queue.get()
            if g is StopIteration:
                break
            start, end, k = g
            do_merges = k != collection.k - 1

            sub_list = sub_lists[k]
            assert isinstance(sub_list, AprioriCollectionSubset)
            prev_ending = tuple()
            for items in sub_list.iter_subset(start, end):
                if len(items) <= 1:
                    continue

                item_prefix = items[0][:-1]
                item_endings = tuple(item[-1] for item in items)
                if do_merges and prev_ending > item_endings[:2]:
                    # the sorting of _local_set will be broken if we append these items. do swap
                    _local_set = local_sets.new_set(_local_set)

                _local_set.extend(sub_list.recreate_item(item_prefix, small_ending, large_ending)
                                  for small_ending, large_ending in combinations(item_endings, 2))
                prev_ending = item_endings[-2:]

            _local_set = local_sets.new_set(_local_set)

        local_sets.merge(full_merge=True)
        filtered_sets_queue.put(local_sets.collections[0])
        filtered_sets_queue.put(StopIteration)

    print('init job queue... ' + timer.elapsed)
    if MP:
        p = deque(Process(target=worker) for _ in range(n_workers))
        for proc in p:
            proc.start()

    for out_k in range(collection.k):
        cuts = [0] + [sub_lists[out_k].find_cut_idx(chunk_sz * i) for i in range(1, n_workers)] + [l]
        for start, end in zip(cuts[:-1], cuts[1:]):
            set_queue.put((start, end, out_k))

    print('workers working...' + timer.elapsed)
    pb, msg = prog_bar(chunks, '')
    pb.start()
    msg('started')
    for _ in range(n_workers):
        set_queue.put(StopIteration)

    def pb_update():
        pb.update(max(chunks - set_queue.qsize(), 0))
    pb_update()

    if not MP:
        worker()

    merge_collections = AprioriCollectionMerger(length)
    sentinels = 0
    results = 0
    print('collecting results from workers...' + timer.elapsed)
    while sentinels < n_workers:
        try:
            result = filtered_sets_queue.get(timeout=5)
        except queues.Empty:
            pb_update()
            continue
        msg('{0} results received'.format(results))
        pb_update()
        if result is StopIteration:
            sentinels += 1
        else:
            results += 1
            merge_collections.append(result)
            pb_update()
    pb.finish()
    print('shutting down workers... ' + timer.elapsed)
    if MP:
        for proc in p:
            assert isinstance(proc, Process)
            proc.join()

    print('doing full merge... ' + timer.elapsed)
    merge_collections.merge(full_merge=True)
    full_set = merge_collections.collections[0]
    print('Done! ' + timer.elapsed)
    print('produced sets: {}'.format(full_set.size), )
    assert full_set.is_sorted()
    return full_set


def join_set(itemSet, n_workers):
    """Join a set with itself and returns the n-element itemsets"""
    return setGenerator(itemSet, n_workers)


def getItemSetTransactionListFromRecord(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = sorted(str(item) for item in record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add((item,))

    itemSet = AprioriCollection.from_lists(sorted(list(itemSet)), 1)

    return transactionList, itemSet


def getItemSetTransactionListFromDataFrame(df: DataFrame):
    transactions = [tuple(sorted(int(el) for el in df.index[df[col] == 1].tolist())) for col in df.columns]
    item_set = AprioriCollection.from_lists(sorted(AprioriSet((int(el),)) for el in df.index.tolist()), 1)
    return transactions, item_set


def runApriori(data, min_support, minConfidence, max_k=None, fp=None, n_workers=4):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    print('Reading in data')
    if isinstance(data, DataFrame):
        large_set = AprioriSession.from_scratch(*getItemSetTransactionListFromDataFrame(data), fp=fp)
    elif hasattr(data, 'send'):
        large_set = AprioriSession.from_scratch(*getItemSetTransactionListFromRecord(data), fp=fp)
    else:
        large_set = AprioriSession.from_fp(data, fp=fp)

    print('Data set loaded -  total_size: {}'.format(large_set.total_size))
    assert isinstance(large_set, AprioriSession)
    last_collect = large_set.last_collection()
    last_collect.sub_set_lists()
    while last_collect.size != 0:
        if max_k and max_k in large_set and large_set[max_k].counts:
            break

        try:
            if last_collect.counts:
                print('Joining sets...')
                large_set[last_collect.k + 1] = join_set(last_collect, n_workers)
            else:
                print('Counting...')
                returnItemsWithMinSupport(last_collect,
                                          large_set.transactions,
                                          min_support,
                                          n_workers)
                large_set.save()
            last_collect = large_set.last_collection()
        except KeyboardInterrupt:
            break

    n_transactions = len(large_set.transactions)

    def supports(counts):
        return [count / n_transactions for count in counts]

    def ret_items():
        for c in large_set.values():
            yield from zip(c.set_list, supports(c.counts))

    toRetItems = list(ret_items())

    toRetRules = []
    n_items = large_set.total_size
    pb, msg = prog_bar(n_items, 'generating rules')
    pb.start()
    for k, collect in large_set.items():
        if k == 1:
            continue

        for count, item in zip(collect.counts, (AprioriSet(_item) for _item in collect.set_list)):
            pb.update(pb.currval + 1)
            for subset in item.subsets():
                remain = AprioriSet(item.difference(subset))
                _k = len(subset)
                confidence = count / large_set[_k].get_count(subset)
                if confidence >= minConfidence:
                    toRetRules.append(((subset, remain), confidence))
    return toRetItems, toRetRules


def printResults(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=itemgetter(1)):
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ RULES:")
    for rule, confidence in sorted(rules, key=itemgetter(1)):
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().rstrip(',')  # Remove trailing comma
        record = frozenset(line.split(','))
        yield record


def main():
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print('No dataset filename specified, system with exit\n')
        sys.exit('System will exit')

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = runApriori(inFile, minSupport, minConfidence)

    printResults(items, rules)


if __name__ == "__main__":
    main()
