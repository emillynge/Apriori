"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
from operator import itemgetter
from itertools import chain, combinations, combinations_with_replacement, zip_longest
from collections import defaultdict, deque
from pandas import DataFrame
from optparse import OptionParser
from functools import partial
import progressbar
import math
import time
from multiprocessing import Pool, Process, Queue
from .items import AprioriSet, AprioriCollection
def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])




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
    pb = progressbar.ProgressBar(maxval=(maxval), widgets=[progressbar.widgets.Percentage(), ' '
                                                                  ' ', progressbar.widgets.Timer(),
                                                                  '   ',
                                                                  progressbar.widgets.ETA(),
                                                                  msg_widget])
    return pb, msg_widget
    
    
def item_support_worker(transactions: list, minSupport, collection_tuple: tuple):
    collect = AprioriCollection.from_tuple(collection_tuple)
    del collection_tuple
    assert isinstance(collect, AprioriCollection)
    collect.build_in_lists()

    for transaction in transactions:
        collect.count_basket(transaction)
    freq_collect = collect.frequent_items_collection(minSupport, len(transactions))

    assert isinstance(freq_collect, AprioriCollection)
    return freq_collect
    #return freq_collect.to_tuple()


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def returnItemsWithMinSupport(item_set: AprioriCollection, transactions, min_support):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        n_workers = 4

        MP = True
        def workload_partition():
            if MP:
                p = Pool(n_workers)
                _map = p.map
            else:
                _map = map
            item_gen = item_set.to_partitioned_tuples(n_workers * 3)
            pb, msg = prog_bar(n_workers * 3, 'counting baskets')
            pb.start()
            for i, items_sub_collect in enumerate(_map(partial(item_support_worker, transactions, min_support), item_gen)):
                #assert isinstance(items_sub_collect, AprioriCollection)
                pb.update(i)
                yield items_sub_collect
            if MP:
                p.close()
            pb.finish()
        new_collect = AprioriCollection.from_collection_iter(workload_partition())
        return new_collect

compl_time_ratios = list()


def merge_worker(merge_collections, msg):
    ii = 0
    count = 0
    for i in range(len(merge_collections)):
        while len(merge_collections[i]) > 1:
            # pop 2 collections from i and append result to i + 1
            merge_collections[i + 1].append(merge_collections[i].pop().merge(merge_collections[i].pop()))
            ii +=1
        count += len(merge_collections[i])
    msg('merged {0} times'.format(ii)) 
    return count
    
    
def setGenerator(collection, length: int):
    l = collection.size
    n_workers = 4
    chunk_sz = 32768
    chunks = int(2 ** math.ceil(math.log2(l / chunk_sz)))

    set_queue = Queue()
    filtered_sets_queue = Queue()#maxsize=(n_workers+1))
    item_list = list(collection.to_item_sets())

    def worker():
        local_set = AprioriCollection(length)
        while True:
            if local_set.size >= chunk_sz:
                filtered_sets_queue.put(local_set)
                local_set = AprioriCollection(length)
            g = set_queue.get()
            if g is StopIteration:
                break
            j, _set = g
            assert isinstance(_set, AprioriSet)
            for i in item_list[:j]:
                _set.union(i, local_set)

        filtered_sets_queue.put(local_set)
        filtered_sets_queue.put(StopIteration)

    
    compl = l ** 2 * math.log(length - 1) * (length - 1)
    if compl_time_ratios:
        t_est = compl / (1+compl_time_ratios[-1]) /2
    else:
        t_est = 0
    print('Generating sets of length {0} with {1} starting sets'.format(length, l))
    print('\tComplexity: {0:.0}\n\tEstimated completion time: {1:3.0f}\n\tworkers: {2}\n\tchunks: {3}'.format(compl, t_est, n_workers, chunks))
    t_start = time.time()
    #pb = maxval=len(itemSet))
    pb, msg = prog_bar(l + chunks + n_workers, 'initializing...')
    MP = True

    if MP:
        p = deque(Process(target=worker) for _ in range(n_workers))
        for proc in p:
            proc.start()
    else:
        n_workers = 1

    for j in enumerate(item_list):
        set_queue.put(j)

    pb.start()
    msg('started')
    pb.update(l - set_queue.qsize())

    for _ in range(n_workers):
        set_queue.put(StopIteration)

    if not MP:
        worker()

    merge_collections = defaultdict(list)

    sentinels = 0
    results = 0
    while sentinels < n_workers:
        result = filtered_sets_queue.get()
        msg('{0} results received'.format(results))
        pb.update(l - set_queue.qsize())
        if result is StopIteration:
            sentinels += 1
        else:
            results += 1
            merge_collections[0].append(result)
            merge_worker(merge_collections, msg)
            pb.update(l - set_queue.qsize())

    if MP:
        for proc in p:
            assert isinstance(proc, Process)
            proc.join()

    
    collect_siz = merge_worker(merge_collections, msg)
    chunks = collect_siz
    pb.maxval = l + collect_siz
    merge_idx = 0
    current_merge = list()
    msg('merging {0} collections'.format(collect_siz))
    while True:
        pb.update(l + chunks - collect_siz)
        while len(current_merge) != 2:
            if collect_siz == 0:
                break

            if merge_collections[merge_idx]:
                current_merge.append(merge_collections[merge_idx].pop())
                collect_siz -= 1
            else:
                merge_idx += 1

        if len(current_merge) == 2:
            merge_collections[merge_idx + 1].append(current_merge.pop().merge(current_merge.pop()))
            collect_siz += 1
        else:
            full_set = current_merge[0]
            full_set.size = len(full_set.set_list)
            full_set.reset_counts()

            break
    pb.finish()
    t_delta = time.time() - t_start
    compl_time_ratios.append(compl / t_delta)
    print('\nDone in {0:3.2f} seconds. complexity/time ratio: {1:.0f}\n'.format(t_delta, compl_time_ratios[-1]))
    return full_set

def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return setGenerator(itemSet, length)


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def getItemSetTransactionList(df: DataFrame):
    transactions = [tuple(sorted(int(el) for el in df.index[df[col] == 1].tolist())) for col in df.columns]
    item_set = AprioriCollection.from_lists(sorted(AprioriSet((int(el),)) for el in df.index.tolist()), 1)
    return item_set, transactions




def runApriori(data_iter, minSupport, minConfidence, max_k=None, min_k=2):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport)

    currentLSet = oneCSet
    k = 2
    while currentLSet.size != 0:
        try:
            print(k)
            largeSet[k-1] = currentLSet
            if max_k is not None and k > max_k:
                break
            currentLSet = joinSet(currentLSet, k)
            currentCSet = returnItemsWithMinSupport(currentLSet,
                                                    transactionList,
                                                    minSupport)
            currentLSet = currentCSet
            k = k + 1
        except KeyboardInterrupt:
            break

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence))
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
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record


if __name__ == "__main__":

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
