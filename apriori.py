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
def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def item_support_worker(transactionList, minSupport, items):

    _itemSet = set()
    localSet = defaultdict(int)

    for item in items:
        if item is None:
            continue
        for transaction in transactionList:
            if item.issubset(transaction):
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count)/len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet, localSet


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""

        n_workers = 4
        p = Pool(n_workers)
        _items = set()
        item_chunks = grouper(itemSet, int((len(itemSet) / n_workers)/4))
        for items, counters in p.map(partial(item_support_worker, transactionList, minSupport), item_chunks):
            _items.update(items)
            for item, count in counters.items():
                freqSet[item] += count
        p.close()
        return _items

compl_time_ratios = list()
def setGenerator(itemSet, length: int):
    n_workers = 4
    set_queue = Queue()
    filtered_sets_queue = Queue(maxsize=(n_workers+1))
    item_list = list(itemSet)
    def worker():
        local_set = set()
        while True:
            g = set_queue.get()
            if g is StopIteration:
                break
            j, _set = g
            for i in item_list[:j+1]:
                un = _set.union(i)
                if len(un) == length:
                    local_set.add(un)
        filtered_sets_queue.put(local_set)

    l = len(itemSet)
    compl = l ** 2 * math.log(length - 1) * (length - 1)
    if compl_time_ratios:
        t_est = compl / (1+compl_time_ratios[-1]) /2
    else:
        t_est = 0
    print('Generating sets of length {0} with {1} starting sets'.format(length, len(itemSet)))
    print('\tComplexity: {0:.0}\n\tEstimated completion time: {1:3.0f}'.format(compl, t_est))
    t_start = time.time()
    #pb = maxval=len(itemSet))

    p = deque(Process(target=worker) for _ in range(n_workers))
    pb = progressbar.ProgressBar(maxval=(l + n_workers), widgets=[progressbar.widgets.Percentage(), ' '
                                                                  ' ', progressbar.widgets.Timer(),
                                                                  '   ',
                                                                  progressbar.widgets.ETA()])


    for proc in p:
        proc.start()
    #chunk_sz = 10**2
    #n_comb = len(itemSet) * len(itemSet)
    #iterations = int(n_comb / chunk_sz)

    for j in enumerate(item_list):
        set_queue.put(j)

    pb.start()
    pb.update(l - set_queue.qsize())
    for _ in range(n_workers):
        set_queue.put(StopIteration)

    while filtered_sets_queue.qsize() < n_workers:
        pb.update(l - set_queue.qsize() + n_workers)
        time.sleep(.1)

    full_set = filtered_sets_queue.get()
    pb.update(l + 1)
    for i in range(n_workers - 1):
        full_set.update(filtered_sets_queue.get())
        pb.update(l + 1 + i)
    pb.finish()

    # for i in progressbar.ProgressBar(maxval=iterations)(range(iterations)):
    #     next_comb = [next(it) for j in range(chunk_sz)]
    #     for gen in p.map(worker, next_comb):
    #         yield from gen
    #     p.terminate()
    #
    # p = Pool(3)
    # for gen in p.map(worker, it):
    #     yield from gen
    # p.terminate()
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


def getItemSetTransactionList(df):
    assert isinstance(df, DataFrame)
    transactions = [frozenset(int(el) for el in df.index[df[col] == 1].tolist()) for col in df.columns]
    itemSet = set(frozenset([int(el)]) for el in df.index.tolist())
    return itemSet, transactions




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
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):
        try:
            print(k)
            largeSet[k-1] = currentLSet
            if max_k is not None and k > max_k:
                break
            currentLSet = joinSet(currentLSet, k)
            currentCSet = returnItemsWithMinSupport(currentLSet,
                                                    transactionList,
                                                    minSupport,
                                                    freqSet)
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
