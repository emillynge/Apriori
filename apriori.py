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


def item_support_worker(transactions: list, minSupport, collection_tuple: tuple):
    collect = AprioriCollection.from_tuple(collection_tuple)
    assert isinstance(collect, AprioriCollection)
    collect.build_in_lists()

    for transaction in transactions:
        collect.count_basket(transaction)
    freq_collect = collect.frequent_items_collection(minSupport, len(transactions))

    assert isinstance(freq_collect, AprioriCollection)
    return freq_collect.to_tuple()


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def returnItemsWithMinSupport(item_set: AprioriCollection, transactions, min_support):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        n_workers = 4

        def workload_partition():
            #p = Pool(n_workers)
            item_gen = item_set.to_partitioned_tuples(n_workers)
            for items_sub_collect in map(partial(item_support_worker, transactions, min_support), item_gen):
                #assert isinstance(items_sub_collect, AprioriCollection)
                yield items_sub_collect
            #p.close()
        new_collect = AprioriCollection.from_tuples(workload_partition())
        return new_collect

compl_time_ratios = list()


def setGenerator(collection, length: int):
    n_workers = 4
    set_queue = Queue()
    filtered_sets_queue = Queue(maxsize=(n_workers+1))
    item_list = list(collection.to_item_sets())
    def worker():
        local_set = AprioriCollection(length)
        while True:
            g = set_queue.get()
            if g is StopIteration:
                break
            j, _set = g
            assert isinstance(_set, AprioriSet)
            for i in item_list[:j]:
                _set.union(i, local_set)

        filtered_sets_queue.put(local_set.to_tuple())

    l = collection.size
    compl = l ** 2 * math.log(length - 1) * (length - 1)
    if compl_time_ratios:
        t_est = compl / (1+compl_time_ratios[-1]) /2
    else:
        t_est = 0
    print('Generating sets of length {0} with {1} starting sets'.format(length, l))
    print('\tComplexity: {0:.0}\n\tEstimated completion time: {1:3.0f}'.format(compl, t_est))
    t_start = time.time()
    #pb = maxval=len(itemSet))
    pb = progressbar.ProgressBar(maxval=(l + n_workers), widgets=[progressbar.widgets.Percentage(), ' '
                                                                                                    ' ', progressbar.widgets.Timer(),
                                                                  '   ',
                                                                  progressbar.widgets.ETA()])
    MP = False

    if MP:
        p = deque(Process(target=worker) for _ in range(n_workers))
        for proc in p:
            proc.start()

    for j in enumerate(item_list):
        set_queue.put(j)

    pb.start()
    pb.update(l - set_queue.qsize())

    for _ in range(n_workers):
        set_queue.put(StopIteration)

    if MP:
        while filtered_sets_queue.qsize() < n_workers:
            pb.update(l - set_queue.qsize() + n_workers)
            time.sleep(.1)
    else:
        worker()

    merge_collections = defaultdict(list)
    merge_collections[0].append(AprioriCollection.from_tuple(filtered_sets_queue.get()))

    if MP:
        pb.update(l + 1)
        for i in range(n_workers - 1):
            merge_collections[0].append(AprioriCollection.from_tuple(filtered_sets_queue.get()))

        for proc in p:
            assert isinstance(proc, Process)
            proc.join()

    merge_idx = 0
    current_merge = list()
    collect_siz = len(merge_collections[0])
    while True:
        pb.update(l + collect_siz)
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
            full_set.reset_counts()
            break
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
    transactions = [tuple(sorted(int(el) for el in df.index[df[col] == 1].tolist())) for col in df.columns]

    item_set = AprioriCollection.from_lists((sorted(int(el) for el in df.index.tolist()),), 1)
    #item_set = set(AprioriSet((int(el),)) for el in df.index.tolist())
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
