Python 3 Implementation of Apriori Algorithm
==========================================
This project is a fork of https://github.com/asaini/Apriori

Changes are:
1. Project is now in python 3 package
2. Complete rewrite of data structure for speed and memory efficiency (find these changes in the .items)
3. Supports 3 types of input: file pointer to a previous session, pandas Data frame and the original

The main speedup comes from keeping all items sorted which can be exploited many places throughout the algorithm.
The memory efficiency comes from storing items as tuples in lists rather than frosensets in sets. Apparantly
this makes a huge difference..

List of files
-------------
1. apriori.py
2. INTEGRATED-DATASET.csv
3. README(this file)

The dataset is a copy of the “Online directory of certified businesses with a detailed profile” file from the Small Business Services (SBS) 
dataset in the `NYC Open Data Sets <http://nycopendata.socrata.com/>`_

Usage
-----
To run the program with dataset provided and default values for *minSupport* = 0.15 and *minConfidence* = 0.6

    python apriori.py -f INTEGRATED-DATASET.csv

To run program with dataset  

    python apriori.py -f INTEGRATED-DATASET.csv -s 0.17 -c 0.68

Best results are obtained for the following values of support and confidence:  

Support     : Between 0.1 and 0.2  

Confidence  : Between 0.5 and 0.7 

License
-------
MIT-License

-------
