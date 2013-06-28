#!/usr/bin/env python
import os
import argparse
import sys
import csv
from operator import itemgetter
import json
import time
import datetime
import numpy as np
from sklearn import cluster
import IPython

NUM_CLUSTERS = 40

csv.field_size_limit(sys.maxsize)

def html_output(filename, clusters, num_docs, num_topics, num_domains, markers, domain_counts, domain_ranks):
    if len(domain_ranks) == 0:
        for i, cluster in enumerate(clusters):
            clusters[i] = sorted(cluster)

    num_clusters = len(clusters)
    with open(filename, "w") as f:
        print >>f, "<!DOCTYPE html>"
        print >>f, "<html>"
        print >>f, "<head>"
        print >>f, "<meta charset=\"utf-8\">"
        print >>f, "<title>{0}/{1} Topics/Clusters for {2} domains</title>".format(num_clusters, num_topics, num_domains)
        print >>f, """<style type="text/css">
table.bottomBorder { border-collapse:collapse; text-align: left; width: 100%;}
table.bottomBorder td { border-bottom:1px dotted #AAA; }
table.bottomBorder th { border-bottom:1px solid #AAA; }
table.bottomBorder th, table.bottomBorder td { padding: 0 1em;}
table.bottomBorder th:first-child, table.bottomBorder td:first-child {width: 35%;}
table.bottomBorder th:nth-child(2), table.bottomBorder td:nth-child(2) {width: 15%;}
table.bottomBorder th:nth-child(3), table.bottomBorder td:nth-child(3) {width: 10%;}
table.bottomBorder th:nth-last-child(1), table.bottomBorder td:nth-last-child(1) {width: 20%;}
table.curvedEdges { border:10px solid #DDD; border-radius:15px; margin: 1em 0;}
table.curvedEdges td, table.curvedEdges th { border-bottom:1px dotted black; padding:5px; }
a:visited, a:active, a:link { font-weight: normal; color: black; text-decoration: none;  }
a:hover { text-decoration: underline; }
.cluster { width: 100%; }
</style>
"""
        print >>f, "</head>"
        print >>f, "<body>"
        print >>f, "<div class=\"cluster\">"
        print >>f, "<h2>Quick stats</h2>"
        print >>f, "<table class=\"curvedEdges\">"
        print >>f, "<tr><td>webpages</td><td>{0}</td></tr>".format(num_docs)
        print >>f, "<tr><td>domains</td><td>{0}</td></tr>".format(num_domains)
        print >>f, "<tr><td>topics</td><td>{0}</td></tr>".format(num_topics)
        print >>f, "<tr><td>clusters</td><td>{0}</td></tr>".format(num_clusters)
        print >>f, "</table>"

        for index, domains in enumerate(clusters):
            print >>f, "<h2>Cluster {0}</h2>".format(index)
            print >>f, "<table class=\"bottomBorder\">"
            print >>f, "<thead><th>domain</th><th>web ranking</th><th># pages</th><th>label</th></thead>"
            for domain in domains:
                print >>f, "<tr><td><a href=\"http://{0}\">{0}</a></td><td>{1}</td><td>{2}</td><td><strong>{3}</strong></td></tr>".format(domain, domain_ranks.get(domain, ""), domain_counts.get(domain, ""), markers.get(domain, ""))
            print >>f, "</table>"
        print >>f, "</div>"
        print >>f, "</div>"
        print >>f, "</body></html>"

def sort_by_rank(domains, domain_ranks):
    ranked = []
    for domain in domains:
        ranked.append((domain_ranks.get(domain, ''), domain))
    ranked = sorted(ranked, key=itemgetter(0))
    return [dr[1] for dr in ranked]

def clusterize(features, domain_means, num_clusters, domain_ranks):
    clusterer = cluster.KMeans(n_clusters=num_clusters)
    clusterer.fit(features)

    domain_list = domain_means.keys()

    # create clusters
    clusters = [[] for i in xrange(num_clusters)]
    for domain in domain_list:
        mean_proportion = domain_means[domain]
        cluster_index = clusterer.predict(mean_proportion)[0]
        clusters[cluster_index].append(domain)

    # sort elements in clusters by rank
    for i in xrange(num_clusters):
        domains = clusters[i]
        clusters[i] = sort_by_rank(clusters[i], domain_ranks)
    clusters = sorted(clusters)

    return clusters

def main():
    time_start = time.time()

    parser = argparse.ArgumentParser(description="generate clusters from mallet output")
    parser.add_argument("document_labels", metavar="[LABELS]", type=str, help="document list with labels", nargs=1)
    parser.add_argument("doc_topic_proportions", metavar="[PROPORTIONS]", type=str, help="document topic proportion file", nargs=1)
    parser.add_argument("output_filename", metavar="[OUTPUT_FILE]", type=str, help="the output file", nargs=1)
    parser.add_argument("--rank-data", "-r", metavar="[RANKING_FILE]", type=str, help="domain ranks for output sorting", nargs=1)
    parser.add_argument("--num-clusters", "-n", metavar="[NUM_CLUSTERS]", type=int, help="number of clusters to create", default=NUM_CLUSTERS)
    parser.add_argument("--page-clusters", "-p", action='store_true', help="to use individual pages for cluster analysis instead of domain averages", default=False)
    args = parser.parse_args()

    doc_domains = [] # per-document domain
    domain_counts = {} # domain pagecounts
    domain_docs = {} # domain topic proportions
    doc_index = 0 # document index
    domain_ranks = {} # domain ranking
    markers = {} # domains to highlight in output
    features = None # features for clustering
    domain_means = {} # mean domain proportions

    # markers for "calibration" by viewer of output
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markers.json'), 'r') as f:
        markers = json.load(f)
    print >>sys.stderr, "{0} loaded domain markers".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))

    # load rank data if provided
    if args.rank_data is not None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rank_data[0]), 'r') as f:
            domain_ranks = json.load(f)

        print >>sys.stderr, "{0} loaded domain ranks".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))
    else:
        print >>sys.stderr, "{0} domain ranks not loaded. clusters will be unsorted".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))


    # collect document domain indices
    with open(args.document_labels[0], 'rb') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for row in tsvreader:
            domain = row[1]
            doc_domains.append(domain)
            if domain_counts.has_key(domain):
                domain_counts[domain] += 1
            else:
                domain_counts[domain] = 1

    print >>sys.stderr, "{0} loaded document labels".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))

    # collect document proportions for each domain
    with open(args.doc_topic_proportions[0], 'rb') as dtpfile:
        tsvreader = csv.reader(dtpfile, delimiter="\t")
        tsvreader.next()
        for row in tsvreader:
            index = int(row[0])
            url = row[1]
            sorted_proportions = row[2:-1]


            proportions = [0 for i in xrange(len(sorted_proportions)/2)]

            i = 0
            while i < len(sorted_proportions):
                topic_id = int(sorted_proportions[i])
                prop = float(sorted_proportions[i+1])
                proportions[topic_id] = prop
                i += 2

            if args.page_clusters:
                if features is None:
                    features = np.array([proportions], np.float64)
                else:
                    features = np.vstack((features, np.array(proportions, np.float64)))

            doc_domain = doc_domains[doc_index]
            if domain_docs.has_key(doc_domain):
                domain_docs[doc_domain] = np.vstack((domain_docs[doc_domain], np.array(proportions, np.float64)))
            else:
                domain_docs[doc_domain] = np.array([proportions], np.float64)
            doc_index += 1

    print >>sys.stderr, "{0} loaded document topic proportions".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))

    # obtain mean proportions per domain
    for domain, docs in domain_docs.iteritems():
        mean_proportions = np.mean(docs, axis=0)
        domain_means[domain] = mean_proportions

    print >>sys.stderr, "{0} calculated domain means".format(str(datetime.timedelta(seconds=int(time.time()-time_start))))
    
    if not args.page_clusters:
        for domain in domain_counts.keys():
            if features is None:
                features = domain_docs[domain]
            else:
                features = np.vstack((features, domain_docs[domain]))

    task_start = time.time()
    clusters = clusterize(features, domain_means, args.num_clusters, domain_ranks)
    task_time = int(time.time()-task_start)
    print >>sys.stderr, "{0} clusterized data. time taken: {1}".format(str(datetime.timedelta(seconds=int(time.time()-time_start))), str(datetime.timedelta(seconds=task_time)))

    # output to html
    num_topics = len(domain_means[doc_domains[0]])
    num_domains = len(domain_counts.keys())
    html_output(args.output_filename[0], clusters, doc_index+1, num_topics, num_domains, markers, domain_counts, domain_ranks)

    print >>sys.stderr, "{0} done! wrote {1}".format(str(datetime.timedelta(seconds=int(time.time()-time_start))), args.output_filename[0])
    #IPython.embed()

if __name__ == "__main__":
    main()
