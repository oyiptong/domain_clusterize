#!/usr/bin/env python
import os
import sys
import argparse
import multiprocessing

THREADS = multiprocessing.cpu_count()
ITERATIONS = 2000
TOP_WORDS = 70

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run mallet with options")
    parser.add_argument("topics", metavar="[NUM_TOPICS]", type=int, help="the number of topics to model", nargs=1)
    parser.add_argument("vectorfile", metavar="[VECTOR_FILE]", type=str, help="the location of the vector file", nargs=1)
    parser.add_argument("--jobs", "-j", metavar="NUM_THREADS", type=int, help="number of threads to use", default=THREADS)
    parser.add_argument("--iterations", "-i", metavar="NUM_ITERATIONS", type=int, help="number of iterations for LDA", default=ITERATIONS)
    parser.add_argument("--words", "-w", metavar="NUM_TOP_WORDS", type=int, help="number of top words for the doc-topic output", default=TOP_WORDS)
    args = parser.parse_args()

    topics = args.topics[0]
    vector_filename = os.path.abspath(os.path.join(os.getcwd(), args.vectorfile[0]))
    threads = args.jobs
    iterations = args.iterations
    top_words = args.words

    if not os.path.isfile(vector_filename):
        print "Please enter the location to a valid vector file"
        sys.exit(1)

    dirname, filename = os.path.split(vector_filename)
    prefix,_ = os.path.splitext(filename)
    output_dirname = "{0}.{1}".format(prefix, topics)

    output_path = os.path.join(dirname, output_dirname)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    doc_topics = os.path.join(output_path, "doc-topics.txt")
    topic_keys = os.path.join(output_path, "topic-keys.txt")
    output_state = os.path.join(output_path, "{0}.{1}.state.gz".format(prefix, topics))

    mallet_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mallet-2.0.7/bin/mallet")
    args =  [
             mallet_location,
             "train-topics",
             "--input", vector_filename,
             "--output-state", output_state,
             "--optimize-interval", str(10),
             "--num-topics", str(topics),
             "--num-iterations", str(iterations),
             "--output-topic-keys", topic_keys,
             "--output-doc-topics", doc_topics,
             "--num-top-words", str(top_words),
             "--num-threads", str(threads),
    ]

    os.execvp(args[0], args)
