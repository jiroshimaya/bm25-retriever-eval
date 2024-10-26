import argparse
from langchain_community.retrievers import BM25Retriever as CurrentBM25Retriever
from langchain_community.retrievers import TFIDFRetriever
from new_bm25 import BM25Retriever as NewBM25Retriever
from new_bm25 import create_bm25_vectorizer
from datasets import load_dataset
import time
import statistics
from tqdm import tqdm

def measure_init_time(retriever_class, corpus, num_runs=10):
    times = []
    for _ in tqdm(range(num_runs), desc=f"Measuring {retriever_class.__name__} init time"):
        start_time = time.time()
        retriever_class(corpus)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)

def measure_query_time(retriever, queries):
    times = []
    for query in tqdm(queries, desc=f"Measuring {type(retriever).__name__} query time"):
        query_times = []
        start_time = time.time()
        retriever.invoke(query)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)


def verify_bm25_retrieval(corpus, queries, k = 100):
    current_bm25 = CurrentBM25Retriever.from_texts(corpus)
    current_bm25.k = k
    new_bm25 = NewBM25Retriever.from_texts(corpus)
    new_bm25.k = k

    mismatch_count = 0
    total_queries = len(queries)

    for query_id, query in enumerate(tqdm(queries, desc="Verifying retriever outputs")):
        current_results = current_bm25.invoke(query)
        new_results = new_bm25.invoke(query)

        # Compare the results
        if len(current_results) != len(new_results):
            print(f"Query {query_id}: Mismatch in number of results")
            mismatch_count += 1
            continue

        for idx, (current_doc, new_doc) in enumerate(zip(current_results, new_results), 1):
            if current_doc.page_content != new_doc.page_content:
                print(f"Query {query_id}: Content mismatch at result #{idx}")
                mismatch_count += 1
                break

    if mismatch_count == 0:
        print("All retriever outputs match.")
    else:
        mismatch_ratio = mismatch_count / total_queries
        print(f"Mismatch ratio: {mismatch_ratio:.2%} ({mismatch_count}/{total_queries} queries)")



def verify_bm25_scores(corpus, queries):
    from rank_bm25 import BM25Okapi
    from new_bm25 import create_bm25_vectorizer
    import numpy as np
    from tqdm import tqdm

    print("Verifying BM25 scores...")

    # Tokenize corpus and queries
    corpus_tokenized = [doc.split() for doc in corpus]
    queries_tokenized = [query.split() for query in queries]

    # Create original BM25 and new vectorizer
    original_bm25 = BM25Okapi(corpus_tokenized)
    new_vectorizer = create_bm25_vectorizer(corpus_tokenized)

    # Calculate scores using original BM25
    original_scores = []
    for query in tqdm(queries_tokenized, desc="Calculating original scores"):
        scores = original_bm25.get_scores(query)
        original_scores.append(scores)

    # Calculate scores using new vectorizer
    corpus_vectors = new_vectorizer.transform(corpus_tokenized)
    query_vectors = new_vectorizer.count_transform(queries_tokenized)
    new_scores = query_vectors.dot(corpus_vectors.T).toarray()
    
    # Convert lists to numpy arrays for easier manipulation
    original_scores_array = np.array(original_scores)
    new_scores_array = new_scores

    # Calculate the difference between new and original scores for top k
    score_difference = np.abs(new_scores_array - original_scores_array)

    # Calculate statistics
    mean_difference = np.mean(score_difference)
    std_difference = np.std(score_difference)
    max_difference = np.max(score_difference)
    min_difference = np.min(score_difference)

    # Calculate average of mean scores for top k
    original_mean_scores = np.mean(original_scores_array, axis=1)
    new_mean_scores = np.mean(new_scores_array, axis=1)
    average_of_means = (np.mean(original_mean_scores) + np.mean(new_mean_scores)) / 2

    print(f"Mean difference: {mean_difference}")
    print(f"Standard deviation of difference: {std_difference}")
    print(f"Max difference: {max_difference}")
    print(f"Min difference: {min_difference}")
    print(f"Average of mean scores: {average_of_means}")

def run_init_speed(corpus):
    tfidf_init_time, tfidf_init_stdev = measure_init_time(TFIDFRetriever.from_texts, corpus)
    current_bm25_init_time, current_bm25_init_stdev = measure_init_time(CurrentBM25Retriever.from_texts, corpus)
    new_bm25_init_time, new_bm25_init_stdev = measure_init_time(NewBM25Retriever.from_texts, corpus)

    print(f"TFIDFRetriever initialization: {tfidf_init_time:.4f} ± {tfidf_init_stdev:.4f} seconds")
    print(f"CurrentBM25Retriever initialization: {current_bm25_init_time:.4f} ± {current_bm25_init_stdev:.4f} seconds")
    print(f"NewBM25Retriever initialization: {new_bm25_init_time:.4f} ± {new_bm25_init_stdev:.4f} seconds")

def run_query_speed(corpus, queries):
    tfidf = TFIDFRetriever.from_texts(corpus)
    current_bm25 = CurrentBM25Retriever.from_texts(corpus)
    new_bm25 = NewBM25Retriever.from_texts(corpus)

    tfidf_query_time, tfidf_query_stdev = measure_query_time(tfidf, queries)
    current_bm25_query_time, current_bm25_query_stdev = measure_query_time(current_bm25, queries)
    new_bm25_query_time, new_bm25_query_stdev = measure_query_time(new_bm25, queries)

    print(f"TFIDFRetriever query time: {tfidf_query_time:.4f} ± {tfidf_query_stdev:.4f} seconds per query")
    print(f"CurrentBM25Retriever query time: {current_bm25_query_time:.4f} ± {current_bm25_query_stdev:.4f} seconds per query")
    print(f"NewBM25Retriever query time: {new_bm25_query_time:.4f} ± {new_bm25_query_stdev:.4f} seconds per query")

def run_verify_retrieval(corpus, queries):
    print("Verifying CurrentBM25Retriever and NewBM25Retriever outputs...")
    verify_bm25_retrieval(corpus, queries)


def run_verify_scores(corpus, queries):
    print("Verifying BM25 scores...")
    verify_bm25_scores(corpus, queries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speed tests for retrievers")
    parser.add_argument("--init_speed", action="store_true", help="Run initialization speed test")
    parser.add_argument("--query_speed", action="store_true", help="Run query speed test")
    parser.add_argument("--verify_retrieval", action="store_true", help="Verify retriever outputs")
    parser.add_argument("--verify_scores", action="store_true", help="Verify BM25 scores")
    
    args = parser.parse_args()

    print("Loading AG News dataset...")
    # Load the AG News dataset
    dataset = load_dataset('ag_news')

    # Define sub-corpus size
    speed_corpus_size = 100000
    accuracy_corpus_size = 10000
    speed_query_size = 10
    accuracy_query_size = 100
    # Extract the text data from the dataset
    speed_corpus = [item['text'] for item in dataset['train'].select(range(speed_corpus_size))]  # First sub_corpus_size items for corpus_a
    accuracy_corpus = [item['text'] for item in dataset['train'].select(range(accuracy_corpus_size))]
    speed_queries = [item['text'] for item in dataset['test'].select(range(speed_query_size))]
    accuracy_queries = [item['text'] for item in dataset['test'].select(range(accuracy_query_size))]
    
    
    if not (args.init_speed or args.query_speed or args.verify_retrieval or args.verify_scores):
        # If no arguments are provided, run all tests
        run_init_speed(speed_corpus)
        run_query_speed(speed_corpus, speed_queries)
        run_verify_retrieval(accuracy_corpus, accuracy_queries)
        run_verify_scores(accuracy_corpus, accuracy_queries)
    else:
        if args.init_speed:
            run_init_speed(speed_corpus)
        if args.query_speed:
            run_query_speed(speed_corpus, speed_queries)
        if args.verify_retrieval:
            run_verify_retrieval(accuracy_corpus, accuracy_queries)
        if args.verify_scores:
            run_verify_scores(accuracy_corpus, accuracy_queries)
