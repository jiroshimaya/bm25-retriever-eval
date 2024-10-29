import argparse
from typing import Union
from langchain_community.retrievers import BM25Retriever as CurrentBM25Retriever
from fastretriever import BM25Retriever as NewBM25Retriever
from fastretriever import BM25Vectorizer
from fastretriever import merge_bm25_retrievers, merge_bm25_vectorizers
from datasets import load_dataset
import time
import statistics
from tqdm import tqdm

BM25RetrieverClass = Union[CurrentBM25Retriever, NewBM25Retriever]



def merge_bm25_from_documents(retriever_list: list[BM25RetrieverClass]) -> CurrentBM25Retriever:
    merged_docs = []
    for retriever in retriever_list:
        merged_docs += retriever.docs
    bm25_params = {}
    bm25_params["k1"] = retriever_list[0].vectorizer.k1
    bm25_params["b"] = retriever_list[0].vectorizer.b
    bm25_params["epsilon"] = retriever_list[0].vectorizer.epsilon
    return retriever_list[0].__class__.from_documents(merged_docs, 
                                        k=retriever_list[0].k, 
                                        preprocess_func=retriever_list[0].preprocess_func,
                                        bm25_params = bm25_params)
    
    

def measure_new_bm25_merge_time(retriever_list: list[NewBM25Retriever], repeat_times = 10):
    times = []
    for _ in tqdm(range(repeat_times), desc=f"Measuring new BM25 retriever merge time"):
        start_time = time.time()
        merge_bm25_retrievers(retriever_list)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)

def measure_merge_time_from_documents(retriever_list: list[BM25RetrieverClass], repeat_times = 10):
    times = []
    for _ in tqdm(range(repeat_times), desc=f"Measuring simple merge time of {retriever_list[0].__class__.__name__}"):
        start_time = time.time()
        merge_bm25_from_documents(retriever_list)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)

def measure_merge_time_of_new_vectorizer(retriever_list: list[NewBM25Retriever], repeat_times = 10):
    times = []
    for _ in tqdm(range(repeat_times), desc=f"Measuring new vectorizer merge time of {retriever_list[0].__class__.__name__}"):
        start_time = time.time()
        vectorizers = [retriever.vectorizer for retriever in retriever_list]
        merge_bm25_vectorizers(vectorizers)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)

    
def measure_merge_time_of_vectorizer_from_documents(retriever_list: list[BM25RetrieverClass], repeat_times = 10):
    merged_docs = []
    for retriever in retriever_list:
        merged_docs += retriever.docs
    preprocess_func = retriever_list[0].preprocess_func
    
    times = []
    for _ in tqdm(range(repeat_times), desc=f"Measuring new vectorizer merge time from documents of {retriever_list[0].__class__.__name__}"):
        start_time = time.time()
        merged_preprocessed_docs = [preprocess_func(doc.page_content) for doc in merged_docs]
        BM25Vectorizer(merged_preprocessed_docs, k1=retriever_list[0].vectorizer.k1, b=retriever_list[0].vectorizer.b, epsilon=retriever_list[0].vectorizer.epsilon)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times), statistics.stdev(times)


def verify_bm25_retrieval(corpus, queries, k = 100):
    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]
    
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1)
    
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1)
    
    current_retriever = merge_bm25_from_documents([current_retriever0, current_retriever1])
    current_retriever.k = k
    new_retriever = merge_bm25_retrievers([new_retriever0, new_retriever1])
    new_retriever.k = k
    
    mismatch_count = 0
    total_queries = len(queries)
    
    doc_to_id = {doc: idx for idx, doc in enumerate(corpus)}

    def observe_retrieval_diff(query_id):
        query = queries[query_id]
        current_results = current_retriever.invoke(query)
        new_results = new_retriever.invoke(query)
        
        for rank, (current_result, new_result) in enumerate(zip(current_results, new_results)):
            if current_result.page_content != new_result.page_content:
                mismatch_rank = rank
                break
            
        current_scores = current_retriever.vectorizer.get_scores(query.split())
        new_query_vector = new_retriever.vectorizer.count_transform([query.split()])
        new_scores = new_query_vector.dot(new_retriever.bm25_array.T).toarray()[0]

        for rank in range(mismatch_rank, min(mismatch_rank+3, 100)):
            print(f"rank: {rank}")
            doc_id = doc_to_id[current_results[rank].page_content]
            print(doc_id, current_scores[doc_id], current_results[rank].page_content)
            doc_id = doc_to_id[new_results[rank].page_content]
            print(doc_id, new_scores[doc_id], new_results[rank].page_content)

    for query_id, query in enumerate(tqdm(queries, desc="Verifying retriever outputs")):
        current_results = current_retriever.invoke(query)
        new_results = new_retriever.invoke(query)

        # Compare the results
        if len(current_results) != len(new_results):
            print(f"Query {query_id}: Mismatch in number of results")
            mismatch_count += 1
            continue

        for idx, (current_doc, new_doc) in enumerate(zip(current_results, new_results), 1):
            if current_doc.page_content.strip() != new_doc.page_content.strip():
                print(f"Query {query_id}: Content mismatch at result #{idx}")
                mismatch_count += 1
                observe_retrieval_diff(query_id)
                break

    if mismatch_count == 0:
        print("All retriever outputs match.")
    else:
        mismatch_ratio = mismatch_count / total_queries
        print(f"Mismatch ratio: {mismatch_ratio:.2%} ({mismatch_count}/{total_queries} queries)")        


def verify_bm25_scores(corpus, queries):
    import numpy as np
    from tqdm import tqdm

    print("Verifying BM25 scores...")

    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]
    
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1)
    
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1)
    
    current_retriever = merge_bm25_from_documents([current_retriever0, current_retriever1])
    new_retriever = merge_bm25_retrievers([new_retriever0, new_retriever1])

    original_vectorizer = current_retriever.vectorizer
    new_vectorizer = new_retriever.vectorizer

    queries_tokenized = [query.split() for query in queries]
    corpus_tokenized = [doc.split() for doc in corpus]
    # Calculate scores using original BM25
    original_scores = []
    for query in tqdm(queries_tokenized, desc="Calculating original scores"):
        scores = original_vectorizer.get_scores(query)
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

def run_merge_speed(corpus):
    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1)
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1)
    new_merge_time, new_merge_stdev = measure_new_bm25_merge_time([new_retriever0, new_retriever1])
    current_merge_from_documents_time, current_merge_from_documents_stdev = measure_merge_time_from_documents([current_retriever0, current_retriever1])
    new_merge_from_documents_time, new_merge_from_documents_stdev = measure_merge_time_from_documents([new_retriever0, new_retriever1])
    print(f"NewBM25Retriever merge time: {new_merge_time:.4f} ± {new_merge_stdev:.4f} seconds")
    print(f"CurrentBM25Retriever merge time from documents: {current_merge_from_documents_time:.4f} ± {current_merge_from_documents_stdev:.4f} seconds")
    print(f"NewBM25Retriever merge time from documents: {new_merge_from_documents_time:.4f} ± {new_merge_from_documents_stdev:.4f} seconds")
def run_merge_speed_ja(corpus):
    import MeCab
    mecab = MeCab.Tagger("-Owakati")
    
    def preprocess_func(text: str)->list[str]:  
        return mecab.parse(text).strip().split()
    
    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]  
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0, preprocess_func=preprocess_func)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1, preprocess_func=preprocess_func)
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0, preprocess_func=preprocess_func)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1, preprocess_func=preprocess_func)
    new_merge_time, new_merge_stdev = measure_new_bm25_merge_time([new_retriever0, new_retriever1])
    current_merge_from_documents_time, current_merge_from_documents_stdev = measure_merge_time_from_documents([current_retriever0, current_retriever1])
    new_merge_from_documents_time, new_merge_from_documents_stdev = measure_merge_time_from_documents([new_retriever0, new_retriever1])
    print(f"NewBM25Retriever merge time ja: {new_merge_time:.4f} ± {new_merge_stdev:.4f} seconds")
    print(f"CurrentBM25Retriever merge time ja: {current_merge_from_documents_time:.4f} ± {current_merge_from_documents_stdev:.4f} seconds")
    print(f"NewBM25Retriever merge time from documents ja: {new_merge_from_documents_time:.4f} ± {new_merge_from_documents_stdev:.4f} seconds")

def run_vectorizer_merge_speed(corpus):
    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1)
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1)
    new_vectorizer_merge_time, new_vectorizer_merge_stdev = measure_merge_time_of_new_vectorizer([new_retriever0, new_retriever1])
    current_vectorizer_merge_from_documents_time, current_vectorizer_merge_from_documents_stdev = measure_merge_time_of_vectorizer_from_documents([current_retriever0, current_retriever1])
    new_vectorizer_merge_from_documents_time, new_vectorizer_merge_from_documents_stdev = measure_merge_time_of_vectorizer_from_documents([new_retriever0, new_retriever1])
    print(f"NewBM25Retriever vectorizer merge time: {new_vectorizer_merge_time:.4f} ± {new_vectorizer_merge_stdev:.4f} seconds")
    print(f"CurrentBM25Retriever vectorizer merge time from documents: {current_vectorizer_merge_from_documents_time:.4f} ± {current_vectorizer_merge_from_documents_stdev:.4f} seconds")
    print(f"NewBM25Retriever vectorizer merge time from documents: {new_vectorizer_merge_from_documents_time:.4f} ± {new_vectorizer_merge_from_documents_stdev:.4f} seconds")
def run_vectorizer_merge_speed_ja(corpus):
    import MeCab
    mecab = MeCab.Tagger("-Owakati")
    
    def preprocess_func(text: str)->list[str]:  
        return mecab.parse(text).strip().split()
    
    sub_corpus0 = corpus[:len(corpus)//2]
    sub_corpus1 = corpus[len(corpus)//2:]
    current_retriever0 = CurrentBM25Retriever.from_texts(sub_corpus0, preprocess_func=preprocess_func)
    current_retriever1 = CurrentBM25Retriever.from_texts(sub_corpus1, preprocess_func=preprocess_func)
    new_retriever0 = NewBM25Retriever.from_texts(sub_corpus0, preprocess_func=preprocess_func)
    new_retriever1 = NewBM25Retriever.from_texts(sub_corpus1, preprocess_func=preprocess_func)
    new_vectorizer_merge_time, new_vectorizer_merge_stdev = measure_merge_time_of_new_vectorizer([new_retriever0, new_retriever1])
    current_vectorizer_merge_from_documents_time, current_vectorizer_merge_from_documents_stdev = measure_merge_time_of_vectorizer_from_documents([current_retriever0, current_retriever1])
    new_vectorizer_merge_from_documents_time, new_vectorizer_merge_from_documents_stdev = measure_merge_time_of_vectorizer_from_documents([new_retriever0, new_retriever1])
    print(f"NewBM25Retriever vectorizer merge time ja: {new_vectorizer_merge_time:.4f} ± {new_vectorizer_merge_stdev:.4f} seconds")
    print(f"CurrentBM25Retriever vectorizer merge time from documents ja: {current_vectorizer_merge_from_documents_time:.4f} ± {current_vectorizer_merge_from_documents_stdev:.4f} seconds")
    print(f"NewBM25Retriever vectorizer merge time from documents ja: {new_vectorizer_merge_from_documents_time:.4f} ± {new_vectorizer_merge_from_documents_stdev:.4f} seconds")
    
def run_verify_retrieval(corpus, queries):
    print("Verifying CurrentBM25Retriever and NewBM25Retriever outputs...")
    verify_bm25_retrieval(corpus, queries)


def run_verify_scores(corpus, queries):
    print("Verifying BM25 scores...")
    verify_bm25_scores(corpus, queries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speed tests for retrievers")
    parser.add_argument("--verify_retrieval", action="store_true", help="Verify retriever outputs")
    parser.add_argument("--verify_scores", action="store_true", help="Verify BM25 scores")
    parser.add_argument("--merge_speed", action="store_true", help="Run merge speed test")
    parser.add_argument("--merge_speed_ja", action="store_true", help="Run merge speed test for Japanese")
    parser.add_argument("--vectorizer_merge_speed", action="store_true", help="Run vectorizer merge speed test")
    parser.add_argument("--vectorizer_merge_speed_ja", action="store_true", help="Run vectorizer merge speed test for Japanese")
    args = parser.parse_args()


    # Define sub-corpus size
    speed_corpus_size = 10000
    accuracy_corpus_size = 10000
    accuracy_query_size = 100

    is_all = not (args.verify_retrieval 
                  or args.verify_scores 
                  or args.merge_speed 
                  or args.merge_speed_ja 
                  or args.vectorizer_merge_speed 
                  or args.vectorizer_merge_speed_ja)
    if is_all or (args.verify_retrieval 
                  or args.verify_scores 
                  or args.merge_speed 
                  or args.vectorizer_merge_speed):
        print("Loading dataset...")
        # Load the AG News dataset
        ag_news_dataset = load_dataset('ag_news')
        # Extract the text data from the dataset
        speed_corpus = [item['text'] for item in ag_news_dataset['train'].select(range(speed_corpus_size))]  # First sub_corpus_size items for corpus_a
        accuracy_corpus = [item['text'] for item in ag_news_dataset['train'].select(range(accuracy_corpus_size))]
        accuracy_queries = [item['text'] for item in ag_news_dataset['test'].select(range(accuracy_query_size))]
    if is_all or (args.merge_speed_ja 
                  or args.vectorizer_merge_speed_ja):
        print("Loading Japanese dataset...")
        # Load the Japanese dataset
        japanese_dataset = load_dataset("llm-book/livedoor-news-corpus", split='train')
        speed_corpus_ja = japanese_dataset["content"][:speed_corpus_size]
    
    
    if is_all:
        # If no arguments are provided, run all tests
        run_verify_retrieval(accuracy_corpus, accuracy_queries)
        run_verify_scores(accuracy_corpus, accuracy_queries)
        run_merge_speed(speed_corpus)
        run_merge_speed_ja(speed_corpus_ja)
        run_vectorizer_merge_speed(speed_corpus)
        run_vectorizer_merge_speed_ja(speed_corpus_ja)
    else:
        if args.verify_retrieval:
            run_verify_retrieval(accuracy_corpus, accuracy_queries)
        if args.verify_scores:
            run_verify_scores(accuracy_corpus, accuracy_queries)
        if args.merge_speed:
            run_merge_speed(speed_corpus)
        if args.merge_speed_ja:
            run_merge_speed_ja(speed_corpus_ja)
        if args.vectorizer_merge_speed:
            run_vectorizer_merge_speed(speed_corpus)
        if args.vectorizer_merge_speed_ja:
            run_vectorizer_merge_speed_ja(speed_corpus_ja)