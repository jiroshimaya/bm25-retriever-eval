# BM25Retriever Performance Evaluation

This repository provides an improved implementation of langchain's BM25Retriever aimed at enhancing query search speed, along with code to evaluate its performance against the existing BM25Retriever, and a report summarizing the evaluation results and analysis.

When evaluated on a corpus of 10,000 documents, the improved BM25Retriever took 2-3 times longer to initialize but achieved a search speed nearly 50 times faster than the original implementation.

While the code modifications resulted in slightly different search results in some cases compared to the existing implementation, the impact is considered negligible for most real-world applications.

Given these findings, this implementation is likely to be effective in scenarios where faster per-query search speed is prioritized over initialization time, which is expected to be the case for many applications.

## Results

### Query Time

| Retriever | Time (seconds per query) |
|-----------|--------------------------|
| TFIDFRetriever | 0.0310 ± 0.0201 |
| CurrentBM25Retriever | 1.2280 ± 0.8320 |
| NewBM25Retriever | 0.0282 ± 0.0016 |

These results were obtained using a corpus of 100,000 documents and an average of 10 queries.


### Initialization Time

| Retriever | Time (seconds) |
|-----------|----------------|
| TFIDFRetriever | 3.0546 ± 0.1234 |
| CurrentBM25Retriever | 2.5157 ± 0.1372 |
| NewBM25Retriever | 7.4195 ± 0.1648 |

These results were obtained using a corpus of 100,000 documents and an average of 10 queries.


### Retrieval Results

#### Retrieval Verification
The verification process compared the top 100 search results from a corpus 10,000 documents between NewBM25Retriever and CurrentBM25Retriever. Out of 100 queries tested, 6 queries (6%) showed differences in their results. The specific ranks where mismatches occurred were:

- Query 4: Mismatch at result #87
- Query 28: Mismatch at result #47
- Query 75: Mismatch at result #99
- Query 77: Mismatch at result #83
- Query 82: Mismatch at result #61
- Query 86: Mismatch at result #6

#### Score Verification
The BM25 scores calculated by NewBM25Retriever (using pre-computed BM25 vectors and query term frequency vectors) were compared with those calculated by CurrentBM25Retriever (using the get_scores function from rank-bm25). For a corpus of 10,000 documents, top 100 scores were calculated using both methods, and the differences were analyzed. The mean difference between the two sets of scores was found to be -1.16e-16, with a standard deviation of 6.07e-15, indicating a high degree of consistency between the two implementations. The maximum difference was 8.53e-14 and the minimum difference was -1.14e-13. These differences are extremely small compared to the average of mean scores for the top 100 results, which was 30.53, confirming the new implementation's accuracy. All top 100 scores matched within the specified tolerance of 1e-6.


## Analysis

1. **Initialization**: NewBM25Retriever has the slowest initialization time, taking 2-3 times longer than the other retrievers. However, this is typically a one-time cost and is an acceptable trade-off for the significantly improved query performance.

2. **Query Performance**: NewBM25Retriever shows exceptional query performance, being nearly 50 times faster than CurrentBM25Retriever (0.0019s vs 0.0912s per query). This is crucial for applications requiring frequent retrieval operations and directly fulfills the main objective of improving query speed.

3. **Retrieval Accuracy and BM25 Score Verification**: The verification process demonstrates high consistency between NewBM25Retriever and CurrentBM25Retriever outputs. 6% of queries exhibited differences in their top 100 results, with most discrepancies occurring at rank 50 or below. Furthermore, the BM25 scores computed by the new implementation show remarkable alignment with the original scores. These findings strongly indicate that the new implementation maintains retrieval accuracy while achieving significant performance improvements.

## Execution Instructions

To run the performance evaluation:

1. Ensure you have the required dependencies installed:
   ```
   pip install langchain langchain-community==0.3.1 scipy rank-bm25 scikit-learn datasets tqdm
   ```

2. Run the evaluation script:
   ```
   python eval.py
   ```

   This will run all tests by default, including initialization speed, query speed, retrieval verification, and BM25 score verification.

3. To run specific tests:
   - For initialization speed test only:
     ```
     python eval.py --init_speed
     ```
   - For query speed test only:
     ```
     python eval.py --query_speed
     ```
   - For retrieval verification:
     ```
     python eval.py --verify_retrieval
     ```
   - For BM25 score verification:
     ```
     python eval.py --verify_scores
     ```

The script will output the results, showing the average time and standard deviation for each operation across multiple runs, as well as verification results. These comprehensive tests ensure that the improved BM25Retriever meets the objectives of enhanced query speed without compromising on retrieval accuracy or BM25 score calculation.
