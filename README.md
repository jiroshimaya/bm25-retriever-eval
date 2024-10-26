# BM25Retriever Performance Evaluation

This repository provides an improved implementation of langchain's BM25Retriever aimed at enhancing query search speed, along with code to evaluate its performance against the existing BM25Retriever, and a report summarizing the evaluation results and analysis.

When evaluated on a corpus of 100,000 documents, the improved BM25Retriever took 2-3 times longer to initialize but achieved a search speed nearly 50 times faster than the original implementation.

While the code modifications resulted in slightly different search results in some cases compared to the existing implementation, the impact is considered negligible for most real-world applications.

Given these findings, this implementation is likely to be effective in scenarios where faster per-query search speed is prioritized over initialization time, which is expected to be the case for many applications.

## Results

### Query Time
The query times of TFIDFRetriever, the existing BM25Retriever (CurrentBM25Retriever), and the improved BM25Retriever (NewBM25Retriever) are compared. The table shows the average and standard deviation for 10 searches conducted on a corpus of 100,000 documents.

| Retriever | Time (seconds per query) |
|-----------|--------------------------|
| TFIDFRetriever | 0.0310 ± 0.0201 |
| CurrentBM25Retriever | 1.2280 ± 0.8320 |
| NewBM25Retriever | 0.0282 ± 0.0016 |

### Initialization Time

To assess the impact of this improvement on the initialization time of the Retriever, a comparison was made of the time required to initialize the TFIDFRetriever, the existing BM25Retriever (CurrentBM25Retriever), and the improved BM25Retriever (NewBM25Retriever). The table shows the average and standard deviation for 10 initializations conducted on a corpus of 100,000 documents.

| Retriever | Time (seconds) |
|-----------|----------------|
| TFIDFRetriever | 3.0546 ± 0.1234 |
| CurrentBM25Retriever | 2.5157 ± 0.1372 |
| NewBM25Retriever | 7.4195 ± 0.1648 |



### Retrieval Results
The extent to which search results differed for the same target documents and queries before and after the improvement was examined.

#### Retrieved documents Verification
A comparison was made between the top 100 search results of NewBM25Retriever and CurrentBM25Retriever on a corpus of 10,000 documents. Out of 100 tested queries, 6 queries (6%) showed differences in results. The rank at which the highest discrepancy occurred for each query was investigated. The results are as follows.

- Query 4: Mismatch at result #87
- Query 28: Mismatch at result #47
- Query 75: Mismatch at result #99
- Query 77: Mismatch at result #83
- Query 82: Mismatch at result #61
- Query 86: Mismatch at result #6

#### Score Verification
BM25 scores obtained by the Retriever before and after the improvement were compared. The BM25 scores calculated by the NewBM25Retriever (using precomputed BM25 vectors and query term frequency vectors) were compared with the scores calculated by the CurrentBM25Retriever (using the rank-bm25 get_scores function). For a corpus of 10,000 documents, the top 100 scores were calculated using both methods, and the differences were analyzed. The average difference between the two scores for the same query was1.04e-15, with a standard deviation of 1.97e-15 and a maximum difference of 1.13e-13. These values were very small compared to the average value of 11.35 for the top 100 BM25 scores.

## Discussion

1. **Query Time**: The NewBM25Retriever achieved a speed approximately 50 times faster than the CurrentBM25Retriever and was comparable to the TFIDFRetriever. This indicates that the expected speed improvement has been realized.

2. **Initialization Time**: The NewBM25Retriever had the slowest initialization time, taking 2-3 times longer than other retrievers. This is considered a drawback of the improvement. However, since the initialization of the Retriever only needs to be done once in many use cases, and given the significant improvement in per-query search speed, this is considered acceptable.

3. **Retrieval result Verification**: The results showed that the difference in BM25 scores between the improved and previous versions of the BM25Retriever was negligibly small. The reason it is not completely zero is thought to be due to floating-point rounding errors. Differences were observed in the top 100 results for 6% of the queries, but most discrepancies occurred below rank 50, suggesting minimal impact on applications using only top-ranked search results. Considering the maximum difference is on the order of 10e-13, it is believed that these discrepancies occur when BM25 scores are near zero.

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
