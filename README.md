# BM25Retriever Performance Evaluation

This report summarizes the performance evaluation of different retriever implementations: TFIDFRetriever, CurrentBM25Retriever, and NewBM25Retriever.

## Results

### Initialization Time

| Retriever | Time (seconds) |
|-----------|----------------|
| TFIDFRetriever | 0.3169 ± 0.1186 |
| CurrentBM25Retriever | 0.2274 ± 0.0214 |
| NewBM25Retriever | 0.6571 ± 0.0252 |

The CurrentBM25Retriever shows the fastest initialization time, followed by TFIDFRetriever. NewBM25Retriever has the slowest initialization time.

### Query Time

| Retriever | Time (seconds per query) |
|-----------|--------------------------|
| TFIDFRetriever | 0.0586 ± 0.1754 |
| CurrentBM25Retriever | 0.0930 ± 0.0024 |
| NewBM25Retriever | 0.0019 ± 0.0002 |

NewBM25Retriever significantly outperforms the other retrievers in query time, being approximately 30 times faster than TFIDFRetriever and 49 times faster than CurrentBM25Retriever.

## Analysis

1. **Initialization**: While NewBM25Retriever has the slowest initialization time, this is typically a one-time cost and may be acceptable if query performance is prioritized.

2. **Query Performance**: NewBM25Retriever shows exceptional query performance, which is crucial for applications requiring frequent retrieval operations.

3. **Consistency**: NewBM25Retriever demonstrates the most consistent performance across queries, as indicated by its low standard deviation in query time.

## Execution Instructions

To run the performance evaluation:

1. Ensure you have the required dependencies installed:
   ```
   pip install langchain langchain-community scipy rank-bm25 scikit-learn datasets tqdm
   ```

2. Run the evaluation script:
   ```
   python eval_speed.py
   ```

   This will run both initialization and query speed tests by default.

3. To run specific tests:
   - For initialization speed test only:
     ```
     python eval_speed.py --init_speed
     ```
   - For query speed test only:
     ```
     python eval_speed.py --query_speed
     ```

The script will output the results, showing the average time and standard deviation for each operation across multiple runs.
