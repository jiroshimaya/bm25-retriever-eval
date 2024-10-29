# Improvements and Evaluation of BM25Retriever

Fast implementation of langchain's BM25Retriever and performance evaluations.

#### [English](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.md) | [日本語](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.ja.md)

# Expected Use Cases
This implementation is likely to be effective in the following scenarios:

- **Improved Search Speed**: You want to improve the search speed while using langchain.BM25Retriever.
- **Improved Merge Speed**: You want to quickly merge BM25Retriever for purposes such as dynamically changing the target documents.
- **Prototyping**: You need to rapidly prototype keyword searches without building a separate database, and the number of target search items is over 100,000.

This implementation provides the following advantages and disadvantages. The performance is based on verification on the developer's laptop.


Advantages
- Improved merge speed (approximately 2-3 times faster in Japanese and 0.7-1.5 times faster in English for a corpus of about 10,000 items)
- Enhanced search speed (approximately 50 times faster for a corpus of 100,000 items)

Disadvantages
- Decreased initialization speed (0.3-0.7 times slower for a corpus of 100,000 items)

Based on the above results, this improvement is considered effective in cases where search speed is more important than initialization or when the frequency of merging is high.

# Usage

## Installation
In addition to langchain-community and rank-bm25, which are required to use the standard langchain.BM25Retriever, please install scipy for fast operations on sparse matrices. This script was developed with langchain-community version 0.3.1, but it should work with the latest version as long as there are no breaking changes to BM25Retriever.

```
pip install langchain-community rank-bm25 scipy
```

## BM25Retriever
Copy `src/fastretriever/bm25.py` into your project and call it from your script using the appropriate path. 
After that, you can use it almost the same way as the standard BM25Retriever. 
That is, you can initialize it with from_texts or from_documents and then perform searches using the invoke function.

Below is a code example when bm25.py is placed in the same directory as the calling script.

```Python
from bm25 import BM25Retriever

retriever = BM25Retriever.from_texts(["Hello", "Good evening"])
docs = retriever.invoke("Good")
print(docs)
```

```
[Document(metadata={}, page_content='Good evening'), Document(metadata={}, page_content='Hello')]
```

If you initialize the class directly without using from_texts or from_documents, additional arguments are required. Please refer to the source code for details.

## Merging
Use the merge_bm25_retrievers function from bm25.py. The retriever class provided to merge_bm25_retrievers must be the BM25Retriever from bm25.py, not the standard BM25Retriever.

```Python
from fastretriever import BM25Retriever, merge_bm25_retrievers
retriever0 = BM25Retriever.from_texts(["Hello"])
retriever1 = BM25Retriever.from_texts(["Good evening"])
merged_retriever = merge_bm25_retrievers([retriever0, retriever1])
docs = merged_retriever.invoke("Good")
print(docs)
```

The acceleration of merging is particularly effective when time-consuming preprocessing, such as Japanese tokenization, is specified in the preprocess_func.

```
pip install mecab-python3 unidic-lite
```
```Python
import MeCab
from bm25 import BM25Retriever, merge_bm25_retrievers

mecab = MeCab.Tagger("-oWakati")
def preprocess_func(text: str) -> list[str]:
    return mecab.parse(text).strip().split()


retriever0 = BM25Retriever.from_texts(["こんにちは、世界"], preprocess_func=preprocess_func)
retriever1 = BM25Retriever.from_texts(["さようなら、おやすみ"], preprocess_func=preprocess_func)
merged_retriever = merge_bm25_retrievers([retriever0, retriever1])
```

# Implementation Points
## search Speed Improvement
The rank-bm25 package used internally by BM25Retriever was slow, so we overrode the class rank-bm25.BM25Okapi to speed it up. Specifically, the slowness was caused by calculating the BM25 score in a dictionary loop every time a search was performed. To address this, we pre-calculate the BM25 weight vectors of the corpus, allowing us to simply take the dot product of the query's word frequency vector and the corpus vector during searches. However, since it is necessary to create the corpus weight vector during initialization, the initialization speed is slower compared to before the improvement.

## Merge Speed Improvement
Since BM25Retriever did not have a standard merge method, it was necessary to initialize it by providing the merged corpus. This was inefficient as it could not utilize the already calculated word frequency dictionaries and preprocessed data from the retrievers before merging. Therefore, in the improved BM25Retriever, we retained the word frequency dictionaries and preprocessed corpus as class properties, allowing them to be accessed externally. We then implemented a function to merge using these resources. Intuitively, by reusing the word frequency dictionaries, the computational complexity is reduced from the number of words to the number of word types.

# Evaluation
You can evaluate the search speed using `eval_query.py` and the merge speed using `eval_merge.py`.

To execute, clone the repository and navigate to the repository root.

```
git clone [repository url]
cd bm25-eval
```

Package management and script execution are done using uv. If not using uv, execute the following to install dependency libraries and packages. This is unnecessary if using uv.

```
pip install datasets tqdm mecab-python3 unidic-lite scikit-learn
pip install -e .
```

The following commands assume the use of uv. If you are not using uv, replace `uv run` at the beginning with `python`. Although it hasn't been checked, it should probably work.

## eval_query.py

Execute comparisons of search speed, initialization speed, and scores or search results with the pre-improvement BM25Retriever.

You can specify which tests to run using arguments. If no arguments are provided, all evaluations will be executed.

```
uv run eval/eval_query.py # Run all tests
uv run eval/eval_query.py --query_speed # Query speed
uv run eval/eval_query.py --init_speed # Initialization speed
uv run eval/eval_query.py --verify_scores # BM25 scores
uv run eval/eval_query.py --verify_retrieval # search results
```

Below are the evaluation results and a brief analysis.

### Search Speed
The search speed of 100,000 English documents is compared using TFIDFRetriever, the pre-improvement BM25Retriever (CurrentBM25Retriever), and the post-improvement BM25Retriever (NewBM25Retriever). Each is executed 10 times, and the average and standard deviation are output.

```
% uv run eval/eval_query.py --query_speed
TFIDFRetriever query time: 0.0315 ± 0.0218 seconds per query
CurrentBM25Retriever query time: 1.1794 ± 0.8022 seconds per query
NewBM25Retriever query time: 0.0271 ± 0.0012 seconds per query
```

The search speed of the improved BM25Retriever is approximately 50 times faster than before the improvement. Additionally, it achieves a speed comparable to the TFIDFRetriever, which also calculates similarity using matrix operations, demonstrating the effectiveness of the improvement.

### Initialization Speed

We compare the initialization speed for 100,000 English documents using the TFIDFRetriever, the pre-improvement BM25Retriever (CurrentBM25Retriever), and the post-improvement BM25Retriever (NewBM25Retriever). Each is executed 10 times, and the average and standard deviation are output.

```
% uv run eval/eval_query.py --init_speed
TFIDFRetriever initialization: 2.9645 ± 0.1880 seconds
CurrentBM25Retriever initialization: 2.4175 ± 0.1267 seconds
NewBM25Retriever initialization: 5.7847 ± 0.2405 seconds
```

The initialization speed of the improved BM25Retriever is 2-3 times slower compared to the pre-improvement version and the TFIDFRetriever. This is likely because the process of calculating the BM25 vectors for the corpus has been added compared to before the improvement. The TFIDFRetriever also calculates the TFIDF vectors for the corpus, but it is thought to be more efficient because it retrieves the idf calculation simultaneously using scikit-learn's fit_transform function. It is possible to adopt a similar method for the improved BM25Retriever, but due to the extensive changes required, it was not adopted this time. If the changes become too extensive, it might be worth considering adopting another library capable of faster calculations than rank-bm25.

### BM25 Score
The BM25 scores calculated by the BM25Retriever before and after the improvement are compared. The objective is to ensure that no differences in scores arise due to differences in calculation methods. BM25 scores are calculated between 100 queries and a corpus of 10,000 documents, and the absolute value of the differences in corresponding values before and after the improvement is computed. Subsequently, the average, standard deviation, maximum, and minimum of these absolute differences are output. Additionally, the average of the BM25 scores is also output.

```
% uv run eval/eval_query.py --verify_score    
Mean difference: 1.0484258183840468e-15
Standard deviation of difference: 1.9738550390587456e-15
Max difference: 1.1368683772161603e-13
Min difference: 0.0
Average of mean scores: 11.353529448685011
```

The implementation of the calculation was changed before and after the improvement, but since the results are expected to be the same, the difference is expected to be zero. As a result, the absolute value of the difference is not zero, but at most on the order of 10e-13, which is negligible compared to the average BM25 score (11). The fact that it is not strictly zero may be due to floating-point rounding errors. As long as it does not affect the ranking of search results at a level that impacts the application, the fact that it is not strictly zero is not a major issue.

### Search Results
The top 100 search results of the NewBM25Retriever and CurrentBM25Retriever in a corpus consisting of 10,000 documents are compared. Each is compared against 100 queries. If the search results differ, the rank at which the results differ and the text and score of the search results around that rank are output for manual confirmation.

```
% uv run eval/eval_query.py --verify_retrieval
Query 4: Content mismatch at result #87
rank: 86
5938 20.632478008166014 Fleisher Takes Lead at Hickory Classic (AP) AP - Bruce Fleisher parlayed eight birdies into a tournament-low 65 Saturday and cruised to a 3-shot lead after the second round of the Greater Hickory Classic.
4061 20.63247800816601 Boxer Bowe to Launch Comeback in Sept. (AP) AP - Former heavyweight champion Riddick Bowe is coming out of retirement after a 17-month prison stint, with a scheduled return to the ring Sept. 25.
rank: 87
4061 20.63247800816601 Boxer Bowe to Launch Comeback in Sept. (AP) AP - Former heavyweight champion Riddick Bowe is coming out of retirement after a 17-month prison stint, with a scheduled return to the ring Sept. 25.
5938 20.63247800816601 Fleisher Takes Lead at Hickory Classic (AP) AP - Bruce Fleisher parlayed eight birdies into a tournament-low 65 Saturday and cruised to a 3-shot lead after the second round of the Greater Hickory Classic.
rank: 88
3228 20.587078082218543 NewsView: Mosque Adds Problems for U.S. (AP) AP - The Bush administration is supporting the Iraqi government as it uses diplomacy and the threat of force to oust radical Shiite cleric Muqtada al-Sadr's militia from a holy shrine in Najaf. At the same time, it is trying to steer clear of becoming the target of an angry Muslim world.
3228 20.58707808221854 NewsView: Mosque Adds Problems for U.S. (AP) AP - The Bush administration is supporting the Iraqi government as it uses diplomacy and the threat of force to oust radical Shiite cleric Muqtada al-Sadr's militia from a holy shrine in Najaf. At the same time, it is trying to steer clear of becoming the target of an angry Muslim world.
...
```

Differences were observed in 6 out of 100 queries (6%). One of the results is shown above. Upon seeing the sentences at the differing ranks, a swap in ranking is observed. Specifically, the 87th rank before the improvement and the 88th rank after the improvement, as well as the 88th rank before the improvement and the 87th rank before the improvement, are the same sentences. These sentences are almost identical, differing by only one word, suggesting that sentences that should have the same BM25 score have a slight score difference due to rounding errors. Therefore, which of the nearly identical sentences ranks higher can be considered a negligible difference in application. It is evident from the absolute score difference being on the order of 10e-13 that such a tendency occurs.

For reference, the query IDs with differing search results and the rank at which the difference was first detected are shown below. In all queries, the same tendency as above was observed, and no differences that would be problematic in application were found.

- Query 4: Mismatch at result #87
- Query 28: Mismatch at result #47
- Query 75: Mismatch at result #99
- Query 77: Mismatch at result #83
- Query 82: Mismatch at result #61
- Query 86: Mismatch at result #6


## eval_merge.py
Performs comparisons of merge speed, scores, and search results with the pre-improvement BM25Retriever.

You can specify the tests to run with arguments. If no arguments are provided, all evaluations will be executed.

```
uv run eval/eval_merge.py # Run all tests
uv run eval/eval_merge.py --merge_speed # Merge speed in English
uv run eval/eval_merge.py --merge_speed_ja # Merge speed in Japanese
uv run eval/eval_merge.py --vectorizer_merge_speed # Vectorizer-only merge speed in English
uv run eval/eval_merge.py --vectorizer_merge_speed_ja # Vectorizer-only merge speed in Japanese
uv run eval/eval_query.py --verify_scores # BM25 scores
uv run eval/eval_merge.py --verify_retrieval # Search results
```

Below are the results of the evaluation script execution and a brief analysis.
### Merge Speed (English)
The execution time is compared when merging two retrievers, each consisting of a sub-corpus of 5000 English documents. The execution times for the improved BM25Retriever's fast merge (BM25Retriever merge time), simple merge (BM25Retriever merge time from documents), and the pre-improvement simple merge (CurrentBM25Retriever merge time from documents) are output. In the simple merge, the corpus is obtained from each retriever, combined, and a new retriever insntace is initialized using the from_documents method.

```
% uv run eval/eval_merge.py --merge_speed     
NewBM25Retriever merge time: 0.3880 ± 0.0370 seconds
CurrentBM25Retriever merge time from documents: 0.2863 ± 0.0536 seconds
NewBM25Retriever merge time from documents: 0.6469 ± 0.0460 seconds
```

The fast merge of the improved BM25Retriever is nearly twice as fast as the simple merge of the same improved retriever. This is considered to be due to the reuse of pre-calculated word frequencies in the sub-corpus, which makes the computation more efficient. On the other hand, it is slower than the simple merge of the existing BM25Retriever. This is because, in the improved BM25Retriever, the calculation of the corpus vector for speeding up search is also performed during the merge. As mentioned later, if the merge is limited to only the vectorizer rather than the entire retriever, it is faster compared to the existing one. Therefore, if you want to improve only the merge speed without enhancing the search speed, it might be desirable to create a retriever that replaces only the vectorizer.

### Merge Speed (Japanese)
Similar to the English version, but outputs the speed when merging two Japanese sub-corpora of approximately 3000 items each (specifically 2946 and 2947 items). The execution time also includes the preprocessing for Japanese morphological analysis (using mecab).

```
% uv run eval/eval_merge.py --merge_speed_ja 
NewBM25Retriever merge time ja: 1.6265 ± 0.0889 seconds
CurrentBM25Retriever merge time ja: 3.7666 ± 0.3905 seconds
NewBM25Retriever merge time from documents ja: 6.0507 ± 0.3864 seconds
```

Unlike the English version, the improved Retriever's fast merge is faster than both the simple merge of the existing Retriever and the simple merge of the improved Retriever. This is because the fast merge of the improved Retriever reuses the preprocessed corpus, allowing for more efficient computation, including preprocessing time, compared to the simple method. This suggests that the current method is particularly effective when there is time-consuming preprocessing, such as Japanese morphological analysis.

### Vectorizer Merge Speed (English)
We compare the execution time when merging two vectorizers, each consisting of a sub-corpus of 5000 English documents. The purpose is to evaluate the impact of vectorizer merge speed on retriever merge speed, focusing solely on the vectorizer rather than the entire retriever. The comparison conditions are the same as those for comparing retriever merge speeds.

```
% uv run eval/eval_merge.py --vectorizer_merge_speed
NewBM25Retriever vectorizer merge time: 0.0386 ± 0.0084 seconds
CurrentBM25Retriever vectorizer merge time from documents: 0.1413 ± 0.0136 seconds
NewBM25Retriever vectorizer merge time from documents: 0.1365 ± 0.0047 seconds
```

Focusing solely on the vectorizer, it can be seen that the improved fast merge method is approximately 3-4 times faster than the simple method.

### Vectorizer Merge Speed (Japanese)
The execution time when merging two Japanese sub-corpora of slightly approxymately 3000 items each (specifically 2946 and 2947 items) is output. The comparison conditions are the same as those for comparing retriever merge speeds.

```
% uv run eval/eval_merge.py --vectorizer_merge_speed_ja
NewBM25Retriever vectorizer merge time ja: 0.0582 ± 0.0033 seconds
CurrentBM25Retriever vectorizer merge time from documents ja: 0.7922 ± 0.0772 seconds
NewBM25Retriever vectorizer merge time from documents ja: 0.7191 ± 0.0320 seconds
```

As with the retriever, in situations like Japanese where preprocessing takes time, a significant difference in merge speed can be observed.

### BM25 Score
The BM25 scores calculated by vectorizers created using fast and simple merge methods are compared. The purpose is to confirm that there is no difference in the calculation results depending on the merge method. The conditions are the same as those used in the evaluation conducted in the section on search speed improvement. The BM25 scores between 100 queries and a corpus of 10,000 items are calculated, and the absolute difference between the corresponding values before and after the improvement is computed. Then, the average, standard deviation, maximum, and minimum of these absolute differences are output. Additionally, the average of the BM25 scores is also output.

```
% uv run eval/eval_merge.py --verify_score  
Mean difference: 1.0497634150841152e-15
Standard deviation of difference: 1.9689170804663834e-15
Max difference: 1.1368683772161603e-13
Min difference: 0.0
Average of mean scores: 11.353529448685011
```

The results are exactly the same as those obtained in the section on search speed improvement, indicating that the score differences due to different merge methods are negligible.

### Search Results
As with the BM25 scores, the results are exactly the same as those obtained in the section on search speed improvement, so they are omitted here.

## For Developers

uv is used for package development.

```
uv run pytest
```