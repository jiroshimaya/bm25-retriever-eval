from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from collections import Counter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "Could not import rank_bm25, please install with `pip install "
        "rank_bm25`."
    )
try:
    from scipy.sparse import csr_matrix
except ImportError:
    raise ImportError(
        "Could not import scipy, please install with `pip install "
        "scipy`."
    )

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

    
class BM25Vectorizer(BM25Okapi):
    def __init__(self, corpus, **bm25_params):            
        super().__init__(corpus, **bm25_params)
        self.vocabulary = list(self.idf.keys())
        self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}

    #override
    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self.nd = nd # add this line
        return nd
            
    def transform(self, queries: list[list[str]]) -> scipy.sparse.csr_matrix:
        
        rows = []
        cols = []
        data = []
        
        for i, query in enumerate(queries):
            query_len = len(query)
            query_count = Counter(query)
            
            for word, count in query_count.items():
                if word in self.word_to_id:
                    word_id = self.word_to_id[word]
                    tf = count
                    idf = self.idf.get(word, 0)
                    
                    # BM25 scoring formula
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * query_len / self.avgdl)
                    
                    score = numerator / denominator
                    
                    rows.append(i)
                    cols.append(word_id)
                    data.append(score)
        
        return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))
            
    def count_transform(self, queries: list[list[str]]) -> scipy.sparse.csr_matrix:
        
        rows = []
        cols = []
        data = []
        
        for i, query in enumerate(queries):
            for word in query:
                if word in self.word_to_id:
                    word_id = self.word_to_id[word]
                    rows.append(i)
                    cols.append(word_id)
                    data.append(1)  # Count is always 1 for each occurrence
        
        return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))


class BM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    preprocessed_docs: List[List[str]] = Field(repr=False)
    """ List of Preprocessed documents."""
    bm25_array: Any = None
    """BM25 array."""
    bm25_params: Optional[Dict[str, Any]] = None
    """ Parameters to pass to the BM25 vectorizer."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Vectorizer(texts_processed, **bm25_params)
        bm25_array = vectorizer.transform(texts_processed)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, bm25_array=bm25_array, preprocess_func=preprocess_func, 
            preprocessed_docs=texts_processed, bm25_params=bm25_params, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        query_vec = self.vectorizer.count_transform([processed_query])
        results = query_vec.dot(self.bm25_array.T).toarray()[0]
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        return return_docs

def merge_bm25_vectorizers(vectorizer_list: list[BM25Vectorizer]) -> BM25Vectorizer:
    if not vectorizer_list:
        raise ValueError("The vectorizer_list is empty")

    # Combine all tokenized corpora
    merged_corpus_size = 0
    merged_doc_freqs = []
    merged_doc_len = []
    
    for vectorizer in vectorizer_list:
        merged_corpus_size += vectorizer.corpus_size
        merged_doc_freqs += vectorizer.doc_freqs
        merged_doc_len += vectorizer.doc_len
    
    merged_nd = {}
    for vectorizer in vectorizer_list:
        for word, nd in vectorizer.nd.items():
            merged_nd[word] = merged_nd.get(word, 0) + nd
            
    merged_vectorizer = BM25Vectorizer(["a"]
                            , tokenizer = vectorizer_list[0].tokenizer
                            , k1 = vectorizer_list[0].k1
                            , b = vectorizer_list[0].b
                            , epsilon = vectorizer_list[0].epsilon)
    
    merged_vectorizer.corpus_size = merged_corpus_size
    merged_vectorizer.doc_freqs = merged_doc_freqs
    merged_vectorizer.doc_len = merged_doc_len
    merged_vectorizer.avgdl = sum(merged_doc_len) / merged_corpus_size
    merged_vectorizer.nd = merged_nd

    merged_vectorizer._calc_idf(merged_nd)
    merged_vectorizer.vocabulary = list(merged_vectorizer.idf.keys())
    merged_vectorizer.word_to_id = {word: i for i, word in enumerate(merged_vectorizer.vocabulary)}
    
    return merged_vectorizer

def merge_bm25_retrievers(retriever_list: list[BM25Retriever]) -> BM25Retriever:
    if not retriever_list:
        raise ValueError("The retriever_list is empty")
    
    merged_docs = []
    merged_preprocessed_docs = []
    for retriever in retriever_list:
      merged_docs += retriever.docs
      merged_preprocessed_docs += retriever.preprocessed_docs
    
    vectorizer_list = [retriever.vectorizer for retriever in retriever_list]
    merged_vectorizer = merge_bm25_vectorizers(vectorizer_list)

    merged_retriever = BM25Retriever(
      vectorizer = merged_vectorizer,
      docs = merged_docs,
      bm25_array = merged_vectorizer.transform(merged_preprocessed_docs),
      k = retriever_list[0].k,
      preprocess_func = retriever_list[0].preprocess_func,
      preprocessed_docs = merged_preprocessed_docs,
      bm25_params = retriever_list[0].bm25_params or {}
    )
    return merged_retriever
