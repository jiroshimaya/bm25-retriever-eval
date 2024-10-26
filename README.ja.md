# BM25Retrieverの改修と評価

#### [English](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.md) | [日本語](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.ja.md)

このリポジトリは、langchainのBM25Retrieverの改良実装と、それにより達成された検索速度改善等の評価結果を提供します。

100,000件のドキュメントのコーパスで評価したところ、改良されたBM25Retrieverは初期化に2〜3倍の時間がかかりましたが、検索速度は元の実装よりも約50倍速くなりました。

コードの変更により、既存の実装と比較して検索結果が若干異なる場合がありましたが、その影響はほとんどの実際のアプリケーションにおいて無視できると考えられます。

これらの結果を踏まえると、この実装は初期化時間よりもクエリごとの検索速度が重視されるシナリオ、つまり、ほとんどのシナリオで有用と考えられます。

## 結果

### 検索時間

TFIDFRetriever、既存のBM25Retriever（CurrentBM25Retriever）、改良後のBM25Retriever（NewBM25Retriever）の検索時間を比較しました。検索対象の文書数が10万件の場合に、10回の検索を行った際の平均値と標準偏差を示します。

| Retriever | Time (seconds per query) |
|-----------|--------------------------|
| TFIDFRetriever | 0.0310 ± 0.0201 |
| CurrentBM25Retriever | 1.2280 ± 0.8320 |
| NewBM25Retriever | 0.0282 ± 0.0016 |



### 初期化時間

本改良が、Retrieverの初期化時間に与える影響を調べるため、TFIDFRetriever、既存のBM25Retriever（CurrentBM25Retriever）、改良後のBM25Retriever（NewBM25Retriever）の初期化に要した時間を比較しました。  
検索対象の文書数が10万件の場合に、10回の初期化を実施した際の、平均値と標準偏差を示します。

| Retriever | Time (seconds) |
|-----------|----------------|
| TFIDFRetriever | 3.0546 ± 0.1234 |
| CurrentBM25Retriever | 2.5157 ± 0.1372 |
| NewBM25Retriever | 7.4195 ± 0.1648 |



### 検索結果
改良前後で、同一の検索対象文書、クエリにおける検索結果がどの程度異なるか調べました。
#### 検索結果文書
10,000件の文書からなるコーパスにおけるNewBM25RetrieverとCurrentBM25Retrieverのトップ100の検索結果を比較しました。テストした100件のクエリのうち、6件のクエリ（6%）で結果に違いが見られました。それぞれのクエリで、最も高い順位で何位の結果が異なっているか調べました。結果は以下です。

- Query 4: Mismatch at result #87
- Query 28: Mismatch at result #47
- Query 75: Mismatch at result #99
- Query 77: Mismatch at result #83
- Query 82: Mismatch at result #61
- Query 86: Mismatch at result #6

#### スコア
改良前後のRetireverによって得られるBM25スコアを比較しました。NewBM25Retrieverによって計算されたBM25スコア（事前に計算されたBM25ベクトルとクエリの単語頻度ベクトルを使用）を、CurrentBM25Retrieverによって計算されたスコア（rank-bm25のget_scores関数を使用）と比較しました。10,000件の文書からなるコーパスに対して、両方の方法でスコアを計算し、その差を分析しました。同一クエリに対する2つのスコアの差の絶対値の平均差は1.04e-15、標準偏差は1.97e-15、差の絶対値の最大は1.13e-13でした。これらはスコアの平均値11.35と比較して非常に小さい値でした。


## 考察

1. **検索時間**: NewBM25Retrieverは、CurrentBM25Retrieverと比較して約50倍の速度でき、TFIDFRetrieverと同程度の速度でした。期待通り高速化ができていると言えます。

1. **初期化**: NewBM25Retrieverは初期化時間が最も遅く、他のリトリーバーの2〜3倍の時間がかかりました。これは改良によるデメリットと考えられます。しかし、多くのユースケースにおいてRetrieverの初期化は一度だけ、前もって実施すればよいため、クエリごとの検索速度が大幅に向上していることを踏まえれば許容範囲と考えられます。

3. **検索結果文書とBM25スコアの検証**: 結果から、改良前後のBM25RetrieverによるBM25スコア差は無視できる程度に小さいことがわかりました。完全にゼロでない理由は、浮動小数点の丸め誤差等に起因するものと考えられます。クエリの6%でトップ100の結果に違いが見られましたが、ほとんどの不一致は50位以下で発生しており、上位の検索結果のみを用いる応用ではほとんど影響がないと考えられます。差の最大値が10e-14のオーダーであることを踏まえると、この異なりは、BM25スコアが0付近のときに起こっていると考えられます。

## 実行手順

1. インストール
   ```
   pip install langchain langchain-community==0.3.1 scipy rank-bm25 scikit-learn datasets tqdm
   ```

2. 全体テスト実行
   ```
   python eval.py
   ```

   引数を省略した場合、初期化速度、クエリ速度、検索結果の検証、BM25スコアの検証を含むすべてのテストがデフォルトで実行されます。

3. 個別テスト実行
   - 初期化速度:
     ```
     python eval.py --init_speed
     ```
   - 検索速度:
     ```
     python eval.py --query_speed
     ```
   - 検索結果:
     ```
     python eval.py --verify_retrieval
     ```
   - BM25スコア:
     ```
     python eval.py --verify_scores
     ```