# BM25Retrieverの改良と評価

このリポジトリでは、langchainのBM25Retrieverを高速化した実装とパフォーマンス評価を公開します。

#### [English](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.md) | [日本語](https://github.com/jiroshimaya/bm25-retriever-eval/blob/main/README.ja.md)

# 想定される利用シーン
本実装は以下のようなケースで有効な可能性が高いです。

- **検索速度の向上**: langchain.BM25Retrieverを使用しており、検索速度を向上したい
- **マージ速度の向上**: 検索対象文書を動的に変更する用途などにより、BM25Retrieverを高速にマージしたい
- **プロトタイピング**: キーワード検索のプロトタイピングを迅速にする必要があり、データベースなどを別途構築したくない、かつ、検索対象件数が10万件以上である

本実装により以下のメリット、デメリットが得られます。パフォーマンスは開発者のラップトップ上での検証によるものです。

メリット
- マージ速度の向上（コーパス約1万件のとき、日本語で約2-3倍、英語で約0.7-1.5倍）
- 検索速度の向上（コーパス10万件のとき約40-50倍）

デメリット
- 初期化速度の低下（コーパス10万件のとき0.3-0.7倍）

上記の結果を踏まえると、初期化よりも検索の速度が重要な場合やマージの頻度が多い場合に本改善が有効であると考えられます。

# 使い方

## インストール
通常のlangchain.BM25Retrieverを使うために必要なlangchain-community、rank-bm25に加え、スパース行列の高速演算のためにscipyをインストールしてください。langchain-communityは0.3.1で動作確認していますが、BM25Retrieverに破壊的な変更が生じない限りは最新版でも動くと思います。

```
pip install langchain-community rank-bm25 scipy
```

## BM25Retriever
`src/fastretriever/bm25.py`をプロジェクトにコピーし、スクリプトから適切なパスで呼び出してください。
その後、通常のBM25Retrieverとほぼ同様に使えます。
つまり、from_textsまたはfrom_documentsで初期化したあと、invoke関数で検索を実行できます。

以下はbm25.pyを呼び出し側のスクリプトと同じ階層においたときのコード例です。

```Python
from bm25 import BM25Retriever

retriever = BM25Retriever.from_texts(["Hello", "Good evening"])
docs = retriever.invoke("Good")
print(docs)
```

```
[Document(metadata={}, page_content='Good evening'), Document(metadata={}, page_content='Hello')]
```

from_textsやfrom_documentsを使わず、クラスを直接初期化する場合は、追加の引数が必要になります。詳細はソースコードをご確認ください。

## マージ
bm25.pyからmerge_bm25_retrievers関数を呼び出して使用します。
merge_bm25_retrieversに与えるretrieverクラスは通常のBM25Retrieverではなくbm25.pyのBM25Retrieverである必要があります。

```Python
from fastretriever import BM25Retriever, merge_bm25_retrievers
retriever0 = BM25Retriever.from_texts(["Hello"])
retriever1 = BM25Retriever.from_texts(["Good evening"])
merged_retriever = merge_bm25_retrievers([retriever0, retriever1])
docs = merged_retriever.invoke("Good")
print(docs)
```

マージの高速化は日本語の分かち書きなど時間がかかる前処理がpreprocess_funcに指定されている場合により効果を発揮します。

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

# 実装のポイント
## 検索速度改善
BM25Retrieverの内部で使用されるrank-bm25が遅かったためrank-bm25.BM25Okapiというクラスをオーバーライドして高速化しました。具体的には、BM25スコアを検索のたびに辞書のループで計算しているところが遅さの原因だったため、コーパスのBM25重みベクトルを事前に計算しておき、検索時には、クエリの単語頻度ベクトルとコーパスベクトルの内積を取るだけですむようにしました。ただし、初期化時にコーパスの重みベクトルを作る必要があるため、改良前に比べ初期化の速度は遅くなりました。

## マージ速度改善
BM25Retrieverには標準的なマージ方法が存在しないため、マージ後のコーパスを与えて初期化する必要がありました。
これでは、マージ前のretrieverですでに計算済みの単語頻度辞書や実施済みの前処理を活用できず非効率でした。
そこで、改良後のBM25Retrieverでは単語頻度辞書や前処理済みのコーパスを外部から取得できるようにクラスのプラパティとして保持するようにしました。そのうえで、それらを使いながらマージする関数を実装しました。
感覚的には単語頻度辞書の再利用によって計算量が単語数から単語の種類数に減るため、コーパス件数が増えるほどマージ速度の改善幅は大きくなります。

# 評価
`eval_query.py`によって検索速度を、`eval_merge.py`によってマージ速度を評価できます。

実行するためには、リポジトリをクローンしてリポジトリルートに移動してください。

```
git clone [repository url]
cd bm25-eval
```

パッケージ管理やスクリプト実行にuvを使用しています。
uvを使用しない場合、依存関係ライブラリとパッケージインストールのために以下を実行してください。uvを使用する場合は不要です。

```
pip install datasets tqdm mecab-python3 unidic-lite scikit-learn
pip install -e .
```

以降、uvの使用を前提としてコマンドを記載します。uvを使用しない場合は先頭の`uv run`を`python`に変更すると、チェックしていませんが、おそらく動くかと思います。

## eval_query.py

検索速度、初期化速度、改良前のBM25Retrieverとのスコアや検索結果の比較を実行します。

引数によって実行するテストを指定できます。引数なしの場合、すべての評価を実行します。

```
uv run eval/eval_query.py #すべてのテストを実行
uv run eval/eval_query.py --query_speed # 検索速度
uv run eval/eval_query.py --init_speed # 初期化速度
uv run eval/eval_query.py --verify_scores # bm25スコア
uv run eval/eval_query.py --verify_retrieval # 検索結果
```

以下、評価の実行結果と簡単な考察です。

### 検索速度
10万件の英文の検索速度をTFIDFRetriever、改良前のBM25Retriever（CurrentBM25Retriever）、改良後のBM25Retriever（NewBM25Retriever）で比較します。それぞれ10回施行し、平均と標準偏差を出力します。

```
% uv run eval/eval_query.py --query_speed
TFIDFRetriever query time: 0.0315 ± 0.0218 seconds per query
CurrentBM25Retriever query time: 1.1794 ± 0.8022 seconds per query
NewBM25Retriever query time: 0.0271 ± 0.0012 seconds per query
```

改良後のBM25Retrieverの検索速度は改良前の40-50倍早いです。
また、改良後と同様に行列演算で類似度を計算しているTFIDFRetrieverと同等の速度となっており、改良の有効性を示しています。

### 初期加速度

10万件の英文での初期化速度をTFIDFRetriever、改良前のBM25Retriever（CurrentBM25Retriever）、改良後のBM25Retriever（NewBM25Retriever）で比較します。それぞれ10回施行し、平均と標準偏差を出力します。

```
% uv run eval/eval_query.py --init_speed
TFIDFRetriever initialization: 2.9645 ± 0.1880 seconds
CurrentBM25Retriever initialization: 2.4175 ± 0.1267 seconds
NewBM25Retriever initialization: 5.7847 ± 0.2405 seconds
```

改良後のBM25Retrieverの初期加速度は改良前やTFIDFRetrieverと比べ2-3倍遅くなっています。
これはコーパスのBM25ベクトルを計算する処理が改良前と比べて追加されたためと考えられます。TFIDFRetrieverも同様にコーパスのTFIDFベクトルを計算してはいるのですがscikit-learnのfit_transform関数でidfの計算と同時に取得しているため効率化されているのだと考えられます。改良後のBM25Retrieverでも同様の方法を採用することも可能と思われますが、変更箇所が多くなりすぎるため、今回は採用していません。あまりに変更箇所が多くなる場合、rank-bm25よりも高速な計算が可能な別ライブラリを採用することも視野に入ってくると思われます。

### BM25スコア
改良前後のBM25Retrieverで計算されるBM25スコアを比較します。計算方法の違いによってスコアに差異が生じないことを確認することが目的です。100件のクエリと1万件のコーパス間のBM25スコアを計算し、改良前後の対応する値の差の絶対値を計算します。その後、その差の絶対値の平均、標準偏差、最大値、最小値を出力します。また、BM25スコアの平均値も出力します。

```
% uv run eval/eval_query.py --verify_score    
Mean difference: 1.0484258183840468e-15
Standard deviation of difference: 1.9738550390587456e-15
Max difference: 1.1368683772161603e-13
Min difference: 0.0
Average of mean scores: 11.353529448685011
```

改良前後で計算の実装に変更を加えましたが結果は同じになることを期待しているため、差はゼロになることが期待されます。実際には、差の絶対値はゼロではありませんが最大でも10e-13オーダーであり、これはBM25スコアの平均値（11）に比べて、無視できる程度の差と言えます。厳密にゼロにならないのは、浮動小数点の丸め誤差が原因かもしれません。アプリケーションに影響するレベルで検索結果の順位に影響を与えなければ、厳密にゼロでないことは大きな問題ではありません。

### 検索結果
10,000件の文書からなるコーパスにおけるNewBM25RetrieverとCurrentBM25Retrieverのトップ100の検索結果を比較します。100件のクエリに対してそれぞれ比較します。検索結果が異なっていた場合、何位の結果が異なっていたかと、目視確認用に、その順位付近の検索結果の文章とスコアを出力します。

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

100件中6件のクエリ（6%）で結果に違いが見られました。そのうち1件の結果を上記に載せています。
違いが見られた順位の文章を目視すると、順位の入れ替わりが発生しています。具体的には改良前の87位と改良後の88位、改良前の88位と改良前の87位が同一の文章です。これらの文章は1単語だけ異なる程度のほぼ同一の文章であることから、本来はBM25スコアが同一になるはずの異なる文章が丸め誤差によってごく僅かなスコア差がついた状況と考えられます。したがって、ほぼ同一の文章のどちらが上位に来るかは応用上、無視できる差異と言えます。スコアの差の絶対値が10e-13オーダーであることからもこのような傾向になることは自明と考えられます。

参考までに、検索結果が異なったクエリのIDと検索結果の異なりが最初に検出された順位を以下に示します。いずれのクエリでも上記と同様の傾向であり、応用上、問題のある差異は認められませんでした。

- Query 4: Mismatch at result #87
- Query 28: Mismatch at result #47
- Query 75: Mismatch at result #99
- Query 77: Mismatch at result #83
- Query 82: Mismatch at result #61
- Query 86: Mismatch at result #6


## eval_merge.py
マージの速度、改良前のBM25Retrieverとのスコアや検索結果の比較を実行します。

引数によって実行するテストを指定できます。引数なしの場合、すべての評価を実行します。

```
uv run eval/eval_merge.py #すべてのテストを実行
uv run eval/eval_merge.py --merge_speed # 英語のマージ速度
uv run eval/eval_merge.py --merge_speed_ja # 日本語のマージ速度
uv run eval/eval_merge.py --vectorizer_merge_speed # ベクトライザのみの英語のマージ速度
uv run eval/eval_merge.py --vectorizer_merge_speed_ja # ベクトライザのみの日本語のマージ速度
uv run eval/eval_query.py --verify_scores # bm25スコア
uv run eval/eval_merge.py --verify_retrieval # 検索結果
```

以下、評価スクリプトの実行結果と簡単な考察です。
### マージ速度（英語）
英文5000件のサブコーパスからなる2つのretrieverをマージするときの実行時間を比較します。改良後のBM25Retrieverの高速マージ（BM25Retriever merge time）、シンプルなマージ（BM25Retriever merge time from documents）、改良前のシンプルなマージ（CurrentBM25Retriever merge time from documents）の実行時間を出力します。シンプルなマージでは各retrieverからコーパスを取得して結合し、from_documentsメソッドで新たなretrieverを初期化しています。

```
% uv run eval/eval_merge.py --merge_speed     
NewBM25Retriever merge time: 0.3880 ± 0.0370 seconds
CurrentBM25Retriever merge time from documents: 0.2863 ± 0.0536 seconds
NewBM25Retriever merge time from documents: 0.6469 ± 0.0460 seconds
```

改良後のBM25Retrieverの高速マージは改良後の同Retrieverのシンプルなマージより2倍弱高速です。これはサブコーパスで計算済みの単語頻度を使い回すことで、計算が効率化できたためと考えられます。
一方で、既存のBM25Retrieverのシンプルなマージよりは、遅くなっています。これは改良後のBM25Retireverでは検索速度を高速化するためのコーパスベクトルの計算がマージの際にも行われるためと考えられます。後述の通りRetirever全体ではなくVectorizerのみのマージに限定すると既存のものと比べても高速化されているため、検索速度の向上が不要でマージの速度だけを向上したい場合は、ベクトライザのみを置き換えたRetrieverを作成するのが望ましいかもしれません。

### マージ速度（日本語）
英語版とほぼ同様ですが、3000件弱の日本語サブコーパス2つ（厳密には2946件と2947件）をマージする際の速度を出力します。日本語のため形態素解析の前処理（mecabを使用）も実行時間に含まれます。

```
% uv run eval/eval_merge.py --merge_speed_ja 
NewBM25Retriever merge time ja: 1.6265 ± 0.0889 seconds
CurrentBM25Retriever merge time ja: 3.7666 ± 0.3905 seconds
NewBM25Retriever merge time from documents ja: 6.0507 ± 0.3864 seconds
```

英語版と異なり、改良後のRetrieverの高速なマージは、既存Retrieverのシンプルなマージ、改良後のRetrieverのシンプルなマージのどちらよりも高速化されています。これは改良後のRetrieverの高速なマージでは前処理済みのコーパスを再利用するため、前処理の時間も込でシンプルな方法よりも計算を効率化できたためと考えられます。
日本語の形態素解析のように、時間のかかる前処理が存在する場合は特に、今回の方法が有効であることが示唆されます。

### ベクトライザのマージ速度（英語）
英文5000件のサブコーパスからなる2つのベクトライザをマージするときの実行時間を比較します。
retrieverのマージ速度への影響はベクトライザのマージ速度の影響が大きいため、retriever全体ではなくベクトライザのみに絞った評価をすることが目的です。比較条件はretrieverのマージ速度の比較と同様です。

```
% uv run eval/eval_merge.py --vectorizer_merge_speed
NewBM25Retriever vectorizer merge time: 0.0386 ± 0.0084 seconds
CurrentBM25Retriever vectorizer merge time from documents: 0.1413 ± 0.0136 seconds
NewBM25Retriever vectorizer merge time from documents: 0.1365 ± 0.0047 seconds
```

ベクトライザのみに絞ると、改良後の高速なマージ手法がシンプルな方法に比べて3-4倍程度高速であることがわかります。

### ベクトライザのマージ速度（日本語）
3000件弱の日本語サブコーパス2つ（厳密には2946件と2947件）をマージするときの実行時間を出力します。
比較条件はretrieverのマージ速度の比較と同様です。

```
% uv run eval/eval_merge.py --vectorizer_merge_speed_ja
NewBM25Retriever vectorizer merge time ja: 0.0582 ± 0.0033 seconds
CurrentBM25Retriever vectorizer merge time from documents ja: 0.7922 ± 0.0772 seconds
NewBM25Retriever vectorizer merge time from documents ja: 0.7191 ± 0.0320 seconds
```

retrieverのときと同様に、前処理に時間がかかる日本語のような状況では、マージ速度により顕著な差が生じる事がわかります。

### BM25スコア
高速なマージ手法とシンプルなマージ手法によって作成されたベクトライザによって計算されるBM25スコアを比較します。マージの方法によって、計算結果に差異がないことを確認することが目的です。
条件等は検索速度改善の節で実施した評価と同一です。100件のクエリと1万件のコーパス間のBM25スコアを計算し、改良前後の対応する値の差の絶対値を計算します。その後、その差の絶対値の平均、標準偏差、最大値、最小値を出力します。また、BM25スコアの平均値も出力します。

```
% uv run eval/eval_merge.py --verify_score  
Mean difference: 1.0497634150841152e-15
Standard deviation of difference: 1.9689170804663834e-15
Max difference: 1.1368683772161603e-13
Min difference: 0.0
Average of mean scores: 11.353529448685011
```

検索速度改善の節で実施した結果と全く同じ結果となっており、マージ方法の違いによるスコアの差は無視できる程度であることがわかります。

### 検索結果
BM25スコアと同様に、検索速度改善の節で実施した結果と全く同じ結果であるため、記載を省略します。

## 開発者向け

パッケージ開発にuvを使用しています。

```
uv run pytest
```
