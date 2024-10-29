from fastretriever import BM25Retriever, merge_bm25_retrievers

def test_merge_bm25_retrievers():
    import MeCab

    mecab = MeCab.Tagger("-Owakati")
    def preprocess_func(text: str) -> list[str]:
        return mecab.parse(text).strip().split()
      
    examples = [
        "今日はいい天気ですね。",
        "明日は雨が降るかもしれません。",
        "昨日の晩ご飯は美味しかったです。",
        "週末は友達と映画を見に行きます。",
        "新しいプロジェクトが始まりました。",
        "最近、運動を始めました。",
        "次の休暇はどこに行こうか考えています。",
        "この本はとても面白いです。",
        "音楽を聴くのが好きです。",
        "毎朝、コーヒーを飲みます。"
    ]
    examples2 = [
        "今日は新しいレストランに行きました。",
        "彼は毎日ジョギングをしています。",
        "この映画はとても感動的でした。",
        "週末に家族とキャンプに行きます。",
        "新しいスマートフォンを買いました。",
        "最近、料理を始めました。",
        "次の旅行はどこに行こうか考えています。",
        "この音楽はとてもリラックスできます。",
        "読書が趣味です。",
        "毎晩、紅茶を飲みます。"
    ]

    retriever0 = BM25Retriever.from_texts(examples, preprocess_func=preprocess_func)
    retriever1 =  BM25Retriever.from_texts(examples2, preprocess_func=preprocess_func)
    merged_retriever = merge_bm25_retrievers([retriever0, retriever1])
    
    docs = merged_retriever.invoke("いい天気")
    assert docs[0].page_content == "今日はいい天気ですね。"
    
    docs = merged_retriever.invoke("紅茶")
    assert docs[0].page_content == "毎晩、紅茶を飲みます。"


test_merge_bm25_retrievers()