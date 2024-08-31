# はじめに

本記事では，前回の記事で少し述べた，Graph RAGについて書いていこうと思います．ただし，RAGの部分はうまくいかなかったのでKnowledge Graphを構築するまでの過程について記載いたします．


# Graph RAGとは

そもそもGraph RAGの意義は，RAG(検索拡張生成：Retrieval-Augmented Generation)という，ドキュメントをベクトルにして，それをもとに回答させる手法を発展させることです．RAGでは，そのドキュメントに含まれる固有名詞の背景や関係性までを汲み取ることができません．指示代名詞が何を指しているのかも前後の文脈で分かる部分であるため，これらを明確にするためGraphを用いるアプローチです．つまり，関係性をわかるようにして，それを判断材料に含めてLLMに回答させることで，より精度の良い結果を得ることを目指しています．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/c0c7020c-1321-801a-92c9-d69967547d89.png)


# お題

今回も前回と同様，私の最も好きな漫画である「進撃の巨人」を題材にします．元となるドキュメントは，[進撃の巨人Wikipedia](https://ja.wikipedia.org/wiki/%E9%80%B2%E6%92%83%E3%81%AE%E5%B7%A8%E4%BA%BA)から物語に関連する部分を自分で抜粋したものです．


# Graphの構築

[こちら](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)を参考に，Langchainとneo4jを用いてGraphを構築します．neo4jはローカルのコンテナで動かします．

<details><summary>neo4jのコンテナ</summary>

```yaml:compose.yml
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j
    ports:
      - 7474:7474
      - 7687:7687
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
      - NEO4JLABS_PLUGINS=["apoc"]
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_uuid_enabled=true
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_security_procedures_whitelist=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_default__listen__address=0.0.0.0
      - NEO4J_dbms_connector_bolt_listen__address=:7687
      - NEO4J_dbms_connector_http_listen__address=:7474
      - NEO4J_dbms_connector_bolt_advertised__address=:7687
      - NEO4J_dbms_connector_http_advertised__address=:7474
      - NEO4J_dbms_allow__upgrade=true
      - NEO4J_dbms_default__database=neo4j
    volumes:
      - ./volumes/neo4j/data:/data
      - ./volumes/neo4j/plugins:/plugins
      - ./volumes/neo4j/logs:/logs
      - ./volumes/neo4j/import:/import
      - ./volumes/neo4j/init:/init
      - ./volumes/neo4j/conf:/conf

```
</details>


`LLMGraphTransformer`を用いてドキュメントをGraphに変換し，それを上記で起動したコンテナに格納します．
ドキュメントは適当にチャンクを区切り，`LLMGraphTransformer`もテンプレート通りにしているのみです．

```python:main.py(抜粋)
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

from src.constants import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


graph = Neo4jGraph(
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)
llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(texts)
graph.add_graph_documents(graph_documents)
```

:::note warn
モデルはgpt-4oを用いました．
VertexAIだと，`LLMGraphTransformer`でエラーが出てしまいました．モデルをgpt-3.5でやってみると，実行はできたものの，うまくEntityの抽出をしてくれませんでした．
:::


## こんな感じ

以下が全体像です．`http://localhost:7474/browser/`にアクセスすると，neo4jの画面NodeとEdgeが無数に広がって何がなんだかわかりませんが，ひとまずこれが今回用いたドキュメントをGraphにしたものになります．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/5fcce529-e4e1-da0d-e00d-9dcd4e11cb2b.png)


主人公である「エレン」(中央の青丸)に絞って抽出してみるとこのようになります．エレンと関連するNodeがたくさんあることがわかります．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/ccf0d927-8590-294c-bfe6-c685aaeca8a1.png)

更に絞って，「エレン」と「アルミン」の関係を示したのが下図(左がエレン，右がアルミン)です．物語の中で，二人の間にはいろいろなことがあったため，関係も複数結びついていることがわかります．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/184ce750-67f0-1579-f8b2-4240d46710b1.png)




# 所感
## Graph構築でうまくいかなかった点

gpt-4oを用いれば何も考えず，テンプレート通りに実行することで，Graphを構築することができました．しかしそれゆえに，見直すべきところがいくつかあります．特に，同義のNodeが複数できることは大きな問題だと感じました．
「アルミン」と「アルミン・アルレルト」それぞれでNodeができていますが，これは同一人物です．それぞれで関係が形成されてしまい，これをこのままGraph RAGに用いてしまうと，結局精度の良い回答は得られないでしょう．


## 料金

今回のドキュメント(約23,000字)でGraphを構築するのに，およそ$2.6かかりました．個人的には安いとは感じず，やみくもにするものではないかなと思います．


# まとめ
今回はLangchainとneo4jを用いてGraphを構築しました．Graph自体は簡単に構築することができ，眺めるだけでも面白かったです．しかし，記載はしませんでしたが，本来の目的である，構築したGraphを用いてLLMに回答させることを試みたところ，思うような回答が得られませんでした．各NodeのPropertyをもう少し丁寧に付与するなど前処理が必要だと考えられます．ここはもう少し工夫の余地があるので，これから改善しようと思います．
また，Microsoftが，[graphrag](https://microsoft.github.io/graphrag/)というフレームワークを公開しているのでこちらを触ってみようと思います．

用いたコードは[こちら](https://github.com/rxmrsd/simple-graph)に格納しております．


# 参考

- [進撃の巨人Wikipedia](https://ja.wikipedia.org/wiki/%E9%80%B2%E6%92%83%E3%81%AE%E5%B7%A8%E4%BA%BA)
- [Enhancing RAG-based application accuracy by constructing and leveraging knowledge graphs](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)
- [graphrag](https://microsoft.github.io/graphrag/)
- [LLMによるナレッジグラフの作成とハイブリッド検索 + RAG](https://zenn.dev/yumefuku/articles/llm-neo4j-hybrid#1.-llm%E3%82%92%E4%BD%BF%E3%81%84%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%8B%E3%82%89%E3%82%B0%E3%83%A9%E3%83%95%E3%82%92%E7%94%9F%E6%88%90)