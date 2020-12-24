import streamlit as st
from bert_serving.client import BertClient
from elasticsearch import Elasticsearch

SEARCH_SIZE = 10


@st.cache
def query(query: str):
    bc = BertClient(output_fmt='list')
    es = Elasticsearch('localhost:9200')

    query_vector = bc.encode([query])[0]
    script_params = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source":
                    "cosineSimilarity(params.query_vector, 'bert_embedding'"
                    ") + 1.0",
                "params": {
                    "query_vector": query_vector
                }
            }
        }
    }

    docs = es.search(index="documents",
                     body={
                         "size": SEARCH_SIZE,
                         "query": script_params,
                         "_source": {
                             "includes": ["pid", "passage"]
                         }
                     })
    return docs


def parse_result(json):
    results = json['hits']['hits']
    assert len(results) == SEARCH_SIZE
    for hit in results:
        yield hit['_source']


if __name__ == "__main__":
    st.title("Query information retrieval featuring ElasticSearch and "
             "Bert-as-service!")
    query_input = st.text_input("What question do you want to ask?",
                                max_chars=100)
    if st.button('Look ! '):
        with st.spinner("Looking zzzzzz"):
            res = query(query_input)
        parsed_res = parse_result(res)
        for n, doc in enumerate(parsed_res):
            st.markdown(f"*{n}*: **{doc.get('passage')}**")
            st.write("\n")
