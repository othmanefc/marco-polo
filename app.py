import streamlit as st

from src.predict import Predict

SEARCH_SIZE = 10
pred = Predict()


@st.cache
<<<<<<< HEAD
def search(pred: Predict, query: str):
    return pred.search(query, n_answers=SEARCH_SIZE)
=======
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
>>>>>>> main


if __name__ == "__main__":
    st.title("Query information retrieval featuring ElasticSearch and "
             "BERT!")
    query_input = st.text_input("What question do you want to ask?",
                                max_chars=100)
    if st.button('Look ! '):
        with st.spinner("Looking zzzzzz"):
            res = search(pred, query_input, batch_size=1)
        for n, doc in enumerate(res):
            st.markdown(f"*{n}*: **{doc.get('full_context')}**")
            st.markdown(f"confidence: {doc.get('confidence')}")
            st.write("\n")
