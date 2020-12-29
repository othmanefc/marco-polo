import streamlit as st
from elasticsearch import Elasticsearch

from src.predict import Predict

SEARCH_SIZE = 10


@st.cache
def search(pred: Predict, query: str):
    return pred.search(query, n_answers=SEARCH_SIZE)


if __name__ == "__main__":
    pred = Predict()
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
