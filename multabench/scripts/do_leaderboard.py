import streamlit as st

from multabench.leaderboard.large_results import display_large_results
from multabench.leaderboard.paper_results import display_paper_benchmark
from multabench.leaderboard.text_results import display_pool_performance
from multabench.leaderboard.text_pool_tar_results import display_text_pool_tar
from multabench.leaderboard.tfidf_results import display_tfidf_benchmark
from multabench.leaderboard.triple_results import display_triple


def display_leaderboard():
    st.title("MulTaBench Leaderboard 🌟")
    tabs = ["🏆 MulTaBench", "🦣 Large", "🏊 Pool", "📈 Text Pool", "📊 TF-IDF", "😍 Triple"]
    multabench, large, pool, text_pool, tfidf, triple = st.tabs(tabs)
    with multabench:
        display_paper_benchmark()
    with large:
        display_large_results()
    with pool:
        display_pool_performance()
    with text_pool:
        display_text_pool_tar()
    with tfidf:
        display_tfidf_benchmark()
    with triple:
        display_triple()


if __name__ == "__main__":
    display_leaderboard()
