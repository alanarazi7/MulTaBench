import streamlit as st

from multabench.leaderboard.large_results import display_large_results
from multabench.leaderboard.paper_results import display_paper_benchmark


def display_leaderboard():
    st.title("MulTaBench Leaderboard 🌟")
    multabench, large = st.tabs(["🏆 MulTaBench", "🦣 Large"])
    with multabench:
        display_paper_benchmark()
    with large:
        display_large_results()


if __name__ == "__main__":
    display_leaderboard()
