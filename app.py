"""This Is Math — an interactive, content-based recommendation demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Item:
    name: str
    creator: str
    description: str
    features: tuple[float, ...]


# Vector positions map to the named qualities below. The compact, fictional
# catalog keeps the data and mathematics visible without needing an external API.
CATALOGS = {

    "Movies": {

        "features": ["Action", "Humor", "Emotion", "Mystery", "Wonder"],

        "items": [

            Item("Interstellar", "Paramount Pictures", "A scientist travels through space to find a new home for humanity.", (3, 1, 5, 4, 5)),

            Item("The Hunger Games", "Lionsgate", "A young woman fights for survival in a dangerous competition.", (5, 1, 4, 3, 3)),

            Item("Hidden Figures", "20th Century Fox", "Three brilliant women help NASA during the space race.", (1, 2, 5, 1, 2)),

            Item("The Martian", "20th Century Fox", "An astronaut must survive alone on Mars and find a way home.", (4, 4, 4, 2, 4)),

            Item("Letters to Juliet", "Summit Entertainment", "A young woman discovers an old love story while traveling in Italy.", (1, 4, 5, 2, 2)),

            Item("Scream", "Dimension Films", "A group of teenagers is targeted by a mysterious masked killer.", (4, 2, 2, 5, 1)),

            Item("10 Things I Hate About You", "Touchstone Pictures", "A high school romance filled with humor, relationships, and misunderstandings.", (1, 5, 4, 1, 1)),

            Item("How to Lose a Guy in 10 Days", "Paramount Pictures", "Two people make a bet that leads to an unexpected romantic relationship.", (1, 5, 4, 1, 1)),

        ],

    },

    "Songs": {

        "features": ["Energy", "Dance", "Warmth", "Dreamy", "Acoustic"],

        "items": [

            Item("Lover, You Should've Come Over", "Jeff Buckley", "An emotional alternative song about love and longing.", (2, 1, 5, 5, 4)),

            Item("JETSKI", "Pedro Sampaio", "An energetic Brazilian dance track with electronic production.", (5, 5, 2, 1, 1)),

            Item("Lovers Rock", "TV Girl", "A nostalgic indie song with dreamy and vintage-inspired sounds.", (2, 2, 4, 5, 2)),

            Item("Beauty and a Beat", "Justin Bieber", "An upbeat pop song with electronic production and a danceable rhythm.", (5, 5, 3, 2, 1)),

            Item("Ela Só Pensa em Beijar", "MC Leozinho", "A lively Brazilian funk song with a playful dance rhythm.", (5, 5, 3, 1, 1)),

            Item("The Morning", "The Weeknd", "A moody R&B song with atmospheric and dreamy production.", (2, 2, 4, 5, 1)),

            Item("Young Hearts Run Free", "Candi Staton", "A soulful disco song with warmth and an energetic groove.", (4, 5, 5, 2, 1)),

            Item("Do You Mind (Crazy Cousinz Remix)", "Kyla", "A dance track with a catchy rhythm and electronic production.", (4, 5, 3, 2, 1)),

            Item("The Spins", "Mac Miller, Empire of the Sun", "A bright hip-hop track with a playful and upbeat feeling.", (4, 4, 3, 2, 2)),

            Item("Last Friday Night", "Katy Perry", "An energetic pop song about a wild night out.", (5, 5, 3, 2, 1)),

            Item("Wonderwall", "Oasis", "A famous alternative rock song with acoustic guitar and emotional lyrics.", (3, 2, 5, 3, 5)),

        ],

    },

}

def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, guarding against a zero-length vector."""
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def recommendations(items: list[Item], ratings: dict[str, int]) -> pd.DataFrame:
    """Score unseen items by their similarity to the user's rated items."""
    columns = ["Item", "By", "Why it may fit", "Match"]
    rated = [(item, rating) for item in items if (rating := ratings.get(item.name, 0))]
    rows = []
    for candidate in items:
        if candidate.name in ratings:
            continue
        scores = []
        for rated_item, rating in rated:
            similarity = cosine_similarity(np.array(candidate.features), np.array(rated_item.features))
            # Three stars is neutral. Farther ratings have greater influence.
            scores.append((similarity, rating - 3))
        weight_total = sum(abs(weight) for _, weight in scores)
        match = sum(similarity * weight for similarity, weight in scores) / weight_total if weight_total else 0.0
        rows.append({"Item": candidate.name, "By": candidate.creator, "Why it may fit": candidate.description, "Match": match})
    # Supplying columns keeps the table valid when every catalog item is rated.
    return pd.DataFrame(rows, columns=columns).sort_values("Match", ascending=False)


st.set_page_config(page_title="This Is Math | Recommendations", page_icon="✨", layout="wide")
st.title("This Is Math ✨")
st.subheader("A tiny recommendation system you can inspect")
st.write("Rate a few things. The app turns their qualities into numbers and uses vector similarity to suggest what to try next.")

with st.sidebar:
    kind = st.radio("Explore", list(CATALOGS), horizontal=True)
    if st.button("Clear my ratings", width="stretch"):
        st.session_state.pop(f"ratings_{kind}", None)
        for item in CATALOGS[kind]["items"]:
            st.session_state.pop(f"{kind}_{item.name}", None)
        st.rerun()

catalog = CATALOGS[kind]
items: list[Item] = catalog["items"]
state_key = f"ratings_{kind}"
ratings = st.session_state.setdefault(state_key, {})

st.markdown("### 1. Rate at least two items")
st.caption("1 = dislike it · 3 = neutral · 5 = love it. Leave items unrated if you have not tried them.")
columns = st.columns(2)
for index, item in enumerate(items):
    with columns[index % 2]:
        rating = st.select_slider(
            f"{item.name} — {item.creator}", options=["Not rated", 1, 2, 3, 4, 5],
            value=ratings.get(item.name, "Not rated"), key=f"{kind}_{item.name}", help=item.description,
        )
        if rating == "Not rated":
            ratings.pop(item.name, None)
        else:
            ratings[item.name] = int(rating)

st.divider()
rated_count = len(ratings)
if rated_count < 2:
    needed = 2 - rated_count
    st.info(f"Rate {needed} more item{'s' if needed != 1 else ''} to get recommendations.")
else:
    result = recommendations(items, ratings)
    st.markdown("### 2. Your recommendations")
    if result.empty:
        st.info("You have rated every item in this catalog. Clear a rating to see a recommendation.")
    else:
        display = result.head(4).copy()
        display["Match"] = display["Match"].map(lambda value: f"{max(0, value) * 100:.0f}%")
        st.dataframe(display, hide_index=True)
        top = result.iloc[0]
        st.success(f"Start with **{top['Item']}** by {top['By']}. It has the strongest calculated match in this small catalog.")

    with st.expander("See the math behind the suggestion"):
        st.write("Every item has a vector of qualities. The feature scales are:")
        st.code("[" + ", ".join(catalog["features"]) + "]")
        st.latex(r"\text{cosine similarity} = \frac{A \cdot B}{\|A\|\|B\|}")
        st.write("The app compares each unrated item with your rated items, then averages those similarities with more influence from ratings far from neutral (3 stars). Similar vectors point in a similar direction, so their cosine score is closer to 1.")
        feature_table = pd.DataFrame([item.features for item in items], index=[item.name for item in items], columns=catalog["features"])
        st.dataframe(feature_table)

st.caption("Educational demo only: ratings stay in this browser session and no personal data is collected.")
