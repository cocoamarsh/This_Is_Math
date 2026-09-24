"""This Is Math — an interactive, content-based recommendation demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Item:
    """One catalog item and its five-dimensional feature vector."""

    name: str
    creator: str
    description: str
    features: tuple[float, ...]


# The rating catalog contains only items a user can rate. Recommendation items
# are separate candidates, so a rating-catalog item can never be recommended.
CATALOGS = {
    "Movies": {
        "features": ["Action", "Humor", "Emotion", "Mystery", "Wonder"],
        "rating_items": [
            Item("Interstellar", "Paramount Pictures", "A scientist travels through space to find a new home for humanity.", (3, 1, 5, 4, 5)),
            Item("The Hunger Games", "Lionsgate", "A young woman fights for survival in a dangerous competition.", (5, 1, 4, 3, 3)),
            Item("Hidden Figures", "20th Century Fox", "Three brilliant women help NASA during the space race.", (1, 2, 5, 1, 2)),
            Item("The Martian", "20th Century Fox", "An astronaut must survive alone on Mars and find a way home.", (4, 4, 4, 2, 4)),
            Item("Letters to Juliet", "Summit Entertainment", "A young woman discovers an old love story while traveling in Italy.", (1, 4, 5, 2, 2)),
            Item("Scream", "Dimension Films", "Teenagers are targeted by a mysterious masked killer.", (4, 2, 2, 5, 1)),
            Item("10 Things I Hate About You", "Touchstone Pictures", "A high-school romance with humor and misunderstandings.", (1, 5, 4, 1, 1)),
            Item("How to Lose a Guy in 10 Days", "Paramount Pictures", "A bet leads to an unexpected romantic relationship.", (1, 5, 4, 1, 1)),
        ],
        "recommendation_items": [
            Item("Jumanji", "TriStar Pictures", "A high-energy fantasy adventure with playful humor.", (4, 5, 3, 3, 5)),
            Item("The Perks of Being a Wallflower", "Summit Entertainment", "An emotional coming-of-age story about friendship and belonging.", (1, 3, 5, 2, 2)),
            Item("Mayday", "Mystery Pictures", "A tense adventure with emotional stakes and unanswered questions.", (3, 2, 4, 4, 3)),
            Item("Uptown Girls", "Metro-Goldwyn-Mayer", "A warm, funny story about an unlikely friendship.", (1, 4, 5, 1, 1)),
            Item("Truth or Dare", "Blumhouse", "A suspenseful horror mystery with dangerous consequences.", (4, 2, 2, 5, 1)),
            Item("The Last Sunrise", "Aurora Films", "A reflective science-fiction mystery with a sense of wonder.", (3, 1, 4, 4, 5)),
        ],
    },
    "Songs": {
        "features": ["Energy", "Dance", "Warmth", "Dreamy", "Acoustic"],
        "rating_items": [
            Item("Lover, You Should've Come Over", "Jeff Buckley", "An emotional alternative song about love and longing.", (2, 1, 5, 5, 4)),
            Item("JETSKI", "Pedro Sampaio", "An energetic Brazilian dance track with electronic production.", (5, 5, 2, 1, 1)),
            Item("Lovers Rock", "TV Girl", "A nostalgic indie song with dreamy, vintage-inspired sounds.", (2, 2, 4, 5, 2)),
            Item("Beauty and a Beat", "Justin Bieber", "An upbeat pop song with electronic production and a danceable rhythm.", (5, 5, 3, 2, 1)),
            Item("Ela Só Pensa em Beijar", "MC Leozinho", "A lively Brazilian funk song with a playful dance rhythm.", (5, 5, 3, 1, 1)),
            Item("The Morning", "The Weeknd", "A moody R&B song with atmospheric, dreamy production.", (2, 2, 4, 5, 1)),
            Item("Young Hearts Run Free", "Candi Staton", "A soulful disco song with warmth and an energetic groove.", (4, 5, 5, 2, 1)),
            Item("Do You Mind (Crazy Cousinz Remix)", "Kyla", "A dance track with a catchy electronic rhythm.", (4, 5, 3, 2, 1)),
            Item("The Spins", "Mac Miller, Empire of the Sun", "A bright hip-hop track with a playful, upbeat feeling.", (4, 4, 3, 2, 2)),
            Item("Last Friday Night", "Katy Perry", "An energetic pop song about a wild night out.", (5, 5, 3, 2, 1)),
            Item("Wonderwall", "Oasis", "Alternative rock with acoustic guitar and emotional lyrics.", (3, 2, 5, 3, 5)),
        ],
        "recommendation_items": [
            Item("Best Part", "Daniel Caesar", "A gentle, warm R&B love song.", (1, 1, 5, 4, 3)),
            Item("White Ferrari", "Frank Ocean", "A reflective, dreamy song with an intimate mood.", (1, 1, 5, 5, 2)),
            Item("Sure Thing", "Miguel", "Smooth, romantic R&B with a relaxed groove.", (2, 3, 5, 4, 1)),
            Item("Sundress", "A$AP Rocky", "A hazy hip-hop track with dreamy production.", (3, 3, 3, 5, 1)),
            Item("Work Out", "J. Cole", "An energetic hip-hop song with a rhythmic beat.", (4, 4, 3, 2, 1)),
            Item("Close to Me", "Ellie Goulding", "Electronic pop with a danceable, atmospheric sound.", (4, 4, 3, 4, 1)),
            Item("The Cure", "Olivia Rodrigo", "A warm pop song with an emotional, melodic feel.", (3, 3, 5, 3, 2)),
        ],
    },
}


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, guarding against a zero-length vector."""
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def recommendations(
    rated_items: list[Item], candidates: list[Item], ratings: dict[str, int]
) -> pd.DataFrame:
    """Rank recommendation candidates using weighted cosine similarity.

    Ratings are centered at neutral (3). A 5-star rating pulls similar items
    upward, while a 1-star rating pushes similar items downward. The resulting
    score is a preference score, not a probability.
    """
    columns = ["Item", "By", "Why it may fit", "Recommendation score"]
    rated = [(item, ratings[item.name]) for item in rated_items if item.name in ratings]
    rows = []

    for candidate in candidates:
        weighted_scores = []
        for rated_item, rating in rated:
            similarity = cosine_similarity(
                np.array(candidate.features, dtype=float),
                np.array(rated_item.features, dtype=float),
            )
            weighted_scores.append((similarity, rating - 3))

        total_weight = sum(abs(weight) for _, weight in weighted_scores)
        score = (
            sum(similarity * weight for similarity, weight in weighted_scores) / total_weight
            if total_weight
            else 0.0
        )
        rows.append(
            {
                "Item": candidate.name,
                "By": candidate.creator,
                "Why it may fit": candidate.description,
                "Recommendation score": score,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        "Recommendation score", ascending=False
    )


def clear_ratings(kind: str) -> None:
    """Clear both the saved ratings and their corresponding slider state."""
    st.session_state.pop(f"ratings_{kind}", None)
    for item in CATALOGS[kind]["rating_items"]:
        st.session_state.pop(f"rating_{kind}_{item.name}", None)


st.set_page_config(page_title="This Is Math | Recommendations", page_icon="✨", layout="wide")
st.title("This Is Math ✨")
st.subheader("A tiny recommendation system you can inspect")
st.write(
    "Rate a few items from the rating catalog. The app turns their qualities into "
    "numbers and uses vector similarity to suggest new items from a separate catalog."
)

with st.sidebar:
    kind = st.segmented_control("Explore", list(CATALOGS), default="Movies")
    if st.button("Clear my ratings", width="stretch", on_click=clear_ratings, args=(kind,)):
        pass

catalog = CATALOGS[kind]
rated_items: list[Item] = catalog["rating_items"]
candidates: list[Item] = catalog["recommendation_items"]
state_key = f"ratings_{kind}"
ratings: dict[str, int] = st.session_state.setdefault(state_key, {})

st.markdown("### 1. Rate at least two items")
st.caption("1 = dislike it · 3 = neutral · 5 = love it. Leave items unrated if you have not tried them.")
rating_columns = st.columns(2)
for index, item in enumerate(rated_items):
    with rating_columns[index % 2]:
        rating = st.select_slider(
            f"{item.name} — {item.creator}",
            options=["Not rated", 1, 2, 3, 4, 5],
            value=ratings.get(item.name, "Not rated"),
            key=f"rating_{kind}_{item.name}",
            help=item.description,
        )
        if rating == "Not rated":
            ratings.pop(item.name, None)
        else:
            ratings[item.name] = int(rating)

st.divider()
if len(ratings) < 2:
    needed = 2 - len(ratings)
    st.info(f"Rate {needed} more item{'s' if needed != 1 else ''} to get recommendations.")
else:
    result = recommendations(rated_items, candidates, ratings)
    st.markdown("### 2. Your recommendations")

    if result.empty:
        st.info("There are no recommendation candidates available for this category yet.")
    else:
        display = result.head(4).copy()
        display["Recommendation score"] = display["Recommendation score"].map(
            lambda score: f"{score:+.3f}"
        )
        st.dataframe(display, hide_index=True)
        top = result.iloc[0]
        st.success(
            f"Start with **{top['Item']}** by {top['By']}. "
            "It has the strongest calculated preference score."
        )

    with st.expander("See the math behind the suggestions"):
        st.write("Every item has a vector of qualities. The feature scales are:")
        st.code("[" + ", ".join(catalog["features"]) + "]")
        st.latex(r"\text{cosine similarity} = \frac{A \cdot B}{\|A\|\|B\|}")
        st.write(
            "The app compares every recommendation-catalog item with each item you rated. "
            "Ratings above neutral (3) reward similar items; ratings below neutral penalize them. "
            "A recommendation score is a relative ranking value, not a percentage or probability."
        )
        feature_table = pd.DataFrame(
            [item.features for item in rated_items + candidates],
            index=[f"Rating: {item.name}" for item in rated_items]
            + [f"Recommend: {item.name}" for item in candidates],
            columns=catalog["features"],
        )
        st.dataframe(feature_table)

st.caption("Educational demo only: ratings stay in this browser session and no personal data is collected.")
