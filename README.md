# This Is Math

An interactive Streamlit project that demonstrates a tiny content-based recommendation system. Users rate movies or songs, then see recommendations calculated with cosine similarity between item-feature vectors.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push this repository to GitHub, create a new app in Streamlit Community Cloud, and choose `app.py` as the entry point. The dependencies in `requirements.txt` are installed automatically.

## How it works

- Choose movies or songs.
- Rate at least two catalog items from 1 (dislike) to 5 (love).
- Each item has a short numeric feature vector, such as action, humor, and mystery for movies.
- The app computes cosine similarity between vectors and weights those similarities by how strongly each item was rated relative to neutral (3 stars).

This is an educational, in-memory demo: it has no accounts, database, API keys, or persistent user data.
