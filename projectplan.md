I need you to modify my existing Python Streamlit recommendation project called **"This Is Math."** Please read my entire code first and understand how it works before making changes.

### What I currently have

My app allows users to rate movies or songs from a main catalog using a slider from 1 to 5. It uses feature vectors and cosine similarity to recommend items.

Currently, the app recommends unrated items from the same catalog. I want to change this.

### What I want the app to do

I want TWO separate catalogs for each category:

**1. Rating Catalog**
These are the movies or songs that the user rates.

Movies:

* Interstellar
* The Hunger Games
* Hidden Figures
* The Martian
* Letters to Juliet
* Scream
* 10 Things I Hate About You
* How to Lose a Guy in 10 Days

Songs:

* Lover, You Should've Come Over — Jeff Buckley
* JETSKI — Pedro Sampaio
* Lovers Rock — TV Girl
* Beauty and a Beat — Justin Bieber
* Ela Só Pensa em Beijar — MC Leozinho
* The Morning — The Weeknd
* Young Hearts Run Free — Candi Staton
* Do You Mind (Crazy Cousinz Remix) — Kyla
* The Spins — Mac Miller, Empire of the Sun
* Last Friday Night — Katy Perry
* Wonderwall — Oasis

**2. Recommendation Catalog**
These are the items that the user DOES NOT rate. The app should use their feature vectors to recommend them based on the user's preferences.

Extra movies:

* Jumanji
* The Perks of Being a Wallflower
* Mayday
* Uptown Girls
* Truth or Dare
* The Last Sunrise

Extra songs:

* Best Part — Daniel Caesar
* White Ferrari — Frank Ocean
* Sure Thing — Miguel
* Sundress — A$AP Rocky
* Work Out — J. Cole
* Close to Me — Ellie Goulding
* The Cure — Olivia Rodrigo

### Important recommendation rules

1. The user should be able to rate as many or as few items as they want. They should NOT be required to rate all items.
2. The user should need to rate at least 2 items before receiving recommendations.
3. The user can rate ALL items in the main catalog.
4. Even if the user rates every single item in the main catalog, the app MUST still provide recommendations.
5. Recommendations must ONLY come from the separate Recommendation Catalog.
6. The app should NEVER recommend an item from the Rating Catalog.
7. The app should recommend the extra movies or songs based on their similarity to the user's ratings.
8. The user should receive multiple recommendations if possible, such as the top 3 or top 4, rather than only one.
9. Keep the existing cosine similarity mathematics, unless you identify a problem that prevents the recommendation system from working correctly.
10. Keep the app's current Streamlit design and interactive rating sliders as much as possible.

### Mathematical requirements

Each item should have a feature vector matching its category:

Movies:
["Action", "Humor", "Emotion", "Mystery", "Wonder"]

Songs:
["Energy", "Dance", "Warmth", "Dreamy", "Acoustic"]

The recommendation system should:

* Use the user's ratings from the Rating Catalog.
* Compare the feature vectors of rated items with each candidate in the Recommendation Catalog.
* Give more influence to ratings above or below neutral (3).
* Rank the candidates based on the calculated similarity.
* Avoid displaying misleading percentages if the calculated Match score is not a true probability.

### What I need you to do

1. Read my existing code carefully.
2. Modify the code instead of creating an unrelated new project.
3. Add the separate Recommendation Catalogs.
4. Update the recommendation function to accept the rated items and recommendation candidates separately.
5. Ensure that rating all items does not result in an empty recommendation list.
6. Check for bugs, including Streamlit session state issues and empty dataframes.
7. Run and test the app if possible.
8. Explain the changes you made in simple terms.
9. Give me the complete updated code so I can copy and paste it.

Please do not remove the mathematics or turn this into a completely different project. I want to preserve the educational purpose of showing how feature vectors and cosine similarity work.
