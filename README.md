# Music-Recommendation-System



This project implements a music recommendation system using TF-IDF and cosine similarity. It analyzes song metadata to find similarities between songs and recommends music based on the user’s selected song. The system provides personalized recommendations without using user ratings. It uses TF-IDF to convert song information into numerical features and cosine similarity to find and recommend songs that are most similar to the user’s selected song. This beginner-friendly project helps understand the basics of content-based recommendation systems.


# Steps:

1. Set Up the Environment and install necessary packages
- Install necessary Python packages (FastAPI, Uvicorn, scikit-learn, pandas, etc).
- Load the music dataset containing song details (title, artist, genre, etc.).
2. Data Preprocessing
- Clean and combine relevant song features into a single text format.
- Apply TF-IDF Vectorizer to convert text data into feature vectors.
3. Implement Recommendation Logic
- Use cosine similarity to calculate similarity between songs.
- When a user selects a song, find and return the most similar songs.
4. Build the Recommendation Function
- Create a function that takes a song name as input.
- Output a list of recommended songs based on similarity scores.
5. Test, Document, and Finalize
- Test the system with different song inputs.
- Document the working of TF-IDF, cosine similarity, and recommendation results
