{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "66448fd5-6d2a-4c96-a921-2a7ea745b853",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3de985ea-3eed-490f-93fa-2160d77bc501",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "df = pd.read_csv('movie_corpora.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8e850b45-5c69-438c-a31c-0d38c790fe05",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract Movie Names and Corpus\n",
    "movie_names = df['Movie Name'].tolist()\n",
    "corpus = df['Corpus'].tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "85cd1d5f-fa10-40f1-a6db-31d245c79284",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize TF-IDF Vectorizer and Matrix\n",
    "tfidf_vectorizer = TfidfVectorizer()\n",
    "tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "78efda58-89ac-4d04-bc76-80afb1565ecd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate Similarity Matrix\n",
    "similarity_matrix = cosine_similarity(tfidf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9aa5b282-365a-4257-925a-5af5a04439a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-10-10 18:21:39.820 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Richard McConkie\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Streamlit App\n",
    "st.title(\"Movie Recommendation System\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "591b4a86-99cc-4e7d-a956-7c29adb56569",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Movie selection dropdown\n",
    "selected_movie = st.selectbox(\"Select a movie:\", movie_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f1d4d85c-e727-4afd-a309-cb9c50f94fcb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the index of the selected movie\n",
    "selected_movie_index = movie_names.index(selected_movie)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "32a91d98-0e90-4305-9409-b84090606456",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get similarity scores for the selected movie\n",
    "similarities_to_selected_movie = list(enumerate(similarity_matrix[selected_movie_index]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d4d21d9f-d6de-4b3e-a18b-2b53cf0ba7c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sort movies by similarity score (descending order)\n",
    "similarities_to_selected_movie = sorted(similarities_to_selected_movie, key=lambda x: x[1], reverse=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "90b19ea7-aebd-483c-9d48-67bec93fde10",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display top similar movies\n",
    "st.subheader(f\"Movies similar to {selected_movie}\")\n",
    "for movie_index, similarity_score in similarities_to_selected_movie[1:6]:  # Display top 5 similar movies\n",
    "    st.write(f\"Movie: {movie_names[movie_index]}, Similarity Score: {similarity_score}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
