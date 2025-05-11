from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load and process data
df = pd.read_csv('/Users/shrple/Documents/ME/Projects/Music Recommendation System using KNN/Spotify-Dataset.csv')
features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_scaled)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    song_name = ''
    
    if request.method == 'POST':
        song_name = request.form['song']
        try:
            song_index = df[df['track_name'].str.lower() == song_name.lower()].index[0]
            distances, indices = knn.kneighbors([X_scaled[song_index]])
            recommendations = [
                {
                    'track': df.iloc[i]['track_name'],
                    'artist': df.iloc[i]['artist_name']
                }
                for i in indices[0][1:]
            ]
        except IndexError:
            recommendations = [{'track': 'Song not found. Try a different name.', 'artist': ''}]
    
    return render_template('index.html', recommendations=recommendations, song_name=song_name)

if __name__ == '__main__':
    app.run(debug=True)
