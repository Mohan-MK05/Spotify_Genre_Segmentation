# Spotify Songs' Genre Segmentation & Recommendation System

## Project objective
Build an automated music-analysis pipeline that preprocesses Spotify playlist data, performs exploratory data analysis, discovers audio-feature-based song segments, predicts playlist genre, and produces similarity-based song recommendations.

## Dataset
The supplied `data/spotify_tracks.csv` is the working dataset. It contains 32,833 rows and 23 columns, including playlist genre/subgenre/name and audio features such as danceability, energy, loudness, acousticness, instrumentalness, valence and tempo.

The mentor-provided Kaggle reference is:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/code

## Main requirements covered
- Data preprocessing and cleaning
- Multiple EDA visualizations
- Correlation matrix
- Genre/subgenre/playlist analysis
- K-Means clustering and cluster evaluation
- PCA visualization of segments
- Random Forest genre prediction baseline
- Nearest-neighbor recommendation engine
- Streamlit demo application

## Project structure
```
Spotify_Genre_Segmentation_Project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── spotify_tracks.csv
├── src/
│   └── train_project.py
├── models/
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   ├── nearest_neighbors.joblib
│   ├── genre_classifier.joblib
│   ├── cluster_features.joblib
│   └── tracks_model_data.pkl
├── outputs/
│   ├── cleaned_spotify_tracks.csv
│   ├── cluster_evaluation.csv
│   ├── cluster_profiles.csv
│   ├── classification_report.txt
│   ├── sample_recommendations.csv
│   ├── project_summary.json
│   └── plots/
├── notebooks/
│   └── Spotify_Genre_Segmentation.ipynb
└── docs/
    ├── Major_Project_Report.docx
    └── Major_Project_Presentation.pptx
```

## How to run
1. Create a Python environment.
2. Install packages:
   `pip install -r requirements.txt`
3. Train/recreate all outputs:
   `python src/train_project.py`
4. Launch the recommendation demo:
   `streamlit run app.py`

## Modeling note
Silhouette evaluation on a representative sample selected K=2 as the strongest purely unsupervised separation. Because the assignment specifically asks for genre segmentation and the dataset has six playlist genres, the final recommendation segmentation uses K=6 for practical interpretability. The report records this choice transparently.

## Results from the supplied CSV
- Input rows: 32,833
- Input columns: 23
- Clean rows: 32,833
- Playlist genres: 6
- Playlist subgenres: 24
- Unique artists: 10,693
- Final segmentation clusters: 6
- Final K=6 silhouette score: 0.1252
- Random Forest genre classification accuracy: 55.98%

The project is intended as an academic ML/data-science implementation, not a replica of Spotify's production recommendation system.
