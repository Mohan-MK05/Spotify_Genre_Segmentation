import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Spotify Genre Segmentation',page_icon='🎵',layout='wide')
ROOT='.'
tracks=pd.read_pickle('models/tracks_model_data.pkl')
scaler=joblib.load('models/scaler.joblib')
kmeans=joblib.load('models/kmeans.joblib')
nn=joblib.load('models/nearest_neighbors.joblib')
features=joblib.load('models/cluster_features.joblib')
classifier=joblib.load('models/genre_classifier.joblib')

st.title('🎵 Spotify Songs Genre Segmentation & Recommendation')
st.caption('Unsupervised audio-feature segmentation + genre prediction + nearest-neighbor recommendations')

with st.sidebar:
    st.header('Song Features')
    danceability=st.slider('Danceability',0.0,1.0,0.65,0.01)
    energy=st.slider('Energy',0.0,1.0,0.70,0.01)
    loudness=st.slider('Loudness (dB)',-30.0,5.0,-7.0,0.5)
    speechiness=st.slider('Speechiness',0.0,1.0,0.10,0.01)
    acousticness=st.slider('Acousticness',0.0,1.0,0.20,0.01)
    instrumentalness=st.slider('Instrumentalness',0.0,1.0,0.05,0.01)
    liveness=st.slider('Liveness',0.0,1.0,0.15,0.01)
    valence=st.slider('Valence',0.0,1.0,0.50,0.01)
    tempo=st.slider('Tempo (BPM)',40.0,220.0,120.0,1.0)
    duration=st.slider('Duration (minutes)',0.5,10.0,3.5,0.1)
    popularity=st.slider('Popularity',0,100,50,1)

if st.button('Recommend Songs',type='primary'):
    x=np.array([[danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration,popularity]])
    xs=scaler.transform(x)
    cluster=int(kmeans.predict(xs)[0])
    genre=classifier.predict(x)[0]
    dists, inds=nn.kneighbors(xs,n_neighbors=15)
    result=tracks.iloc[inds[0]].copy()
    result=result[result['cluster']==cluster].head(10)
    if len(result)<5:
        result=tracks.iloc[inds[0]].head(10)
    st.subheader('Predicted playlist genre')
    st.metric('Genre',genre)
    st.metric('Music segment',tracks.loc[tracks.cluster==cluster,'segment_name'].iloc[0])
    st.subheader('Recommended tracks')
    st.dataframe(result[['track_name','track_artist','playlist_genre','playlist_subgenre','track_popularity','segment_name']],use_container_width=True,hide_index=True)

st.divider()
st.subheader('Explore learned segments')
summary=tracks.groupby(['cluster','segment_name']).agg(tracks=('track_id','count'),avg_popularity=('track_popularity','mean'),avg_energy=('energy','mean'),avg_danceability=('danceability','mean')).reset_index()
st.dataframe(summary.round(3),use_container_width=True,hide_index=True)
