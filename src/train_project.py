import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA=os.path.join(ROOT,'data','spotify_tracks.csv')
OUT=os.path.join(ROOT,'outputs')
PLOT=os.path.join(OUT,'plots')
MODEL=os.path.join(ROOT,'models')
os.makedirs(PLOT,exist_ok=True); os.makedirs(MODEL,exist_ok=True)

sns.set_theme(style='whitegrid')

df=pd.read_csv(DATA)
initial_shape=df.shape

# ---------------- Data preprocessing ----------------
# Remove exact duplicates and rows missing essential song metadata.
df=df.drop_duplicates().copy()
for c in ['track_name','track_artist','track_album_name']:
    df[c]=df[c].fillna('Unknown')
# Coerce numeric columns and repair invalid values.
audio_features=['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','track_popularity']
for c in audio_features:
    df[c]=pd.to_numeric(df[c], errors='coerce')
df=df.dropna(subset=audio_features+['playlist_genre','playlist_subgenre','playlist_name']).copy()
df['duration_min']=df['duration_ms']/60000
# Clip physically bounded Spotify-style features to remove impossible outliers.
for c in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']:
    df[c]=df[c].clip(0,1)
df['tempo']=df['tempo'].clip(lower=1)
df['track_popularity']=df['track_popularity'].clip(0,100)

# Save cleaned dataset
clean_path=os.path.join(OUT,'cleaned_spotify_tracks.csv'); df.to_csv(clean_path,index=False)

# ---------------- EDA ----------------
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(PLOT,name),dpi=160,bbox_inches='tight'); plt.close()

# 1 Popularity distribution
plt.figure(figsize=(10,5)); sns.histplot(df['track_popularity'],bins=30,kde=True,color='steelblue'); plt.title('Track Popularity Distribution'); plt.xlabel('Popularity'); plt.ylabel('Number of Tracks'); savefig('01_popularity_distribution.png')
# 2 Top genres by count
plt.figure(figsize=(10,6)); df['playlist_genre'].value_counts().head(10).sort_values().plot(kind='barh'); plt.title('Top 10 Playlist Genres by Track Count'); plt.xlabel('Tracks'); savefig('02_top_genres_count.png')
# 3 Top subgenres
plt.figure(figsize=(10,7)); df['playlist_subgenre'].value_counts().head(15).sort_values().plot(kind='barh'); plt.title('Top 15 Playlist Subgenres by Track Count'); plt.xlabel('Tracks'); savefig('03_top_subgenres_count.png')
# 4 Audio feature distributions
features=['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']
fig,axes=plt.subplots(4,2,figsize=(12,14)); axes=axes.ravel()
for ax,c in zip(axes,features): sns.histplot(df[c],kde=True,ax=ax,color='steelblue'); ax.set_title(c.title())
axes[-1].axis('off'); plt.tight_layout(); plt.savefig(os.path.join(PLOT,'04_audio_feature_distributions.png'),dpi=160,bbox_inches='tight'); plt.close()
# 5 Correlation matrix
corr_cols=['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_min']
plt.figure(figsize=(12,9)); sns.heatmap(df[corr_cols].corr(),cmap='coolwarm',center=0,square=False,vmin=-1,vmax=1); plt.title('Correlation Matrix of Numerical Features'); savefig('05_correlation_matrix.png')
# 6 energy danceability
plt.figure(figsize=(9,6)); sample=df.sample(min(6000,len(df)),random_state=42); sns.scatterplot(data=sample,x='danceability',y='energy',hue='playlist_genre',alpha=.45,s=22,legend=False,palette='tab10'); plt.title('Energy vs Danceability'); savefig('06_energy_vs_danceability.png')
# 7 popularity by genre
pg=df.groupby('playlist_genre')['track_popularity'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6)); pg.plot(kind='bar'); plt.title('Average Track Popularity by Playlist Genre'); plt.ylabel('Average Popularity'); plt.xticks(rotation=45,ha='right'); savefig('07_avg_popularity_genre.png')
# 8 feature means by genre heatmap
means=df.groupby('playlist_genre')[['danceability','energy','acousticness','instrumentalness','valence','speechiness']].mean()
plt.figure(figsize=(11,7)); sns.heatmap(means,cmap='viridis',vmin=0,vmax=1); plt.title('Audio Feature Profile by Playlist Genre'); savefig('08_genre_feature_heatmap.png')
# 9 duration
plt.figure(figsize=(10,5)); sns.histplot(df['duration_min'],bins=40,kde=True,color='steelblue'); plt.xlim(0,15); plt.title('Track Duration Distribution'); plt.xlabel('Duration (minutes)'); savefig('09_duration_distribution.png')
# 10 Popularity spread by genre
plt.figure(figsize=(11,6)); sns.boxplot(data=df,x='playlist_genre',y='track_popularity'); plt.title('Popularity Distribution by Playlist Genre'); plt.xlabel('Playlist Genre'); plt.ylabel('Popularity'); plt.xticks(rotation=35,ha='right'); savefig('10_popularity_by_genre_boxplot.png')

# ---------------- Clustering / genre segmentation ----------------
cluster_features=['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_min','track_popularity']
X=df[cluster_features].copy()
scaler=StandardScaler(); Xs=scaler.fit_transform(X)
# evaluate k values using a representative sample for fast, reproducible validation
ks=range(2,11); eval_rows=[]
rng=np.random.RandomState(42); eval_idx=rng.choice(len(Xs),size=min(5000,len(Xs)),replace=False); X_eval=Xs[eval_idx]
for k in ks:
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels_eval=km.fit_predict(X_eval)
    eval_rows.append([k,silhouette_score(X_eval,labels_eval),calinski_harabasz_score(X_eval,labels_eval),davies_bouldin_score(X_eval,labels_eval),km.inertia_])
eval_df=pd.DataFrame(eval_rows,columns=['k','silhouette','calinski_harabasz','davies_bouldin','inertia'])
eval_df.to_csv(os.path.join(OUT,'cluster_evaluation.csv'),index=False)
# Model-selection note: silhouette favors a very broad 2-segment solution.
# For the requested genre-segmentation project, we use 6 final segments because the dataset
# contains six playlist genres and this gives a more useful, interpretable recommendation space.
optimal_k_by_silhouette=int(eval_df.loc[eval_df['silhouette'].idxmax(),'k'])
best_k=6
# Fit final model
kmeans=KMeans(n_clusters=best_k,random_state=42,n_init=20)
df['cluster']=kmeans.fit_predict(Xs)
# cluster profiles
profiles=df.groupby('cluster')[cluster_features].mean().round(3)
profiles['track_count']=df['cluster'].value_counts().sort_index(); profiles.to_csv(os.path.join(OUT,'cluster_profiles.csv'))
# dominant playlist genre/subgenre per cluster
for col in ['playlist_genre','playlist_subgenre']:
    dom=df.groupby('cluster')[col].agg(lambda x: x.value_counts().index[0])
    profiles[col+'_dominant']=dom
profiles.to_csv(os.path.join(OUT,'cluster_profiles.csv'))
# label clusters with interpretable names from dominant genre + feature profile
cluster_names={}
for cl,row in profiles.iterrows():
    g=row['playlist_genre_dominant']; e=row['energy']; d=row['danceability']; a=row['acousticness']; v=row['valence']; i=row['instrumentalness']
    if e>.72 and d>.62: vibe='High-Energy Dance'
    elif a>.55: vibe='Acoustic/Chill'
    elif i>.25: vibe='Instrumental'
    elif v<.38: vibe='Low-Valence/Moody'
    elif d>.65: vibe='Danceable'
    else: vibe='Balanced'
    cluster_names[cl]=f'Segment {cl} – {vibe} / {g.title()}-dominant'
df['segment_name']=df['cluster'].map(cluster_names)
# evaluation plots
plt.figure(figsize=(9,5)); plt.plot(eval_df['k'],eval_df['silhouette'],marker='o'); plt.axvline(best_k,ls='--'); plt.title('Silhouette Score by Number of Clusters'); plt.xlabel('K'); plt.ylabel('Silhouette Score'); savefig('11_silhouette_scores.png')
# PCA visualization
pca=PCA(n_components=2,random_state=42); Z=pca.fit_transform(Xs)
plot_df=pd.DataFrame({'PC1':Z[:,0],'PC2':Z[:,1],'cluster':df['cluster'].astype(str)})
ps=plot_df.sample(min(8000,len(plot_df)),random_state=42)
plt.figure(figsize=(10,7)); sns.scatterplot(data=ps,x='PC1',y='PC2',hue='cluster',palette='tab10',s=25,alpha=.55); plt.title(f'PCA Visualization of {best_k} Music Segments'); plt.legend(title='Cluster'); savefig('12_pca_clusters.png')
# cluster sizes
plt.figure(figsize=(9,5)); df['segment_name'].value_counts().sort_values().plot(kind='barh'); plt.title('Music Segment Sizes'); plt.xlabel('Tracks'); savefig('13_cluster_sizes.png')

# ---------------- Genre prediction baseline ----------------
# A supervised model complements clustering: given audio features, predict the dataset's
# playlist genre. This can be used as a genre-aware layer in a recommendation system.
X_cls=df[cluster_features].copy(); y_cls=df['playlist_genre']
Xtr,Xte,ytr,yte=train_test_split(X_cls,y_cls,test_size=0.20,random_state=42,stratify=y_cls)
rf=RandomForestClassifier(n_estimators=160,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1,class_weight='balanced_subsample')
rf.fit(Xtr,ytr); pred=rf.predict(Xte)
cls_acc=accuracy_score(yte,pred)
with open(os.path.join(OUT,'classification_report.txt'),'w') as f: f.write(classification_report(yte,pred))
cm=confusion_matrix(yte,pred,labels=sorted(y_cls.unique()))
plt.figure(figsize=(9,7)); sns.heatmap(cm,annot=True,fmt='d',xticklabels=sorted(y_cls.unique()),yticklabels=sorted(y_cls.unique()),cmap='Blues'); plt.title('Random Forest Genre Classification Confusion Matrix'); plt.xlabel('Predicted Genre'); plt.ylabel('Actual Genre'); savefig('14_genre_classification_confusion_matrix.png')
fi=pd.Series(rf.feature_importances_,index=cluster_features).sort_values(ascending=True)
plt.figure(figsize=(9,6)); fi.plot(kind='barh'); plt.title('Random Forest Feature Importance for Genre Prediction'); plt.xlabel('Importance'); savefig('15_feature_importance.png')
joblib.dump(rf,os.path.join(MODEL,'genre_classifier.joblib'))

# ---------------- Recommendation engine ----------------
# Nearest-neighbor recommender within the learned cluster space.
nn=NearestNeighbors(n_neighbors=11,metric='euclidean').fit(Xs)
# Save artifacts
joblib.dump(scaler,os.path.join(MODEL,'scaler.joblib'))
joblib.dump(kmeans,os.path.join(MODEL,'kmeans.joblib'))
joblib.dump(nn,os.path.join(MODEL,'nearest_neighbors.joblib'))
joblib.dump(cluster_features,os.path.join(MODEL,'cluster_features.joblib'))
# Store only necessary recommendation columns to keep artifact compact
rec_cols=['track_id','track_name','track_artist','track_album_name','playlist_genre','playlist_subgenre','playlist_name','track_popularity']+cluster_features+['cluster','segment_name']
df[rec_cols].to_pickle(os.path.join(MODEL,'tracks_model_data.pkl'))

# sample recommendation for a recognizable track: first row
sample_idx=0
_, inds=nn.kneighbors(Xs[sample_idx:sample_idx+1],n_neighbors=min(11,len(df)))
recs=df.iloc[inds[0][1:]][['track_name','track_artist','playlist_genre','playlist_subgenre','track_popularity','segment_name']]
recs.to_csv(os.path.join(OUT,'sample_recommendations.csv'),index=False)

# summary
summary={
 'input_rows':int(initial_shape[0]),'input_columns':int(initial_shape[1]),
 'clean_rows':int(len(df)),'duplicates_removed':int(initial_shape[0]-len(df)),
 'missing_values_after_cleaning':int(df.isna().sum().sum()),
 'genres':int(df['playlist_genre'].nunique()),'subgenres':int(df['playlist_subgenre'].nunique()),
 'artists':int(df['track_artist'].nunique()),'best_k':best_k,
 'optimal_k_by_silhouette':optimal_k_by_silhouette,
 'silhouette_score_final_k':float(eval_df.loc[eval_df.k==best_k,'silhouette'].iloc[0]),
 'genre_classifier_accuracy':float(cls_acc),
 'calinski_harabasz':float(eval_df.loc[eval_df.k==best_k,'calinski_harabasz'].iloc[0]),
 'davies_bouldin':float(eval_df.loc[eval_df.k==best_k,'davies_bouldin'].iloc[0]),
 'cluster_names':{str(k):v for k,v in cluster_names.items()}
}
with open(os.path.join(OUT,'project_summary.json'),'w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print('\nTop sample recommendations:\n',recs.to_string(index=False))
