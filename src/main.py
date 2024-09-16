import pandas as pd
import os
import lyricsgenius
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("../Billboard_Hot100_Songs_Spotify_1946-2022.csv")

token = os.getenv('GENIUS_TOKEN')
genius = lyricsgenius.Genius(token)

# Amostra para teste
df_subset = df.head(30)

def get_lyrics(song_title, artist_name):
    try:
        song = genius.search_song(song_title, artist_name)
        return song.lyrics if song else None
    except:
        return None

df_subset['lyrics'] = df_subset.apply(lambda row: get_lyrics(row['Song'], row['Artist Names']), axis=1)

df_subset.to_csv('../Billboard_Hot100_Songs_with_Lyrics_Sample.csv', index=False)
