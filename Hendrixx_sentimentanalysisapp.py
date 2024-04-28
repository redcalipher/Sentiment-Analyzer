from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
from PIL import Image

st.header('Brighton Sentiment Analyzer')
with st.expander('Analyze Text Log'):
    text = st.text_input('Text your input here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))

        pre = st.text_input('Clean Text Log: ')
        if pre:
            st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                     stopwords=True, lowercase=True, numbers=True, punct=True))


with st.expander('Analyze CSV File'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity


    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_csv(upl)
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(100))
        img = Image.open("C:/Users/Hendrixx/PycharmProjects/Sentiment Analysis Tool/Figure_1.png")
        img.show()
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')





