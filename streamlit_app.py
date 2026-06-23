import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk import pos_tag
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('wordnet',quiet=True)
nltk.download('average_perceptron_tagger_eng',quiet=True)

#Loading models
@st.cache_resource
def load_models():
    model=joblib.load("models/linear_svc_model.pkl")
    vectorizer= joblib.load("models/tfidf_vectorizer.pkl")
    return model,vectorizer

model,vectorizer = load_models()

def load_data():
    return pd.read_csv("sentiment_tags.csv")

df = load_data()

#Titles
st.title("Airbnb Sentiment Analysis")
st.markdown("Reviews sentiment classifier")

lemmatizer = nltk.stem.WordNetLemmatizer()

tag_dict = {
    'J': wordnet.ADJ,
    'N': wordnet.NOUN,
    'V': wordnet.VERB,
    'R': wordnet.ADV
}

#cleaning functions
def get_word_net(tag):
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def handle_negations(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'\bdidnt\b', 'did not', text)
    text = re.sub(r'\bdont\b', 'do not', text)
    text = re.sub(r'\bisnt\b', 'is not', text)
    text = re.sub(r'\bcant\b', 'cannot', text)
    text = re.sub(r'\bnot\s+(\w+)\s+(\w+)', r'not_\1_\2', text)
    text = re.sub(r'\bnot\s+(\w+)', r'not_\1', text)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = handle_negations(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    tagged = pos_tag(words)
    text = ' '.join([lemmatizer.lemmatize(w, get_word_net(t)) for w, t in tagged])
    return text

#Tabs
tab1,tab2,tab3 = st.tabs(["EDA Explorer","Model Comparison","Live Predictor"])

#EDA explorer tab
with tab1:
    st.header("EDA Explorer")
    st.info("""The visualizations below represent the original Airbnb reviews distribution.
SMOTE was applied only on the training data to improve model performance on minority classes.
    """)
    
    col1,col2,col3,col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df))
        
    with col2:
        positive_pct = (df['sentiment']=='positive').mean()*100
        st.metric("Positive comments",f"{positive_pct:.1f}%")

    with col3:
        negative_pct = (df['sentiment']=='negative').mean()*100
        st.metric("Negative comments",f"{negative_pct:.1f}%")

    with col4:
        st.metric("Avg Words",f"{df['word_count'].mean():.0f}")
    
    #Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(data=df,
                  x='sentiment',
                  order=['positive','neutral','negative'],
                  hue='sentiment',
                  palette=['#2ecc71', '#f39c12', '#e74c3c'],
                  legend=False,
                  ax=ax
                 )
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    #Review lengths
    st.subheader("Review length by sentiment")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df,
                x='sentiment',
                y='word_count',
                order=['positive','neutral','negative'],
                hue='sentiment',
                palette=['#2ecc71', '#f39c12', '#e74c3c'],
                legend=False,
                ax=ax2
               )
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Word Count')
    st.pyplot(fig2)

    #WordCloud
    st.subheader("Most Frecuent Words by Sentiment")
    col1,col2,col3 = st.columns(3)
    for col, sentiment, color in zip([col1,col2,col3],
                                     ['positive','neutral','negative'],
                                     ['Greens','Oranges','Reds']):
        text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_comments'].dropna())
        wc = WordCloud(width=400,
                       height=300,
                       background_color='white',
                       colormap=color
                      ).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(5,3))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(sentiment.capitalize())
        with col:
            st.pyplot(fig_wc)

    #Interactive explorer
    st.subheader("Interactive Review Explorer")

    col1,col2 = st.columns(2)

    with col1:
        sentiment_filter = st.selectbox(
            "Select sentiment",
            ['positive','neutral','negative']
        )

    with col2:
        min_len = st.slider("Minimum word count",0,200,10)

    #Filtered
    filtered_df = df[
        (df['sentiment'] == sentiment_filter) &
        (df['word_count'] >= min_len)
    ]

    st.write(f"Showing {len(filtered_df)} reviews")

    #Mostrar muestras
    if len(filtered_df)>0:
        for review in filtered_df['cleaned_comments'].sample(min(3,len(filtered_df))):
            st.write("•", review)
    else:
        st.warning("No reviews match the selected filter")
            
#Model Comparison
with tab2:
    st.header("Model Comparison")
    st.markdown("Performance comparison of all models tested in this project.")

    data = {
        'Model':['Linear SVC','Logisitic Regression','Random Forest','XGBoost','LSTM','DistilBERT'],
        'Accuracy':['91%','90%','89%','90%','87%','N/A'],
        'Negative Precision':['90%', '89%', '93%', '92%', '65%', 'N/A'],
        'Neutral Precision':['91%', '89%', '91%', '87%', '87%', 'N/A'],
        'Positive Precision':['93%', '92%', '85%', '91%', '92%', 'N/A'],
        'Notes':[
            'Selected as final model',
            'Good baseline',
            'Lower positive precision',
            'Requires label encoding',
            'Struggles with negative',
            'No neutral class'
        ]
    }

    st.dataframe(pd.DataFrame(data), use_container_width=True)
    st.info("**Final Model: Linear SVC** -- Best overall accuracy and strongest performance on negative detection (90% precision, 97% recall).")

    
    
#Live Predictor
with tab3:
    st.header("Live Predictor")
    st.markdown("Write a review and see the predicted sentiment")

    user_input = st.text_area("Enter a review:", height=150,
    placeholder='e.g. Amazing place, very clean and great location!')

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Prease enter a review first.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            #Confidence Score
            confidence_score = abs(model.decision_function(vectorized)[0][0])
            prob = 1/(1+np.exp(-confidence_score))
            st.write(f"Predicted Probability: {prob:.1%}")

            if prediction == 'positive':
                st.success("Positive")
            elif prediction == 'negative':
                st.error("Negative")
            else:
                st.warning("Neutral")
