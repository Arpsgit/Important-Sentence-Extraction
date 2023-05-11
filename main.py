import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
stopw = stopwords.words('english')
punc = ['!', '(', ')', '-', '[', ']', '{', '}', ';', ':', '\\', '<', '>', '.', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~', ',', '\'', '\"']


def important_sentences(text):
    r = Rake(stopwords=stopw, punctuations=punc)
    r.extract_keywords_from_text(text)
    keyword = r.get_ranked_phrases_with_scores()
    sentences1 = sent_tokenize(text)
    imp_sen = []
    if len(keyword) >= 5:
        for i in range(5):
            kw = 'we used ' + 'this method ' + 'technique ' + keyword[i][1]
            sentences = sent_tokenize(text)
            sentences.append(kw)
            embedded_sentences = sentence_embedding_model.encode(sentences)
            val = cosine_similarity([embedded_sentences[-1]], embedded_sentences[0:-1])
            indices = sorted(range(len(val[0])), key=lambda index: val[0][index], reverse=True)
            imp_sen.extend(indices[:2])
        imp_sen1 = list(set(imp_sen))
    else:
        imp_sen1 = None

    return imp_sen1, sentences1, keyword[:5]


st.title('Important Sentence Extraction from Research Paper')
input_text = st.text_area("Enter your text: ")


if st.button('Find'):
    if input_text != '':

        imp_sen2, sentences2, keywords = important_sentences(input_text)
        if imp_sen2 != None:
            with st.expander('See key takeaways or important phrases from the text:(Click this drop down)'):
                for kw in keywords:
                    st.write(f':pushpin: {kw[1]}')
            st.subheader('Important Sentences: ')
            for sen in imp_sen2:
                st.text(sentences2[sen])
                st.divider()
        else:
            st.subheader('Enter a long passage of text')
    else:
        st.subheader('Please enter a text!')
