import re
import string
from pprint import pprint

import pandas as pd

import nltk
from nltk import word_tokenize,pos_tag
from nltk.stem import wordnet
from nltk.corpus import stopwords
from pip._vendor.distlib.compat import raw_input

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import pairwise_distances

###download essential package
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def read_dataset():
    df=pd.read_csv('chat_data.tsv',sep='\t',
                           header=0, encoding='utf-8', engine='python')
    print(df.head())
    print(df.shape[0])
    return df

def text_normalization(txt):
    txt=str(txt).lower()
    #tokenizer = RegexpTokenizer(r'\w+')
    clean_txt=re.sub(r'[^a-z]',' ',txt) #remove special char
    tokens=word_tokenize(clean_txt)
    #print(tokens)
    lema=wordnet.WordNetLemmatizer()
    tags_list=pos_tag(tokens,tagset=None)
    lema_words=[]
    #pprint(tags_list)
    for token,pos_t in tags_list:
        pos_val=''
        if(pos_t.startswith('V')):
            pos_val='v'
        elif(pos_t.startswith('J')):
            pos_val='a'
        elif(pos_t.startswith('R')):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_words.append(lema_token)
    scenten_with_stopword=" ".join(lema_words)
    return stopword_removing(scenten_with_stopword)

def stopword_removing(txt):
    stop=stopwords.words('english')
    #print(stop)
    Q=[]
    a=txt.split()
    for i in a:
        if i in stop:
            continue
        else:
            Q.append(i)
    return " ".join(Q)

def model(cv):
    df = read_dataset()
    # lema_scentence=text_normalization('you already making me the greatest hurt?')
    # print(lema_scentence)
    print(df.columns.tolist())
    # print(df['Question'])
    df['lemmatize_text'] = df['Question'].apply(text_normalization)
    print(df.head())
    X = cv.fit_transform(df['lemmatize_text']).toarray()
    features = cv.get_feature_names()
    df_bow = pd.DataFrame(X, columns=features)
    return cv,df,df_bow

#
def chat_tfidf(df_bow,question_bow,df):
    cosine_value = 1 - pairwise_distances(df_bow, question_bow, metric='cosine')
    #pprint(cosine_value)
    index_value = cosine_value.argmax()
    answer=df['Answer'].loc[index_value]
    return answer

if __name__ == '__main__':
    df=read_dataset()
    # #lema_scentence=text_normalization('you already making me the greatest hurt?')
    # #print(lema_scentence)
    # print(df.columns.tolist())
    # #print(df['Question'])
    df['lemmatize_text']=df['Question'].apply(text_normalization)
    # print(df.head())
    cv = CountVectorizer()
    #cv,df,df_bow=model(cv)
    X=cv.fit_transform(df['lemmatize_text']).toarray()
    features=cv.get_feature_names()
    df_bow=pd.DataFrame(X,columns=features)
    pprint(df_bow.head(100))
    # print(df_bow.shape[0])
    #question='want to meet'
    print("Let's chat:")
    while True:
        question=raw_input()
        print(question)
        question_lema=text_normalization(question)
        question_bow=cv.transform([question_lema]).toarray()
        print(question_bow)
        answer=chat_tfidf(df_bow,question_bow,df)
        print(answer)



