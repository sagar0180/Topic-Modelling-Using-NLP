import streamlit as st 
import pandas as pd
from pickle import dump
from pickle import load
from PIL import Image
import sklearn
import base64
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

CV = CountVectorizer(stop_words="english")

from load_css import local_css

local_css("D:/All Documents/project/P-60 Group 7/style.css")

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 0rem;
    }}
   
</style>
""",
        unsafe_allow_html=True,
    )

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    width: auto;
    height: auto;
    }
  }
    
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
                                      
set_background('Pic1.jpg')                          


col1, col2 = st.beta_columns(2)

names = "<div><span class='blue_heading'>Topic Modelling</span></div>"
col1.markdown(names, unsafe_allow_html=True)

image = Image.open('excelr.png')
col2.image(image, caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

names = "<div><span class='blue'>Anitha-Suraj-Shriniwas</span></div>"
st.markdown(names, unsafe_allow_html=True)


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
     
def clean(doc):
  stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
  punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
  normalized = [lemma.lemmatize(word) for word in punc_free.split()]
  return normalized

sentence = st.text_area("Input your sentence here:",height=200)
sentence1 = [sentence]   
arr_2d = np.reshape(sentence1, (-1, 1))
print(arr_2d)

#loaded_model = load(open('model.pickle', 'rb'))   #pass sentence1
#loaded_model = load(open('method2.pickle', 'rb'))   #pass sentence1
loaded_model = load(open('clf_6.pickle', 'rb'))   #pass sentence1
#loaded_model = load(open('SD-model.pickle', 'rb'))
#loaded_model = load(open('final_model.pickle', 'rb'))
#loaded_model = load(open('ModelXGB.pickle','rb'))
#loaded_model = load(open('gbb_final.pickle', 'rb'))   #pass sentence1

def Convert(string):
    li = list(string.split(" "))
    return li

  
# Driver code    
#sentence = Convert(sentence)
#test = np.array(test).reshape(1, -1)
#sentence1 = CV.fit_transform(sentence)


#t_car = "The greater the distance driven per year, the more likely the total cost of ownership for an electric car will be less than for an equivalent ICE car.[53] The break even distance varies by country depending on the taxes, subsidies, and different costs of energy.  In some countries the comparison may vary by city, as a type of car may have different charges to enter different cities; for example, the UK city of London charges ICE cars more than the UK city of Birmingham does"
#t_air = "An aircraft is a vehicle or machine that is able to fly by gaining support from the air. It counters the force of gravity by using either static lift or by using the dynamic lift of an airfoil, or in a few cases the downward thrust from jet engines."
#t_ev = "The cost of installing charging infrastructure has been estimated to be repaid by health cost savings in less than 3 years."
#t_hyb = "The twelfth generation of the Corolla line-up was launched in Brazil in September 2019, which included an Altis trim with the first version of a flex-fuel hybrid powered by a 1.8-litre Atkinson engine. By February 2020, sales of "
#t_plug = "TMC experienced record sales of hybrid cars during 2013, with 1,279,400 units sold worldwide, and it took nine months to achieve one million hybrid sales. Again in 2014, TMC sold a record one million hybrids in nine months. "
#sentence1 = [t_plug]
#t1 = CV.fit_transform(sentence)

if(len(sentence)!=0):
    prediction = loaded_model.predict(sentence1)
    #col1.markdown(prediction)
    if(prediction==[0]): col1.markdown("<div><span class='purple'>TOPIC - Electric Aircraft</span></div>",unsafe_allow_html=True)
    if(prediction==[1]): col1.markdown("<div><span class='purple'>TOPIC - Electric Car</span></div>",unsafe_allow_html=True)
    if(prediction==[2]): col1.markdown("<div><span class='purple'>TOPIC - Electric Vehicle</span></div>",unsafe_allow_html=True)
    if(prediction==[3]): col1.markdown("<div><span class='purple'>TOPIC - Hybrid EV</span></div>",unsafe_allow_html=True)
    if(prediction==[4]): col1.markdown("<div><span class='purple'>TOPIC - PlugIn EV</span></div>",unsafe_allow_html=True)
