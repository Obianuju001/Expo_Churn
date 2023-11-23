import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns

data = pd.read_csv('expresso_processed.csv')
data.drop(['Unnamed: 0', 'MRG'], axis = 1, inplace = True)
data.head()


# Check if the DEPENDENTS column has been identified into its right datatype
categoricals = data.select_dtypes(include = ['object', 'category'])
numericals = data.select_dtypes(include = 'number')

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()


#-------Streamlit development------
model = pickle.load(open('expModel.pkl', "rb"))

st.markdown("<h1 style = 'color: #00092C; text-align: center; font-family:  'Segoe UI', Tahoma, Geneva, Verdana,  Helvetica, sans-serif'>EXPRESSO CHURN</h1>", unsafe_allow_html=True)
st.markdown("<h6 style = 'margin: -15px; color: #B20600; text-align: center; font-family: Arial, Helvetica, sans-serif'>Churn Prediction for Expresso Clients</p></h6>", unsafe_allow_html=True)

st.image('expresso_image.jpg',width = 550) #---- to give it image
st.markdown("<h5 style='color: #ffffff; background-color: #333333; text-align: center; padding: 5px; font-family: Arial, sans-serif;'>BACKGROUND OF STUDY</h5>", unsafe_allow_html=True)

st.markdown('<br1>', unsafe_allow_html= True)

st.markdown("<h6>Expresso, a well-known telecommunications provider in Africa under the Sudatel Group, connects various nations with vital services like internet and mobile phone service. As one of the major telecom companies in Africa, Expresso is essential to improving connectivity, fostering social interaction, and advancing economic development. This work aims to forecast the likelihood of churn among Expresso customers by utilizing a dataset comprising over 2.5 million users and over 15 behavior elements. To effectively plan customer retention campaigns, handle obstacles, and capitalize on innovation opportunities in this fast-paced sector, telecommunications companies must comprehend and anticipate customer attrition.</h6>", unsafe_allow_html=True)

st.sidebar.image('user_image.png')


input_type = st.sidebar.radio("Select Your Prefered Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    MONTANT = st.sidebar.slider("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE_RECH = st.sidebar.slider("FREQUENCE_RECH", data['FREQUENCE_RECH'].min(), data['FREQUENCE_RECH'].max())
    REVENUE = st.sidebar.slider("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ARPU_SEGMENT = st.sidebar.slider("ARPU_SEGMENT", data['ARPU_SEGMENT'].min(), data['ARPU_SEGMENT'].max())
    FREQUENCE = st.sidebar.slider("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())
    DATA_VOLUME = st.sidebar.slider("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    ON_NET = st.sidebar.slider("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    REGULARITY = st.sidebar.slider("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    TENURE = st.sidebar.select_slider("TENURE", data['TENURE'].unique())
else:
    st.sidebar.header('Input Your Information')
    MONTANT = st.sidebar.number_input("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE_RECH = st.sidebar.number_input("FREQUENCE_RECH", data['FREQUENCE_RECH'].min(), data['FREQUENCE_RECH'].max())
    REVENUE = st.sidebar.number_input("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ARPU_SEGMENT = st.sidebar.number_input("ARPU_SEGMENT", data['ARPU_SEGMENT'].min(), data['ARPU_SEGMENT'].max())
    FREQUENCE = st.sidebar.number_input("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())
    DATA_VOLUME = st.sidebar.number_input("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    ON_NET = st.sidebar.number_input("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    REGULARITY = st.sidebar.number_input("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    TENURE = st.sidebar.number_input("TENURE", data['TENURE'].unique())

st.header('Input Values')

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'TENURE': TENURE, 'MONTANT': MONTANT, 'FREQUENCE_RECH': FREQUENCE_RECH, 'REVENUE': REVENUE, 'ARPU_SEGMENT': ARPU_SEGMENT, 'FREQUENCE': FREQUENCE, 'DATA_VOLUME': DATA_VOLUME, 'ON_NET': ON_NET, 'REGULARITY': REGULARITY}])
st.write(input_variable)


df = data.copy()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in data.drop('CHURN', axis = 1).columns: # ...................................................... If the selected column is found in the general dataframe
        input_variable[i] = scaler.fit_transform(input_variable[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in data.drop('CHURN', axis = 1).columns: # ...................................................... If the selected columns are found in the general dataframe
        input_variable[i] = encoder.fit_transform(input_variable [i])# .................................. encode it


st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h4 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h4>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('CHURNERS Predicted')
    st.image('pred_tick.jpg', width = 200)
    st.success(f'predicted CHURN with provided information is {predicted}')

st.markdown('<br><br>', unsafe_allow_html= True)
st.markdown("<h8>Expresso Churn built by Obianuju Onyekwelu</h8>", unsafe_allow_html=True)

