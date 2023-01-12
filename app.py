import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st

df=pickle.load(open('df.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

#background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://wesecureapp.com/wp-content/uploads/2022/01/Tinted-Bg-5-1-%E2%80%93-24-960x604.png");;
background-size: 100%;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
   
#title
st.title('Heart Disease Prediction'.upper())

# Input Features

#age
age=st.number_input('Age',min_value=18)

#sex
sex=(lambda x:1 if x=='Male' else 0)(st.selectbox('Sex',options=['Male','Female']))

#cp
cp=st.selectbox('Chest Pain Type',options=['Typical Angina','Non-anginal Pain','Atypical Angina','Asymptomatic'])

#trestbps
trestbps=st.number_input('Resting Blood Pressure')

#chol
chol=st.number_input('Serum Cholestoral in mg/dl')

#fbs
fbs=(lambda x:1 if x=='Yes' else 0)(st.selectbox('Fasting Blood Sugar > 120 mg/dl',options=['Yes','No']))

#restecg
restecg=st.selectbox('Resting Electrocardiographic Results',['Normal','Abnormality','Left Ventricular Hypertrophy'])

#thalach
thalach=st.number_input('Maximum Heart Rate Achieved')

#exang
exang=(lambda x:1 if x=='Yes' else 0)(st.selectbox('Exercise Induced Angina',options=['Yes','No']))

#oldpeak
oldpeak=st.number_input('ST depression')

#slope
slope=st.selectbox('ST/HR slope',options=[0,1,2])

#ca
ca=st.selectbox('Number Of Major Vessels (0-3)',options=[0,1,2,3,4])

#thal
thal=st.number_input('Thalassemia')

#preprocessing the features

#cp
if cp=='Typical Angina':
    cp=0
elif cp=='Non-anginal Pain':
    cp=2
elif cp=='Atypical Angina':
    cp=1
elif cp=='Asymptomatic':
    cp=3

#restecg
if restecg=='Normal':
    restecg=0
elif restecg=='Abnormality':
    restecg=1
elif restecg=='Left Ventricular Hypertrophy':
    restecg=2

sc=StandardScaler()
columns=['age','trestbps','chol','thalach','cp','restecg','slope','ca','thal','oldpeak']
sc.fit_transform(df[columns])
features=list(sc.transform([[age,trestbps,chol,thalach,cp,restecg,slope,ca,thal,oldpeak]])[0])

if st.button('Predict'):
    query=[features[0],sex,features[4],features[1],features[2],fbs,features[5],features[3],exang,features[9],features[6],features[7],features[8]]
    st.title("The Patient is predicted with " +(lambda x: 'a Heart Disease' if x==1 else 'no Heart Disease')(model.predict(np.array([query]))))


