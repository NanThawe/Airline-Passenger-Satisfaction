import numpy as np
import pandas as pd
import os
import plotly.express as px
import streamlit as st
import joblib
from utilities import new_process

model = joblib.load("xgboost.pkl")

TRAIN_PATH = os.path.join(os.getcwd(), "train.csv")
df_train = pd.read_csv(TRAIN_PATH, index_col="Unnamed: 0")
TEST_PATH = os.path.join(os.getcwd(), "test.csv")
df_test = pd.read_csv(TEST_PATH, index_col="Unnamed: 0")
df = pd.concat([df_train, df_test], axis=0)

df.drop(columns=["id"], inplace=True)

# convert column names to string
conv_2_str = ['Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                 'Online boarding', 'Seat comfort','Inflight entertainment', 'On-board service', 'Leg room service','Baggage handling', 'Checkin service', 'Inflight service','Cleanliness']
df[conv_2_str] = df[conv_2_str].astype("str")

st.set_page_config(
    layout='wide',
    page_title='Airline Passenger Satisfaction App',
    page_icon='👥'
)
st.header('Airline Passenger Satisfaction Prediction', divider='rainbow')

def satisfaction_deployment():
        
        # categorical features
        Customer_Type = st.selectbox("Customer Type", options=['Loyal Customer', 'disloyal Customer'])
        type_of_travel = st.selectbox("Type of Travel", options=['Personal Travel', 'Business travel'])
        Class = st.selectbox("Class", options=['Eco Plus', 'Business', 'Eco'])

        # numerical features
        flight_Distance = st.number_input("Input approximate flight distance".title(), value=int(df["Flight Distance"].mean()), step=100)
        arrival_delay = st.number_input("Input Arrival Delay (Minutes)".title(), value=df["Arrival Delay in Minutes"].median(), step=10.0)
        age = st.number_input("Age", min_value=1, max_value=100, step=1)
        # Radio button questions arranged side by side
        col1, col2= st.beta_columns(2)

        with col1:
            Leg_room = st.radio("Leg room service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            online_boarding = st.radio("Online boarding rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            onboard_service = st.radio("On-board service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            online_booking = st.radio("Online booking rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            inflight_entertainment = st.radio("Inflight entertainment rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            seat_comfort = st.radio("Seat comfort rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        with col2:
            food_drink = st.radio("Food and drink rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            baggage_handling = st.radio("Baggage handling rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            inflight_wifi = st.radio("Inflight wifi service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)  
            cleanliness = st.radio("Cleanliness rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
            inflight_service = st.radio("Inflight service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True) 
            checkin_service = st.radio("Checkin service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)   
        
        if st.button('Predict Satsfication'):
            new_input = np.array([flight_Distance, arrival_delay, age,
                                  Customer_Type, type_of_travel, Class,
                                  online_booking, Leg_room, online_boarding, inflight_service,
                                  inflight_wifi, food_drink, inflight_entertainment, cleanliness,
                                  onboard_service, baggage_handling, seat_comfort, checkin_service])
            
            sample_processed = new_process(new_sample=new_input)
            pred = model.predict(sample_processed)
            
            # dissatisfied = 0
            # satisfied = 1
            if pred[0] == 1:
                pred = "satisfied"
            elif pred[0] == 0:
                pred = "dissatisfied"
        
            st.success(f'Satsfication Prediction is : {(pred.title())}')
            
            if (pred == "satisfied"):
                st.image("happy.jpg")
            elif (pred =="dissatisfied"):
                st.image("unhappy.jpg")
            
if __name__ == '__main__':
    satisfaction_deployment() 