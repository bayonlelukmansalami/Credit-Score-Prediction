#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('credit_train - Copy.zip')

X = df[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age","Payment_of_Min_Amount", "Monthly_Balance"]]

X["Credit_Mix"] = X["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})

X["Payment_of_Min_Amount"] = X["Payment_of_Min_Amount"].map({"NM": 0, 
                               "Yes": 1, 
                               "No": 0})

y = df[["Credit_Score"]]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify=y, random_state=4)

model = RandomForestClassifier(n_estimators=15,random_state=4)
model.fit(X_train, y_train)

social_acc = ['About', 'Kaggle', 'Medium', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Salami Lukman Bayonle</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Analytics/Scientist (Python/R/SQL/Tableau) \n 
    • Maintenance Specialist (Nigerian National Petroleum Company Limited) \n 
    • IBM/GOOGLE/DATACAMP Certified Data Analyst and Data Scientist''')
    st.sidebar.markdown("[ Visit Github](https://github.com/bayonlelukmansalami)")

elif social_acc_nav == 'Kaggle':
    st.sidebar.image('kaggle.jpg')
    st.sidebar.markdown("[Kaggle](https://www.kaggle.com/bayonlesalami)")

elif social_acc_nav == 'Medium':
    st.sidebar.image('medium.jpg')
    st.sidebar.markdown("[Click to read my blogs](https://medium.com/@bayonlelukmansalami/)")

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/salamibayonlelukman/)")
    



st.title('Credit Score Prediction Web App')
st.write('Banks and credit card companies calculate your credit score to determine your creditworthiness')
st.write("It helps banks and credit card companies immediately to issue loans to customers with good creditworthiness")   
st.write("There are three credit scores that banks and credit card companies use to label their customers:1. Good, 2. Standard, 3. Poor")
st.write("A person with a good credit score will get loans from any bank and financial institution.")


Annual_Income = st.number_input('Annual Income')

Monthly_Inhand_Salary = st.number_input('Monthly Inhand Salary')

Number_of_bank_accounts = st.number_input('Number of Bank Accounts')

Number_of_credit_card = st.number_input('Number of Credit cards')

Interest_rate = st.number_input('Interest rate')

Number_of_loans = st.number_input('Number of Loans')

Averaga_Number_of_days_delayed = st.number_input('Average number of days delayed by the person')

Number_of_delayed_payments = st.number_input('Number of delayed payments')

Credit_Mix = st.number_input('Credit Mix (Bad: 0, Standard: 1, Good: 3)')

Outstanding_debts = st.number_input('Outstanding Debt')

Credit_history_age = st.number_input('Credit History Age')

Payment_Minimun_balance = st.number_input('Payment of Minimum Balance (No: 0, Yes: 1)')

Monthly_Balance = st.number_input('Monthly Balance')




features = [Annual_Income, Monthly_Inhand_Salary, Number_of_bank_accounts, Number_of_credit_card, Interest_rate,
           Number_of_loans, Averaga_Number_of_days_delayed, Number_of_delayed_payments, Credit_Mix,
           Outstanding_debts, Credit_history_age, Payment_Minimun_balance, Monthly_Balance]
            
features_np  = np.array([features])

st.table(features_np)


if st.button('Predict'):
    prediction = model.predict(features_np)
    st.write('Predicted Credit Score = ', model.predict(features_np))


# In[ ]:




