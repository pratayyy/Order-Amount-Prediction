import numpy as np
import pandas as pd
import pickle

def predict(df):
    df = pd.DataFrame(df)

    df.fillna("NaN", inplace = True)

    df['ORDER_CREATION_DATE'] = pd.to_datetime(df['ORDER_CREATION_DATE'], format = "%Y%m%d", errors = 'coerce')
    df['REQUESTED_DELIVERY_DATE'] = pd.to_datetime(df['REQUESTED_DELIVERY_DATE'], format = "%Y%m%d", errors = 'coerce')
    
    df.drop(df[df['ORDER_CREATION_DATE'] > df['REQUESTED_DELIVERY_DATE']].index, inplace = True)
                                                   
    df['ORDER_AMOUNT'] = df['ORDER_AMOUNT'].astype(str).replace('-', '')
    df['RELEASED_CREDIT_VALUE'] = df['RELEASED_CREDIT_VALUE'].astype(str).replace('-', '')
                                                   
    df['ORDER_AMOUNT'] = df['ORDER_AMOUNT'].str.replace(',', '.')                                               
    df['RELEASED_CREDIT_VALUE'] = df['RELEASED_CREDIT_VALUE'].str.replace(',', '.')

    exchange_rates = {
    'USD':1,                                                     
    'EUR': 1.08,   
    'AUD': 0.65,    
    'CAD': 0.74,    
    'GBP': 1.24,    
    'MYR': 0.22,    
    'PLN': 0.24,    
    'AED': 0.27,   
    'HKD': 0.13,     
    'CHF': 1.11,     
    'RON': 0.22,     
    'SGD': 0.74,     
    'CZK': 0.045,     
    'HU1': 0.0029,     
    'NZD': 0.61,       
    'BHD': 2.65,      
    'SAR': 0.27,       
    'QAR': 0.27,       
    'KWD': 3.25,       
    'SEK': 0.094
    }
    df['ORDER_AMOUNT'] = pd.to_numeric(df['ORDER_AMOUNT'], errors = 'coerce')
    df['amount_in_usd'] = (df['ORDER_AMOUNT'] * df['ORDER_CURRENCY'].map(exchange_rates)).round(2)  
                                                   
    df['unique_cust_id'] = df['CUSTOMER_NUMBER'].astype(str) + df['COMPANY_CODE'].astype(str)

    q1, q3 = np.percentile(df['amount_in_usd'], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outlier_index = df[(df['amount_in_usd'] < lower_bound) | (df['amount_in_usd'] > upper_bound)].index
    df.loc[outlier_index, 'amount_in_usd'] = df['amount_in_usd'].median()
    
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['DISTRIBUTION_CHANNEL'] = label_encoder.fit_transform(df['DISTRIBUTION_CHANNEL'])
    df['DIVISION'] = label_encoder.fit_transform(df['DIVISION'])
    df['PURCHASE_ORDER_TYPE'] = label_encoder.fit_transform(df['PURCHASE_ORDER_TYPE'])
    df['CREDIT_CONTROL_AREA'] = label_encoder.fit_transform(df['CREDIT_CONTROL_AREA'])
    df['ORDER_CURRENCY'] = label_encoder.fit_transform(df['ORDER_CURRENCY'])
    df['RELEASED_CREDIT_VALUE'] = label_encoder.fit_transform(df['RELEASED_CREDIT_VALUE'])
    df['unique_cust_id'] = label_encoder.fit_transform(df['unique_cust_id'])
    
    df['ORDER_AMOUNT'] = df['ORDER_AMOUNT'].astype(float)
    df['CREDIT_STATUS'] = df['CREDIT_STATUS'].astype(float)
    df['amount_in_usd'] = df['amount_in_usd'].astype(float)
    
    x = ['CUSTOMER_ORDER_ID','ORDER_CREATION_TIME','SOLD_TO_PARTY','ORDER_AMOUNT','CUSTOMER_NUMBER','amount_in_usd','COMPANY_CODE','SALES_ORG']
    for col in x:
        df[col] = df[col].apply(lambda x: np.log1p(x))
        
    df['duration_days'] = (df['REQUESTED_DELIVERY_DATE'] - df['ORDER_CREATION_DATE']).dt.days
    df['duration_weeks'] = (df['REQUESTED_DELIVERY_DATE'].dt.week - df['ORDER_CREATION_DATE'].dt.week) % 52
    
    def difference_in_days(melt, lags, ffday, customer_id_col, create_date_col, net_amount_col):
        for i in range(ffday, lags+1):
            melt['last-'+str(i)+'day_sales'] = melt.groupby([customer_id_col])[net_amount_col].shift(i)

        melt = melt.reset_index(drop = True)

        for i in range(ffday, lags+1):
            melt['last-'+str(i)+'day_diff']  = melt.groupby([customer_id_col])['last-'+str(i)+'day_sales'].diff()
        melt = melt.fillna(0)
        return melt

    difference_in_days(df, 3, 1, 'unique_cust_id', 'ORDER_CREATION_DATE', 'amount_in_usd')
    
    df = df.drop('CREDIT_STATUS', axis = 1) 
    df.fillna(0, inplace = True)
    
    df = df.drop(['amount_in_usd', 'ORDER_AMOUNT', 'ORDER_CREATION_DATE', 'REQUESTED_DELIVERY_DATE', 'CUSTOMER_NUMBER', 'COMPANY_CODE', 'CREDIT_CONTROL_AREA', 'SOLD_TO_PARTY', 'SALES_ORG', 'DISTRIBUTION_CHANNEL', 'CUSTOMER_ORDER_ID'], axis = 1)

    import xgboost as xgb
    
    model = pickle.load(open("finalized_model.sav", 'rb'))
    
    prediction = np.expm1(model.predict(df))
    
    return prediction