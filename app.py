
# # import pandas as pd
# # from datetime import timedelta
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.metrics import mean_absolute_error, mean_squared_error

# # file_path = "transformed_data.csv" 

# # try:
# #     data = pd.read_csv(file_path)  
# #     print(f"Data preview:\n{data.head()}")
# # except Exception as e:
# #     print(f"Error loading data: {e}")
# #     exit()


# # required_columns = {'date', 'outlet_id', 'product', 'quantity'}
# # if not required_columns.issubset(data.columns):
# #     print(f"Missing required columns. Expected: {required_columns}, Found: {set(data.columns)}")
# #     exit()


# # data['date'] = pd.to_datetime(data['date'], errors='coerce')


# # data = data[data['product'] == 'Farmhouse Pizza']


# # data.sort_values(by=['outlet_id', 'date'], inplace=True)
# # data['day_of_week'] = data['date'].dt.dayofweek


# # data['sales_lag_1'] = data.groupby('outlet_id')['quantity'].shift(1)
# # data['sales_lag_7'] = data.groupby('outlet_id')['quantity'].shift(7)


# # data.dropna(inplace=True)


# # X = data[['outlet_id', 'day_of_week', 'sales_lag_1', 'sales_lag_7']]
# # y = data['quantity']

# # X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)


# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # model = GradientBoostingRegressor(random_state=42)
# # model.fit(X_train, y_train)


# # y_pred = model.predict(X_test)
# # mae = mean_absolute_error(y_test, y_pred)
# # rmse = mean_squared_error(y_test, y_pred, squared=False)

# # print(f"Model Performance:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")


# # today = data['date'].max()
# # next_day = today + timedelta(days=1)


# # outlet_ids = data['outlet_id'].unique()
# # predictions = []

# # for outlet_id in outlet_ids:
# #     recent_data = data[data['outlet_id'] == outlet_id].iloc[-1]
# #     sales_lag_1 = recent_data['quantity']
    

# #     sales_lag_7 = data[(data['outlet_id'] == outlet_id) & 
# #                        (data['date'] == today - timedelta(days=7))]['quantity'].values
# #     sales_lag_7 = sales_lag_7[0] if len(sales_lag_7) > 0 else 0

   
# #     input_features = {
# #         'outlet_id': outlet_id,
# #         'sales_lag_1': sales_lag_1,
# #         'sales_lag_7': sales_lag_7,
# #     }
    

# #     for day in range(1, 7): 
# #         input_features[f'day_of_week_{day}'] = 1 if day == next_day.weekday() else 0


# #     input_df = pd.DataFrame([input_features])


# #     predicted_sales = model.predict(input_df)[0]
# #     predictions.append({'outlet_id': outlet_id, 'predicted_sales': predicted_sales})

# # predictions_df = pd.DataFrame(predictions)
# # print("\nPredictions for next day:")
# # print(predictions_df)
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import streamlit as st

# # Load the dataset
# st.title("Pizza Sales Prediction")
# st.sidebar.header("Upload CSV File")
# file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# if file:
#     try:
#         data = pd.read_csv(file)
#         st.write("### Data Preview", data.head())
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.stop()

#     required_columns = {'date', 'outlet_id', 'product', 'quantity'}
#     if not required_columns.issubset(data.columns):
#         st.error(f"Missing required columns. Expected: {required_columns}, Found: {set(data.columns)}")
#         st.stop()

#     # Data preprocessing
#     data['date'] = pd.to_datetime(data['date'], errors='coerce')
#     data = data[data['product'] == 'Farmhouse Pizza']
#     data.sort_values(by=['outlet_id', 'date'], inplace=True)
#     data['day_of_week'] = data['date'].dt.dayofweek
#     data['sales_lag_1'] = data.groupby('outlet_id')['quantity'].shift(1)
#     data['sales_lag_7'] = data.groupby('outlet_id')['quantity'].shift(7)
#     data.dropna(inplace=True)

#     # Prepare features and target
#     X = data[['outlet_id', 'day_of_week', 'sales_lag_1', 'sales_lag_7']]
#     y = data['quantity']
#     X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model training
#     model = GradientBoostingRegressor(random_state=42)
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)

#     st.write("### Model Performance")
#     st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#     st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#     # Prediction for the next day
#     today = data['date'].max()
#     next_day = today + timedelta(days=1)

#     outlet_ids = data['outlet_id'].unique()
#     predictions = []
#     for outlet_id in outlet_ids:
#         recent_data = data[data['outlet_id'] == outlet_id].iloc[-1]
#         sales_lag_1 = recent_data['quantity']
#         sales_lag_7 = data[(data['outlet_id'] == outlet_id) &
#                            (data['date'] == today - timedelta(days=7))]['quantity'].values
#         sales_lag_7 = sales_lag_7[0] if len(sales_lag_7) > 0 else 0

#         input_features = {
#             'outlet_id': outlet_id,
#             'sales_lag_1': sales_lag_1,
#             'sales_lag_7': sales_lag_7,
#         }
#         for day in range(1, 7):
#             input_features[f'day_of_week_{day}'] = 1 if day == next_day.weekday() else 0

#         input_df = pd.DataFrame([input_features])
#         predicted_sales = model.predict(input_df)[0]
#         predictions.append({'outlet_id': outlet_id, 'predicted_sales': predicted_sales})

#     predictions_df = pd.DataFrame(predictions)
#     st.write("### Predictions for Next Day")
#     st.dataframe(predictions_df)

#     # Visualize predictions
#     st.write("### Sales Predictions")
#     st.bar_chart(predictions_df.set_index('outlet_id')['predicted_sales'])
# else:
#     st.info("Please upload a CSV file to proceed.")

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import io


st.title("Pizza Sales Prediction")
st.sidebar.header("Upload and Configure")
file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])


n_estimators = st.sidebar.slider("Number of Estimators", min_value=50, max_value=300, value=100, step=10)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)

if file:
    try:
        data = pd.read_csv(file)
        st.write("### Data Preview", data.head())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    required_columns = {'date', 'outlet_id', 'product', 'quantity'}
    if not required_columns.issubset(data.columns):
        st.error(f"Missing required columns. Expected: {required_columns}, Found: {set(data.columns)}")
        st.stop()

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data[data['product'] == 'Farmhouse Pizza']
    data.sort_values(by=['outlet_id', 'date'], inplace=True)
    data['day_of_week'] = data['date'].dt.dayofweek
    data['sales_lag_1'] = data.groupby('outlet_id')['quantity'].shift(1)
    data['sales_lag_7'] = data.groupby('outlet_id')['quantity'].shift(7)
    data.dropna(inplace=True)


    X = data[['outlet_id', 'day_of_week', 'sales_lag_1', 'sales_lag_7']]
    y = data['quantity']
    X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.write("### Model Performance")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
    st.bar_chart(feature_importance.set_index('Feature'))

  
    today = data['date'].max()
    next_day = today + timedelta(days=1)

    outlet_ids = data['outlet_id'].unique()
    predictions = []
    for outlet_id in outlet_ids:
        recent_data = data[data['outlet_id'] == outlet_id].iloc[-1]
        sales_lag_1 = recent_data['quantity']
        sales_lag_7 = data[(data['outlet_id'] == outlet_id) & 
                           (data['date'] == today - timedelta(days=7))]['quantity'].values
        sales_lag_7 = sales_lag_7[0] if len(sales_lag_7) > 0 else 0

        input_features = {'outlet_id': outlet_id, 'sales_lag_1': sales_lag_1, 'sales_lag_7': sales_lag_7}
        for day in range(1, 7):
            input_features[f'day_of_week_{day}'] = 1 if day == next_day.weekday() else 0

        input_df = pd.DataFrame([input_features])
        predicted_sales = model.predict(input_df)[0]
        predictions.append({'outlet_id': outlet_id, 'predicted_sales': predicted_sales})

    predictions_df = pd.DataFrame(predictions)
    st.write("### Predictions for Next Day")
    st.dataframe(predictions_df)

   
    st.write("### Sales Predictions")
    st.bar_chart(predictions_df.set_index('outlet_id')['predicted_sales'])

 
    st.write("### Download Predictions")
    buffer = io.BytesIO()
    predictions_df.to_csv(buffer, index=False)
    st.download_button(label="Download Predictions as CSV", data=buffer.getvalue(),
                       file_name="predictions.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to proceed.")
