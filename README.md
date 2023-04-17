# Multivariate Time Series Forecasting using LSTM Deep Learning Model
This project is focused on predicting the future values of multivariate time series data using a Long Short-Term Memory (LSTM) deep learning model. Specifically, we will be using TESLA stock data from Yahoo Finance as our dataset to perform training and forecasting.

### Prerequisites
To run this project, you will need the following dependencies:

Python 3.x\
TensorFlow\
Pandas\
Numpy\
Matplotlib\
Scikit-learn\
### Dataset
The dataset used in this project is TESLA stock data, which was obtained from Yahoo Finance. The dataset contains the following columns:

Date\
Open\
High\
Low\
Close\
Volume 

### LSTM Model
The LSTM model used in this project is a deep learning architecture that is well-suited for time series forecasting tasks. It is 
designed to handle the inherent sequential dependencies and long-term dependencies that exist in time series data.

The model consists of multiple LSTM layers, followed by a dense output layer.The input data is prepared as a sequence of past values of the multivariate time series, and the output is a sequence of future values. The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer.

#### Training and Evaluation
The dataset is split into training and testing sets, with 80% of the data used for training and the remaining 20% used for testing. The model is trained on the training data for a specified number of epochs, with early stopping implemented to prevent overfitting.

#### The trained model is then evaluated on the test data using various metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

#### Results
The results of the multivariate time series forecasting using the LSTM model are presented graphically, showing the actual and predicted values of the TESLA stock data. The performance metrics of the model, including MAE and RMSE, are also reported.

### Conclusion
In conclusion, this project demonstrates the effectiveness of LSTM deep learning models for multivariate time series forecasting tasks. By using TESLA stock data as an example, we have shown how the LSTM model can be trained and evaluated to predict the future values of a multivariate time series dataset. This project can be extended to other time series forecasting tasks using different datasets.
