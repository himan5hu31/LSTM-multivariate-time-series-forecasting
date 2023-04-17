# Multivariate Time Series Forecasting using CNN-LSTM
This project is focused on predicting the future values of multivariate time series data using a Long Short-Term Memory (LSTM) deep learning model. Specifically, we will be using TESLA stock data from Yahoo Finance as our dataset to perform training and forecasting.
![image](https://user-images.githubusercontent.com/130960032/232433493-8b2237e4-ba95-493b-bb69-6e366790cb33.png)
### Prerequisites
To run this project, you will need the following dependencies:

Python 3.x\
TensorFlow\
Pandas\
Matplotlib\
Scikit-learn


### Dataset
The dataset used in this project is TESLA stock data, which was obtained from Yahoo Finance. The dataset contains the following columns:

* Date
* Open
* High
* Low
* Close
* Volume 

## Models
In this project, we have used three different models to forecast sales. These models are:

* CNN Model
* LSTM Model
* CNN-LSTM Model

**CNN Model :**
The CNN model takes the time series data as input and uses a 1D convolutional layer to extract features from the time series. The output of the convolutional layer is then passed through a fully connected layer and a final output layer to generate the forecast.

**LSTM Model :**
The LSTM model takes the time series data as input and uses a sequence of LSTM layers to extract features from the time series. The output of the LSTM layers is then passed through a fully connected layer and a final output layer to generate the forecast.

**CNN-LSTM Model :**
The CNN-LSTM model combines the advantages of both CNN and LSTM models. The time series data is first passed through a 1D convolutional layer to extract features.The output of the convolutional layer is then passed through a sequence of LSTM layers to extract temporal dependencies in the data. Finally, the output of the LSTM layers is passed through a fully connected layer and a final output layer to generate the forecast.
![image](https://user-images.githubusercontent.com/130960032/232433557-47fdc655-3437-4ee9-ab10-47892c89e065.png)


#### Training and Evaluation
The dataset is split into training and testing sets, with 80% of the data used for training and the remaining 20% used for testing. The model is trained on the training data for a specified number of epochs, with early stopping implemented to prevent overfitting.

#### The trained model is then evaluated on the test data using various metrics such as Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).

#### Results
The results of the multivariate time series forecasting using the LSTM model are presented graphically, showing the actual and predicted values of the TESLA stock data. The performance metrics of the model, including MAE and RMSE, are also reported.


### Conclusion
In conclusion, this project demonstrates the effectiveness of LSTM deep learning models for multivariate time series forecasting tasks. By using TESLA stock data as an example, we have shown how the LSTM model can be trained and evaluated to predict the future values of a multivariate time series dataset. This project can be extended to other time series forecasting tasks using different datasets.
