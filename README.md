# Machine Learning Market Analysis

Goal is to implement a Neural Network on financial data to predict market movement of a stock based solely on quantitative values. Markets are complex adaptive systems, generating alpha is obviously not as easy as using a simple ML implementation with a couple financial features. Want to learn more about Neural Net implementation and how to effectively select features and target variables.

Part 0 of the readme is my notes on Neural Networks and other parts of the project. Parts 1 and 2 are objectives I have set but are loose as the project is more an exploration than a concrete plan.

## Part 0: Exploratory

### Data
Using Finazon to grab histoical stock data. I'm on the cheapest plan which allows 5 calls per minute, no daily limit. I can run in background where I pull minute by minute data for each day within a time period.

### Recurrent Neural Network
Notes prior to implementation. A lot of the math is not really neccessary to know but wanted to better understand deep learning.
#### Sequential Model
* Linear stack of layers where each layer has one input tensor and one output tensor
* This architecture is suitable for sequence-to-value tasks, such as predicting future stock prices based on historical data

#### LSTM
* Long term short term memory, addresses vanishing gradient issue
* Each unit in LSTM layer is a "memory cell" that can retain information over time
* LSTM layer processes the input sequence one step at a time where it produces a vector that represents the LSTM's understanding or memory
    * Encapsulation of all the information it has learned from the entire input sequence, capturing the patterns and features the LSTM uses for making prediction
    * LSTM layers stack, making transformed data an input for the next layer, allowing for complex relationships to be seen
* Parameters
    * Units or neurons, this determines the dimensionality of the output space of this layer
    * return_sequences: 
        * If true the LSTM will return the full sequence of outputs for each input sequence rather than just the output from the last step. 
        * If false the vector will represent the memory at the point last point. From here it will be used for predictions, not training. 

#### Dense
* Fully connected layer where every neuron in a layer is connected to every neuron in the previous layer
    * Connections in network have a weight that the network optimizes during training
* Summary: Dense layers are used to transform the output from a previous layer. 

##### Linear combination
* Each neuron takes a weighted sum of all the inputs it receives from the previous layer, this is known as a linear combination of the inputs
$$z = w_{1} \cdot x_{1} + w_{2} \cdot x_{2} + ... + w_{n} \cdot x_{n} + b$$
* $w_{n}$ are weights (learned parameters) associated with the inputs
* $x_{n}$ are the input values
* $b$ is the bias term (learned parameter)
* z is the output of the linear combination for that neuron
##### Activation function
* Purpose: makes the model more capable of learning more complex relationships, without it the stack of linear layers would be equivalent to a single linear transformation
* Activation functions:
    * Rectified Linear Unit: RELU($z$) = $max(0,z)$
        * Default in Dense
        * Introduce non-linearity and prevent negative output values
    * Sigmoid: $\sigma (z)$ = ${1 \over 1 + e^{-z}}$
        * Often used in the outputy layer for binary classification tasks
    * Tanh: tanh($z$) = ${e^{z} - e^{-z} \over e^{z} + e^{-z}}$
        * Outputs values between -1 and 1, useful when you want the model output to be centered around zero.

#### Other Possible Layers in Neural Net
* GRU (Gated Recurrent Unit): simpler variant of LSTM with fewer parameters
    * GRU(50, return_sequences=True, input_shape=input_shape)
* Dropout: randomly drops some units to prevent overfitting
    * Dropout(0.2)
* Batch Normalization: normalizes outputs of previous layers
    * BatchNormalization()
* Bidirectional LSTM: allows LSTM to capture information from both past and future contexts, processing forward and backward across time steps
    * Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape)
* Convolutional Layer: capture local patterns before feeding to LSTM layer
    * Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)

#### Model Compilation
Sets up the training process, defining how the model's parameters will be updated during training and how model performance is measured.
##### Optimizer
* An optimizer is an algorithm used to adjust the weights of the neural network to minimize the loss function
* Determines how the model learns from the data and updates its weights
* Adam (Adaptive Moment Estimation)
    * Stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
        * The first-order moment tracks the direction in which the gradients are moving, it can be considered an average of the gradients
        * The second-order moment scales the learning rates for each parameter based on the variance of the gradients
            <!-- $$m_{t} = \beta_{1} m_{t-1} +(1-\beta_{1}) g_{t}$$
            * $g_{t}$ is the gradient of the loss function at time step $t$
            * $\beta_{1}$ is the decay rate for the first moment estimate
            * $m_{t-1} is the previous first order moment estimate
            <!-- $$ v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^2$$
            * $v_{t}$ is the second-order moment estimate at a given time step
            * $\beta_2$ is the decay rate for the second moment estimate
            * $g_{t}^2$ os the square of the gradient at a given time step -->
    * https://keras.io/api/optimizers/adam/
    * Learning rate: controls hows much the model's weights are updated during each step of training
        * Smaller will mean smaller smaller updates and more precise convergence
* Loss function
    * Defines how well the model's predictions match the actual target values
    
#### Model Fitting/Training
Looking specifically at .fit() when training neural network model. Function iteratively adjusts the model's parameters (weights and biases) to minimize loss function.
* 'X_train' - training input data, NumPy array or TensorFlow tensor, shape of 'X_train' is usually number of number of features, represents the features that the model will learn from
* 'y_train' target data corresponding to X_train, also a NumPy array or a TensorFlow tensor, shape is number of samples or (number of samples, output dimension)
* 'batch_size' controls how many samples of the training data are processed before the model's internal parameters are updated
    * Typically set to a power of 2
    * Smaller batches can reduce memory requirements and through frequent updates, lead to faster convergence. Smaller batches can add noise to the gradient estimates
* 'epochs' defines how many times the entire training dataset will pass through the model during the training process
    * Each epoch consists of iterations, one for each batch
    * After each epoch the model will have seen all the training data once
    * More epochs allow the model to refine its parameters but too many epochs can lead to overfitting
* 'validation_data' optional data to evaluate the model's generalization after each epoch
    * The validation dataset is used to evaluate the model's ability to generalize to new, unseen data
    * This helps in assessing whether the model is overfitting or underfitting
##### Overall Process
1. Model is initialized with random weights
2. For each epoch
    - Training data is divided into batches, for each batch
        - Forward pass: process input data to produce predictions
        - Loss calculation: loss function compares the predictions to the actual target values
        - Backward Pass: The model calculates gradients of the loss with respect to each parameter using backpropogation
        - Parameter update: The optimizer updates the model's parameters to minimize the loss
    - Validation: After processing all batches, the model's performance on the validation data is evaluated
3. After all epochs are completed, the final trained model is created

### Quantitative Analysis
I want to really look at momentum and high frequency trades because it is less dependent on actual long term market movements which are harder to quantify. In the period of a day, there is less likely to be qualitative factors at play than a year. 

I scrolled through Investopedia to find factors, they are listed below to keep a record. Math behind these is a lot simpler than the math behind the RNN so I left it out, it's implemented in code.

Markets are complex adaptive systems. Using volume and other traditional factors as inputs will probably result in low accuracy models but it's a good starting point. 

#### Relative Strength Index
* Purpose: Signals bullish and bearish momentum
* Meaning: Compares a security's strength on days when prices go up to its strength on days when prices go down
    * 70\< is considered overbought
    * \>30  is considered oversold

#### Moving Average
* Purpose: Indicates trend direction of a stock over a period of time
* Simple Moving Average (SMA): calculated by taking the arithmetic mean of a set of values
* Exponential Moving Average (EMA): gives more weight to recent prices in an attempt to make them more responsive to new information

#### Moving Average Convergence Divergence (MACD)
* Purpose: monitors relationship between two exponential moving averages
* Meaning: Subtract 2 MAs, typically subtracting a 26-day exponential moving average from a 12-day moving average
##### Signal line
* Purpose: helps identify crossover
* Meaning: 9 day EMA
    * When MACD is positive, the short-term average is above the long-term average and is an indicator of upward momentum

#### Price Rate of Change (ROC)
* Purpose: Momentum based indicator that measures the percentage change in price between the current prive and the price a certain number of periods ago
* Meaning: Current price subtracted by an old price, divided by old price

#### Money Flow Index (MFI)
* Purpose: Technical indicator that generates overbought or oversold signals using prices and volume
* Meaning: Looking at previous 14 periods, for each period mark whether the typical price was higher or lower than the prior period 

## Part 1: RNN Implementation
* Get data for stocks for training
* Implement neural network
* Find most effective/interesting method of prediction
    * Local min/max
    * Predicting actual price in next time step, a day, in a week, etc.
    * Maximize profit and make it buy/sell
* Feature engineering: analyze strength of different features

## Part 2: Showcasing
* Web page
    * Django / React stack?
    * AWS
* Showcase findings

## References
* https://keras.io/api/
* https://finazon.io/dataset/sip_non_pro
* https://www.investopedia.com/