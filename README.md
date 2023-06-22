# Twitter Sentimental Analysis using RNN and LSTM

### Step 1: Downloading and Uploading dataset

 First, we will download the twitter sentiment dataset from Kaggle the link is given below:

 **Dataset Link:** https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv

 After downloading this the word in labels as follows:

 **Negative Sentiment:** 0

 **Positive Sentiment:** 1

 After this we will upload the data on our drive that we will mount on the Google Colab.

 ### Step 2: Changing runtime and loading the data in Colab

Create a new notebook in Colab\
Go on the Runtime tab and change the Runtime type to GPU and save it.
Mount the Drive in which you uploaded the Dataset you want to train the model. 

**Output**
<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210324947-b333a83b-66df-4c61-8f55-f8148b6f8483.png">
</p>

 ### Step 3: Vectorizing the embedded dataset                      

Now we will run  to vectorize the words.         
And load it into the dictionary function so it can be called directly.                                                         


### Step 4: Downloading required NLP libraries and Tokenizing the data

Download wordnet and omw file to deal with string data from NLP library After this we will tokenize the data. Which divides the data into small parts, so the sentences become easier to use or translate into other language or format. After this we will Lemmatizer our data. Lemmatizer will convert the words to the most common form of that word or the most used similar word so it is easier to  detect.

**Required Lib:**

nltk.download('omw-1.4')

nltk.download('wordnet')

**Output**
Before tokenizing:
<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210325887-843e3cad-92bf-4909-a504-8c6ae872d1ba.png">
</p>

 After tokenizing:

 <p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210325948-d0a25ea6-cdc9-44f0-8c73-be9937a73b98.png">
</p>

Before Lemmatizer:

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210326015-50620f78-81bb-47af-9705-c7a159651d4d.png">
</p>

After Lemmatizer:

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210326051-13912715-c88b-4bdc-bfc4-7fa1b399d485.png">
</p>

### Step 5: Vectorizing and Dividing Test and Training data

Now we will create a function that will loop and vectorize the data and return as a float/decimal valued data.
Now we will divide the data into training and testing with 70% for training and 30% for testing which is considered a standard.

### Step 6: Finding out the overall size of data and padding

We will define df_to_X\_y so we can count the sentences size
Then will plot to see the max and min num of tokens our dataset has in a sentence.                                                       
After this we will pad our dataset to the max size of the token, so we don't get bad results. So, our training is accurate, and our shape of data is relevant.         
Function for counting the token size for our dataset
Now we print and get an output graph for the token size

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210326719-520c7306-af5e-4a2e-8a56-50abbc28b5ba.png">
</p>


<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210326736-c7062520-4a05-4283-a900-88e71aaa799a.png">
</p>

### Step 7: Modeling RNN
We will create our model by designing Lstm, Dropout, Dense and flatten layer. We will get lstm layers to assign weights Dropout to remove overfitting Dense to get the most values Then flatten to get the output layer.
 
 We are using shape 57 as the max token size is 50
 Using three LSTM layers and 64 sized filter
 Dropout of 0.2 so our model doesn't over fit
 Flatten over layer in the end
 And dense to get the values and sigmoid activation function because it's a binary class.

 <p align="center">
<img width="620" height="520" src="https://user-images.githubusercontent.com/73955220/210327272-0038c75f-2233-423f-89d8-5e87b5eb9a55.png">
</p>

### Step 8: Compiling model                                    
First, we are defining the location where we want to save our trained weights.                                                
Optimizer we used is Adam.                                        
Loss is Binary Crossentropy.                                     


### Step: 9 Fitting the model

Now we will fit our model using the fit command. 
We are giving 20 epochs to train our data.

### Step: 10 Loading our model and printing the accuracy of our model

Now we will load our trained model and the weights.
The give it a test data to get the accuracy of our model given below.

 **Output:**

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210327787-5da9b6d7-80da-4d40-9b7a-78e8a758d47a.png">
</p>


We have accuracy of positive sentiment: 88 %

We have accuracy of negative sentiment: 81 %

### Step 11: Now we will give it a tweet to check if it gives good results                                                         

**Output:**

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210328333-93c3a377-ee1d-4cb0-a8fc-ab0ded7a698f.png">
</p>

**Actual:** Positive Sentiment

**Predicted:** Positive Sentiment

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/210328497-ed7f2661-f7ae-4ad5-8590-5bf2f00078b7.png">
</p>

 **Actual:** Negative Sentiment

 **Predicted:** Negative Sentiment

