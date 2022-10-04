
<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">University Navigation Chatbot Service</h3>

  <p align="center">
    <a href="https://pdfhost.io/v/td79AkEs2_Report_Chatbotdocx"><strong>PDF Report »</strong></a>
    <br />
    
  </p>
</div>


## Motivation
To find out basic information regarding courses or their enrollment, students in various schools and colleges have to
undergo a tedious process of either navigating websites to reach the desired information or, have to personally go
and get that information. The main purpose of the project is to ensure that students have a common interface, which
in our case is a chatbot, which will help them find out whatever information they have regarding their course,
subjects, scholarships or any college/school related query they might have. We have synthesized our own dataset and
trained a chatbot using various models and NLP techniques in order to obtain the best fit. Along the way we have
made comparisons between some algorithms and plotted the results. We have even made a GUI which would depict
how the chatbot operates.

## Exploratory Data Analysis
- To ensure the chatbot offers a fully customized experience for queries regarding college/school matters, we created
- our own dataset.
- The dataset consists of a list of objects which have a field of ‘tag’. This ‘tag’ helps the model identify in which
domain does the question asked belong to.
- Apart from the ‘tag’ there is a field called ‘patterns’ which contains the sample questions a student might ask.
- The chat-bot is primarily going to be trained on the basis of these patterns.
- Based on these patterns, we have a list of responses corresponding to the pattern of the questions. These responses
and the tag itself serve as target variables and a means to interact with the database as well.
<br />

| **Object Key** | **Type**        | **Description**                                                          |
|----------------|-----------------|--------------------------------------------------------------------------|
| Tag            | String          | Contains the category or type of the question                            |
| Patterns       | List of Strings | Contains sample questions which serve as the input variable for training |
| Responses      | List of Strings | Contains the appropriate responses corresponding to the questions        |

| ![space-1.jpg](https://i.imgur.com/vWgcvyR.png) |
|:--:|
| <b>Figure 1: A sample from the created dataset</b>|
<br />

## Data Preprocessing
In order to generate an input set which would be understandable to whichever model we choose, we used the NLTK
library to carry out the following steps for data preprocessing:
- Tokenization and Segmentation
- Normalization
- Noise Removal
- Conversion into the “bag of words” form
<br />

| ![space-2.jpg](https://i.imgur.com/IkNhPC0.png) |
|:--:|
| <b>Figure 2: Data Preprocessing Pipeline</b>|


### 1. Tokenization and Segmentation
- Converted words into their stems eg: smarter becomes smart
- Tokenization also breaks up the sentence into different words
<br />

| ![space-3.jpg](https://i.imgur.com/e6uNAxH.png) |
|:--:|
| <b>Figure 3: Tokenization Example</b>|

### 2. Normalization
- Converts words like ‘b4’ to ‘before’.
- Normalization also involves converting words into their lower case

| ![space-4.jpg](https://i.imgur.com/SckdDCW.png) |
|:--:|
| <b>Figure 4: Normalization Example</b>|

### 3. Noise Removal
- Involves the removal of stop words, which are the most common words used in sentences but won’t help the model in any way.
- It also helps in removing any stray characters which might be present in a sentence like hashtags or special ascii characters.

| ![space-5.jpg](https://i.imgur.com/4K2cnFu.png) |
|:--:|
| <b>Figure 5: Noise Removal Example</b>|

### 4. The Bag of Words Process
- This pre processing technique allows us to collect all the important words used in the dataset in a list and on the basis of the current question asked to the bot, it assigns a value of 0 to all words in the list but not in the question and +1 to all words in the list which are in the question
- This would serve as the input layer to the model.

| ![space6-.jpg](https://i.imgur.com/3dzEwpD.png) |
|:--:|
| <b>Figure 6:  Converting sentences into the bag of words representation </b>|

## Predictive Models
Traditional models such as SVM could not be used over here because of the input size. As the number of features
increases, the performance of traditional algorithms starts to deteriorate and additional feature engineering might be
required to achieve better results. The deep learning techniques identify the relevant features on their own and
therefore additional feature engineering is not required. As our dataset was limited and also based on some research
we selected the below mentioned algorithms:

- Multi-Layer Perceptron networks (MLP)
- Long Short Term Memory networks (LSTM)

As mentioned in the Exploratory Data Analysis section, a custom dataset was created, which has been used to train
the data. An additional test script was written to test the model by providing a sentence and verifying the tag
predicted by the model. Training loss and accuracy were also considered to judge the model.

Below mentioned are some of the parameters which are common to both the models:
- Hidden layers: These are the layers in between the input and the output layers where the non-linear
transformation of input takes place
- Learning Rate: The learning rate controls the change of the model in response to the estimated error each
time the model weights are updated.
- Activation Function: These are the non-linear transformations that we add to input to decide whether the
neuron should be activated or not.
- Optimizer: These are the methods or algorithm to minimize the error by updating the various other
parameters
- Epoch: It refers to one cycle through the full training dataset
- Batch size: It the number of training examples utilized in one iteration

## Experiments and Results
### LSTM Model
The LSTM model was trained for 500 epochs and produced the following results:

| ![space-7.jpg](https://i.imgur.com/ELnD5rg.png) |
|:--:|
| <b>Figure 7: Epoch vs Loss graph for the LSTM model</b>|

| ![space-8.jpg](https://i.imgur.com/ELnD5rg.png) |
|:--:|
| <b>Figure 8: Epoch vs Accuracy graph for the LSTM model</b>|

The LSTM model never reached 0.00 loss and never gave 100% accuracy.
This was undesired as we want the chatbot to work perfectly on the training data as any question in the training data
must be answered correctly

### MLP Model
|           | layers | epochs | hidden-layer size | learning rate |
|-----------|--------|--------|-------------------|---------------|
| MLP Model | 3      | 1000   | 18                | 0.01          |

The MLP model was trained for 1000 epochs and produced the following results:
#### 1. Comparing different learning rates
- We followed a step based approach in order to find the optimum learning rate. The one which gave
the least loss on the training set and preserved accuracy for the testing set was chosen.
- We found that a higher learning rate did not provide a decreasing loss, thus it shouldn’t be
selected. Upon seeing the graphs and the accuracy on the testing set, we found 0.01 to be the most
optimal learning rate.

| ![space-9.jpg](https://i.imgur.com/pLu9nDJ.png) |
|:--:|
| <b>Figure 9: Graph comparing the loss corresponding to epochs based on the learning rate(High) </b>|

| ![space-10.jpg](https://i.imgur.com/vgChiF5.png) |
|:--:|
| <b>Figure 10: Graph comparing the loss corresponding to epochs based on the learning rate(Low)</b>|

#### 2. Comparing different number of neurons in the hidden layer
- We compared the losses corresponding to multiple cases of the number of neurons selected. The
graphs depict the variations between a low and high number of neurons
- The criteria was that we must select those number of neurons which have consistently dropping
losses and which perform well on the test set. We found 18 neurons did the best job.

| ![space-11.jpg](https://i.imgur.com/3RcaYsI.png) |
|:--:|
| <b>Figure 11: Graph comparing the loss corresponding to epochs(from 50) based on the number of neurons(Low)</b>|

| ![space-12.jpg](https://i.imgur.com/K3bbEJV.png) |
|:--:|
| <b>Figure 12: Graph comparing the loss corresponding to epochs(from 50) based on the number of neurons(High)</b>|

## Chatbot Architecture
We first identified cases in which the chatbot could answer without having to ask multiple questions in
order to get the id/registration number of the student.
- This lead to the creation of a simple event loop which would start with the user inputting some text
and then the text would be processed by the chatbot which would return a response and the
response would directly be presented to the user

In order to target questions which pertain to asking the Id of a student, we took the following approach:
- We introduced memory in the bot by keeping track of the previous question
- Based on the tag identified by the bot, we checked whether it was part of these special cases
- If it was, we allowed the user to input their id, attached the id to the question which was stored and
sent it as an input to the bot
- The bot would then extract this id, fetch the information from the database and then output it to the
student

| ![space-13.jpg](https://i.imgur.com/CEhPkIv.png) |
|:--:|
| <b>Figure 13: Pipeline for questions involving database queries</b>|

### Working Model
The chatbot has been trained to account for scenarios like the following:
- Events hosted in the college
- Department specific events
- Fee for the students enrolled per year
- Faculty list per department
- Course plan for the current semester based on the id of the student and subject
- General course structure for the selected course
- Registered courses based on the id
- Courses for the next semester based on the id
- Scholarships available
- Branch wise waiting list and much more…

The following are a few images demonstrating the working of the bot:

| ![space-14.jpg](https://i.imgur.com/W5L7TyI.png) |
|:--:|
| <b>Figure 14: Chat-Bot working when asked about events</b>|

| ![space-15.jpg](https://i.imgur.com/BQy1m9z.png) |
|:--:|
| <b>Figure 15: Chat-Bot working when asked about courses</b>|

| ![space-16.jpg](https://i.imgur.com/Psfle12.png) |
|:--:|
| <b>Figure 16: Chat-Bot working when asked about course details and faculty</b>|

| ![space-17.jpg](https://i.imgur.com/Bznbbf0.png) |
|:--:|
| <b>Figure 17: The chatbot recognizing general commands</b>|

- The chatbot fetches these courses from the sqlite database
- These courses have been previously stored with the primary key being the id of the student

The database is as follows: 
| ![space-18.jpg](https://i.imgur.com/OIo6DTo.png) |
|:--:|
| <b>Figure 18: Database generation for the chatbot</b>|

## Running the project
### Training 
`python train.py`

### Testing
`python test_bot.py`

### ChatBot
`python chat.py`

### GUI App
`python .\GUI_APP\app.py`
