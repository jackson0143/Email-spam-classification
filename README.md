# Email Spam Classifier

A machine learning pipeline for classifying emails as **spam** or **not spam** using neural networks. Currently the model has an accuracy of **95%**

The dataset used is the [Spam email classification - Ashfak Yeafi](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) and [Email Spam Text Classification Dataset - Kucev Roman](https://www.kaggle.com/datasets/tapakah68/email-spam-classification) which contains the content of the email messages with labels
- 'Spam' indicates that the email is classified as spam.
- 'Ham' (or 'not spam') denotes that the email is not spam.

<br/>

All testing was done in notebooks/main.ipynb and combined_dataset.ipynb. The final python script main.py can be edited to use your own dataset (provided that the data is cleaned)

Future development involves:
- training with more datasets
- setting up a validation set for training accuracy
- optimising the parameters to see if a better accuracy is achieved
- refactoring the file structure/code for a more dynamic flow
- Perhaps even interactive on a site??
### 🔧 Tech Stack
- Pytorch & sklearn
- Pandas for dataframes



