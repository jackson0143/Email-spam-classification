# Email Spam Classifier

A pipeline for classifying emails as **spam** or **not spam** using neural networks (forwardfeed). Currently the model has an accuracy of approximately **95%**
### 📊 Classification Report

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Ham** | 0.98 | 0.99 | 0.99 | 975 |
| **Spam** | 0.95 | 0.90 | 0.92 | 157 |
| **Accuracy** |       |       | **0.98** | 1132 |
| **Macro avg** | 0.97 | 0.94 | 0.95 | 1132 |
| **Weighted avg** | 0.98 | 0.98 | 0.98 | 1132 |

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


