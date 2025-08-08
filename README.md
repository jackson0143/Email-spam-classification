# Email Spam Classifier

A pipeline for classifying emails as **spam** or **not spam** using neural networks (forwardfeed). Currently the model has an accuracy of approximately **90%** on unseen data
### 📊 Classification Report

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Ham** | 0.92 | 0.95 | 0.93 | 58 |
| **Spam** | 0.88 | 0.81 | 0.84 | 26 |
| **Accuracy** |       |       | **0.90** | 84 |
| **Macro avg** | 0.90 | 0.88 | 0.89 | 84 |
| **Weighted avg** | 0.90 | 0.90 | 0.90 | 84 |

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


