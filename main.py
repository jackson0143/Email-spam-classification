import torch
import pickle
from models.ffnn_1 import SpamClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def load_data(path):
    df = pd.read_csv(path)
    return df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_vectorizer(path):
    with open(path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer


def load_model(vectoriser, model_path):
    #the input size is the number of features in the vectoriser, which we defined as 3000
    #this essentially means that every email will be represented by 3000 features
    input_size = len(vectoriser.get_feature_names_out())
    model = SpamClassifier(n_features=input_size,n_hidden=[128,64]).to(device)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    return model

def clean_data_emailspam2(df):
    df = df.dropna()
    df = df.iloc[:-1]
    df = df[['Category', 'Message']]  
    # Handle the label encoding
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Category'])
    return df

def predict_email(model, vectoriser, email_array):
    X = vectoriser.transform(email_array).toarray()
    #convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(X_tensor)
        # predictions = (outputs > 0.5).int().squeeze().cpu().numpy()
        probs = outputs.squeeze().cpu().numpy()
    return probs




def predict_and_evaluate(model, vectoriser, df):
    X = vectoriser.transform(df['Message']).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = outputs.squeeze().cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)
    
    # Calculate accuracy
    true_labels = df['Label'].values
    accuracy = (predictions == true_labels).mean()
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total samples: {len(true_labels)}")
    print(f"Correct predictions: {(predictions == true_labels).sum()}")
    print(f"Incorrect predictions: {(predictions != true_labels).sum()}")
    
    # Print classification report
    print(f"\n=== Classification Report ===")
    print(classification_report(true_labels, predictions, target_names=['Not Spam', 'Spam']))
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels,
    }



def main():
    VECTORIZER_PATH = "vectorisers/tfidf_vectoriser_v2.pkl"
    MODEL_PATH = "weights/ffnn_model_v2.pth"

    
    df = load_data('datasets/email_spam2.csv')
    df = clean_data_emailspam2(df)


    vectoriser = load_vectorizer(VECTORIZER_PATH)
    model = load_model(vectoriser, MODEL_PATH)
    results = predict_and_evaluate(model, vectoriser, df)
  
if __name__ == "__main__":
    main()
