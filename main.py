import torch
import pickle
from models.ffnn_1 import SpamClassifier

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


def predict_email(model, vectoriser, email_array):
    X = vectoriser.transform(email_array).toarray()
    #convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(X_tensor)
        # predictions = (outputs > 0.5).int().squeeze().cpu().numpy()
        probs = outputs.squeeze().cpu().numpy()
    return probs

def main():
    VECTORIZER_PATH = "tfidf_vectoriser.pkl"
    MODEL_PATH = "weights/ffnn_model.pth"
    input_emails = ["Free entry in a weekly contest! Click on the link to win a prize", "Can we meet tomorrow at noon?"]




    vectoriser = load_vectorizer(VECTORIZER_PATH)
    model = load_model(vectoriser, MODEL_PATH)
    if not input_emails:
        raise ValueError("Input email list is empty.")

    #predict the emails
    results = predict_email(model, vectoriser, input_emails)

    for text, prob in zip(input_emails, results):
        label_text = "[Spam]" if prob > 0.5 else "[Not Spam]"
        print(f"{label_text} ({prob*100:.2f}%) {text}")


if __name__ == "__main__":
    main()