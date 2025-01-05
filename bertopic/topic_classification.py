# topic_classification.py
from bertopic import BERTopic

def classify_text(text):
    # Carregar modelo pré-treinado
    topic_model = BERTopic.load("models/bertopic_model")
    
    # Classificar o texto
    topics, probs = topic_model.transform([text])
    return topics[0]