from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch


# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER lexicon once
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the transformer model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # RoBERTa sentiment analysis
    encoded_input = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Convert numpy array to Python list (fix for TypeError)
    scores_list = scores.tolist()

    roberta_result = {
        'roberta_neg': scores_list[0],
        'roberta_neu': scores_list[1],
        'roberta_pos': scores_list[2]
    }

    return {**vader_result, **roberta_result}

def sentiment_to_stars(sentiment_score):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    if sentiment_score <= thresholds[0]:
        return 1
    elif sentiment_score <= thresholds[1]:
        return 2
    elif sentiment_score <= thresholds[2]:
        return 3
    elif sentiment_score <= thresholds[3]:
        return 4
    else:
        return 5

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sentiment_scores = analyze_sentiment(text)
    star_rating = sentiment_to_stars(sentiment_scores['roberta_pos'])

    # Convert float32 values to standard float
    sentiment_scores = {k: float(v) for k, v in sentiment_scores.items()}

    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }

    return jsonify(response)

# Health check endpoint
@app.route('/')
def health_check():
    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=5000)
