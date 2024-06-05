from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import wandb
import os

# Set the W&B API key
os.environ['WANDB_API_KEY'] = 'f2ed7e9fc4402d58c2eea85e50256b7f8f781048'

# Initialize Flask app
app = Flask(__name__)

# Initialize W&B
wandb.init(
    project="flask",  # Ensure this matches your project name on W&B
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the transformer model and tokenizer (e.g., RoBERTa)
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')


def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # RoBERTa sentiment analysis
    encoded_input = tokenizer(text, return_tensors='pt')
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
    sentiment_scores = {
        'compound': float(sentiment_scores['compound']),
        'neg': float(sentiment_scores['neg']),
        'neu': float(sentiment_scores['neu']),
        'pos': float(sentiment_scores['pos']),
        'roberta_neg': float(sentiment_scores['roberta_neg']),
        'roberta_neu': float(sentiment_scores['roberta_neu']),
        'roberta_pos': float(sentiment_scores['roberta_pos'])
    }

    # Log sentiment scores to W&B
    wandb.log({
        "compound": sentiment_scores['compound'],
        "neg": sentiment_scores['neg'],
        "neu": sentiment_scores['neu'],
        "pos": sentiment_scores['pos'],
        "roberta_neg": sentiment_scores['roberta_neg'],
        "roberta_neu": sentiment_scores['roberta_neu'],
        "roberta_pos": sentiment_scores['roberta_pos'],
        "star_rating": star_rating,
        "text": text
    })

    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
