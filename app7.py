from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from model_loader import get_DistilBERT_analyzer

# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER
nltk.download('vader_lexicon', download_dir='/tmp')
sia = SentimentIntensityAnalyzer()

# Get the pre-loaded DistilBERT analyzer
DistilBERT_analyzer = get_DistilBERT_analyzer()

def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # DistilBERT sentiment analysis
    DistilBERT_result = DistilBERT_analyzer(text)
    DistilBERT_score = DistilBERT_result[0]['score']

    return {
        **vader_result,
        'DistilBERT_score': DistilBERT_score
    }

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
    star_rating = sentiment_to_stars(sentiment_scores['DistilBERT_score'])
    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
