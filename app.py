import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the transformer model and tokenizer (e.g., RoBERTa)
logging.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
logging.info("Model and tokenizer loaded successfully")

def analyze_sentiment(texts):
    # VADER sentiment analysis
    vader_results = [sia.polarity_scores(text) for text in texts]

    # RoBERTa sentiment analysis
    encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**encoded_inputs)
    scores = outputs.logits.detach().numpy()
    scores = softmax(scores, axis=1)
    roberta_results = [{
        'roberta_neg': score[0],
        'roberta_neu': score[1],
        'roberta_pos': score[2]
    } for score in scores]

    return [{'vader_result': vader_result, 'roberta_result': roberta_result} for vader_result, roberta_result in zip(vader_results, roberta_results)]

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
    try:
        data = request.json
        texts = data['texts']
        
        results = analyze_sentiment(texts)
        
        star_ratings = [sentiment_to_stars(result['roberta_result']['roberta_pos']) for result in results]
        response = {
            'results': results,
            'star_ratings': star_ratings
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
