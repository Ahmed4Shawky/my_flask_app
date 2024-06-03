from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load NLTK's VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the transformer model and tokenizer (e.g., RoBERTa)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    return tokenizer, model

def analyze_sentiment(texts, tokenizer, model):
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
        
        tokenizer, model = load_model_and_tokenizer()
        results = analyze_sentiment(texts, tokenizer, model)
        
        star_ratings = [sentiment_to_stars(result['roberta_result']['roberta_pos']) for result in results]
        response = {
            'results': results,
            'star_ratings': star_ratings
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
