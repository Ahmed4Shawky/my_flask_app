from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Load the BERT sentiment analysis model once
bert_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Perform sentiment analysis using VADER
    vader_sentiment = vader_analyzer.polarity_scores(text)
    vader_result = {
        'neg': vader_sentiment['neg'],
        'neu': vader_sentiment['neu'],
        'pos': vader_sentiment['pos'],
        'compound': vader_sentiment['compound']
    }

    # Perform sentiment analysis using BERT
    bert_sentiment = bert_analyzer(text)
    bert_result = {
        'label': bert_sentiment[0]['label'],
        'score': bert_sentiment[0]['score']
    }

    # Combine results
    result = {
        'vader': vader_result,
        'bert': bert_result
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
