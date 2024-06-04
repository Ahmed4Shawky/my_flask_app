from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize Flask app
app = Flask(__name__)

# Load the smaller DistilBERT sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

def analyze_sentiment(text):
    # DistilBERT sentiment analysis
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    distilbert_result = {
        'distilbert_neg': float(scores[0]),  # Convert numpy float32 to Python float
        'distilbert_pos': float(scores[1])   # Convert numpy float32 to Python float
    }

    return distilbert_result

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
    star_rating = sentiment_to_stars(sentiment_scores['distilbert_pos'])
    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
