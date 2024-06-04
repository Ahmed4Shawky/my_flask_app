from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize Flask app
app = Flask(__name__)

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

  # Log the sentiment scores and star rating (optional)
  # import logging
  # logger = logging.getLogger(__name__)
  # logger.info("Sentiment scores: %s", sentiment_scores)
  # logger.info("Star rating: %s", star_rating)

  response = {
      'sentiment_scores': sentiment_scores,
      'star_rating': star_rating
  }

  # Log the complete response before returning it (optional)
  # logger.info("Complete response: %s", response)

  return jsonify(response)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
