import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import logging
import threading

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NLTK's VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Pre-load the smallest model (flaubert-base-uncased)
tokenizer = AutoTokenizer.from_pretrained('flaubert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(tokenizer.model_name_or_path)
logging.info("Model and tokenizer downloaded successfully")


def analyze_sentiment(texts, max_len=32):  # Experiment with a smaller max_len
  # VADER sentiment analysis
  vader_results = [sia.polarity_scores(text) for text in texts]

  # Batch processing for memory efficiency (consider smaller batch sizes)
  batch_size = 4  # Experiment with smaller batch sizes
  encoded_inputs = []
  for i in range(0, len(texts), batch_size):
      batch_texts = texts[i:i+batch_size]
      batch_encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
      encoded_inputs.append(batch_encoded)
  outputs = model(**torch.cat(encoded_inputs, dim=0))  # Concatenate batches
  scores = outputs.logits.detach().numpy()
  scores = softmax(scores, axis=1)
  roberta_results = []
  for i, score in enumerate(scores):
      roberta_results.append({
          'roberta_neg': score[0],
          'roberta_neu': score[1],
          'roberta_pos': score[2]
      })

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

# Hard-code the port value
port = 8000
logging.info(f"Starting app on port {port}")

if __name__ == '__main__':
  # Run the app
  app.run(host='0.0.0.0', port=port)
