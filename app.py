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
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Pre-load the model and tokenizer (for better memory efficiency)
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
model = AutoModelForSequenceClassification.from_pretrained(tokenizer.model_name_or_path)
logging.info("Model and tokenizer downloaded successfully")


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze sentiment of a given text using both NLTK's VADER and Hugging Face model.

    Returns:
        JSON: Dictionary containing sentiment scores from both models.
    """
    if request.method == 'POST':
        try:
            data = request.json
            text = data.get('text')  # Get text from request body

            if text:
                # Clean and pre-process text
                cleaned_text = clean_text(text)

                # Analyze with NLTK's VADER
                vader_scores = sia.polarity_scores(cleaned_text)

                # Analyze with Hugging Face model
                inputs = tokenizer(cleaned_text, return_tensors="pt")  # Convert text to model input
                outputs = model(**inputs)
                predictions = softmax(outputs.logits.detach().cpu().numpy()[0])
                sentiment = predictions.argmax()  # Get the most likely sentiment class

                # Combine and return results
                results = {
                    "vader": vader_scores,
                    "transformer": {
                        "sentiment": sentiment,
                        "scores": predictions.tolist()  # List of all sentiment class probabilities
                    }
                }
                return jsonify(results), 200
            else:
                return jsonify({'error': 'Missing text in request body'}), 400
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Unsupported method'}), 405


def clean_text(text):
    # Implement your text cleaning logic here
    # For example, remove special characters, stop words, etc.
    cleaned_text = text.strip()
    return cleaned_text


if __name__ == '__main__':
    # Run the app (Render will handle port assignment)
    app.run(host='0.0.0.0')
