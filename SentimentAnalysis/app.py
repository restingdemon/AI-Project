from flask import Flask, render_template, request
from collections import Counter
import string
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

def process_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]
    
    return final_words

def analyze_emotions(final_words):
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in final_words:
                emotion_list.append(emotion)

    return Counter(emotion_list)

def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative Sentiment"
    elif pos > neg:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        text = request.form['text']
        final_words = process_text(text)
        emotion_counter = analyze_emotions(final_words)
        sentiment = sentiment_analyse(text)
        return render_template('result.html', emotion_counter=emotion_counter, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '_main_':
    app.run(debug=True)