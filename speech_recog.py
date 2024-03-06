import speech_recognition
import pyttsx3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from recommender_system import recommend, recommend_categories
import gensim.downloader as api
import json
import operator
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

nltk.download('stopwords')

# recognizer = speech_recognition.Recognizer()

# Stop words are the common high-frequency words that are not useful for analysis
stop_words = set(stopwords.words('english'))

#POS_TAGS are the type of tokens
#We have to pick the types which are useful for our analysis
impPostags = ['CD', 'NN', 'NNS', 'LS', 'NNP']
impWords = []

print("Importing the word vectors")
model = api.load('glove-wiki-gigaword-50')
word_vectors = model
print("Imported the word vectors")

def tokenizeFunc(text, data):
    print("Inside tokenizeFunc")
    print(text)
    tokens = word_tokenize(text)
    
    # Removing Stop Words
    filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]
    
    #POS Tagging
    posTags = pos_tag(filtered_tokens)
    print("POS Tags:")
    print(posTags)
    impWords = [item[0] for item in posTags if item[1] in impPostags]
    
    print("The important words are:")
    print(set(impWords))
    
    # print("Importing json data")
    # data = json.load(open('./recommenderData.json'))
    # print("Data loaded")
    
    indices = dict()
    
    for word in set(impWords):
        recommended_catergories_indices = recommend_categories(word=word, data=data, word_vectors=word_vectors)
        # print("Recommend category indices")
        # print(recommended_catergories_indices)
        for key, value in recommended_catergories_indices.items():
            # print(key,": ", value)
            try:
                if key not in indices:
                    indices[key] = value
                    print(indices)
                else:
                    indices[key] = max(indices[key], value[0])
            except:
                continue
    # print("Indices values after categories")
    # print(indices)
    
    for word in set(impWords):
        recommended_indices = recommend(word=word, data=data, word_vectors=word_vectors)
        # indices.add(index)
        for key, value in recommended_indices.items():
            try:
                print(key,": ", value[0])
                if key not in indices:
                    indices[key] = value[0]
                else:
                    indices[key] = max(indices[key], value[0])
            except:
                continue
    print("Indices values after recommendation")
    print(indices)
    
    indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=True))
    indices = {key: indices[key] for key in list(indices)[:3]}
    print("The final indices dict is as follows")
    print(indices)
    
    # print("The recommended products are:")
    # keys = list(data.keys())
    # for key, value in indices.items():
    #     for word in set(impWords):
    #         if word not in list(data[keys[key]]):
    #             data[keys[key]][word] = 1
    #         else:
    #             data[keys[key]][word] += 1
                
                
                
        
    # with open('./recommenderData.json', 'w') as file:
    #     json.dump(data, file, indent=2)
    
    return [list(indices.keys()), list(impWords)]

@app.route('/', methods = ['GET'])
@cross_origin()
def speechRecog():
    recognizer = speech_recognition.Recognizer()
    print("Welcome to speech recognition system")
    print("Please pause for a while to let the system process your words")
    print("To quit, just speak 'exit'")
    text = ""
    # indices = dict()
    # while text != "exit":
    try:
        recData = request.args.get('recData')
        data = json.loads(recData)
        # print(recData)
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise (mic, duration=0.2) 
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google (audio)
            text = text.lower()
            return tokenizeFunc(text, data=data)
        
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        return {}
        # continue
    # return json.dumps(indices)
    
if __name__ == "__main__":
    app.run()