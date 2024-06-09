import random
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatterbot import ChatBot

# downloaded NLTK data packages
nltk.download('punkt')
nltk.download('wordnet')

# Initialize chatbot
chatbot = ChatBot('Historical Chatbot')


# Preprocess the data
def preprocess(text):
    tokens = nltk.word_tokenize(str(text))
    return " ".join([word.lower() for word in tokens])


# Load and vectorize data
def vectorize_data(file_path):
    df = pd.read_csv(file_path)

    df['preprocessed'] = df.apply(
        lambda row: preprocess(row['Name of Incident']) + " " + preprocess(row['Impact']) + " " + preprocess(
            row['Important Person/Group Responsible']), axis=1)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['preprocessed'])

    return df, vectorizer, X


def get_response0(user_input, df, vectorizer, X):
    user_input = preprocess(user_input)
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, X)
    most_similar_index = similarities.argmax()

    event = df.iloc[most_similar_index]

    # Determine which information is requested
    if "date" in user_input:
        response = f"\nDate: {event['Date']} {event['Month']} {event['Year']}"
    elif "country" in user_input:
        response = f"\nCountry: {event['Country']}"
    elif "type" in user_input:
        response = f"\nType of Event: {event['Type of Event']}"
    elif "place" in user_input:
        response = f"\nPlace: {event['Place Name']}"
    elif "impact" in user_input:
        response = f"\nImpact: {event['Impact']}"
    elif "population" in user_input:
        response = f"\nAffected Population: {event['Affected Population']}"
    elif "person" in user_input or "group" in user_input:
        response = f"\nImportant Person/Group Responsible: {event['Important Person/Group Responsible']}"
    elif "outcome" in user_input:
        response = f"\nOutcome: {event['Outcome']}"
    elif "event" in user_input or "incident" in user_input:
        response = f"\nEvent: {event['Name of Incident']}"
    elif "info" in user_input or "information" in user_input:
        response = (
            f"Event: {event['Name of Incident']}\n"
            f"Date: {event['Date']} {event['Month']} {event['Year']}\n"
            f"Country: {event['Country']}\n"
            f"Type of Event: {event['Type of Event']}\n"
            f"Place: {event['Place Name']}\n"
            f"Impact: {event['Impact']}\n"
            f"Affected Population: {event['Affected Population']}\n"
            f"Important Person/Group Responsible: {event['Important Person/Group Responsible']}\n"
            f"Outcome: {event['Outcome']}"
        )
    else:
        # else response should be given from the chatbot
        response = str(chatbot.get_response(user_input))
    return response


# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "bye")
GREETING_RESPONSES = ["hi", "hey", "hello", "I am glad you are talking to me", "Bye! Have a great time!"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            if word.lower() == "bye":
                return "Bye! Have a great time!"
            else:
                return random.choice(GREETING_RESPONSES)


def main(user_input, df, vectorizer, X):
    user_input = user_input.lower()
    if user_input != 'bye':
        if user_input in ['thanks', 'thank you']:
            return "historical chatbot: That's my job! Any other questions?"
        else:
            if greeting(user_input) is not None:
                return "historical chatbot: " + greeting(user_input)
            else:
                return "historical chatbot:" + get_response0(user_input, df, vectorizer, X)
    else:
        return "historical chatbot: Bye! Have a great time!"


if __name__ == "__main__":
    print("Hello, there! I am the historical chatbot. I will answer your queries. If you want to exit, type 'Bye!'")

    # Load your dataset
    dataset_path = r'C:\Users\reyam\PycharmProjects\pythonProject13\World Important Dates.csv'
    df, vectorizer, X = vectorize_data(dataset_path)

    while True:
        user_input = input("You: ")
        response = main(user_input, df, vectorizer, X)
        print(response)
        if user_input.lower() == 'bye':
            break
