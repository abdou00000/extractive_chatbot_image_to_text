import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from chatterbot import ChatBot
from PIL import Image
from pytesseract import pytesseract
import enum
import re

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained chatbot model from the database
chatbot = ChatBot(
    'Historical Chatbot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.db'
)


# Preprocess the data
def preprocess(text):
    tokens = nltk.word_tokenize(str(text))
    return " ".join([word.lower() for word in tokens])


# Load and vectorize data
@st.cache
def vectorize_data(file):
    df = pd.read_csv(file)

    # Preprocess the data
    df['preprocessed'] = df.apply(
        lambda row: preprocess(row['Name of Incident']) + " " + preprocess(row['Impact']) + " " + preprocess(
            row['Important Person/Group Responsible']), axis=1)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['preprocessed'])

    return df, vectorizer, X


def get_response0(user_input, df, vectorizer, X):
    user_input = preprocess(user_input)
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, X)
    most_similar_index = similarities.argmax()

    event = df.iloc[most_similar_index]

    # Determine which specific information is being requested
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


# Image to text extraction class and function
class Language(enum.Enum):
    ENG = 'eng'
    FRA = 'fra'


class ImageReader:

    def __init__(self):
        self.windows_path = r'C:\Users\reyam\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        pytesseract.tesseract_cmd = self.windows_path

    def extract_text(self, image: Image, lang: [Language]) -> str:
        lang_str = '+'.join([language.value for language in lang])
        extracted_text = pytesseract.image_to_string(image, lang=lang_str)
        return extracted_text


# Streamlit app
def main():
    st.title("Historical Chatbot")
    st.write("Hello, there! I am the historical chatbot. I will answer your queries. If you want to exit, type 'Bye!'")

    dataset_path = r'C:\Users\reyam\PycharmProjects\pythonProject13\World Important Dates.csv'  # Replace with the actual path to your dataset CSV file
    df, vectorizer, X = vectorize_data(dataset_path)

    # Input handling
    input_type = st.radio("Choose input type:", ("Text", "Image"))

    user_input = ""
    if input_type == "Text":
        user_input = st.text_input("You: ", "")
    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            ir = ImageReader()
            extracted_text = ir.extract_text(image, lang=[Language.ENG, Language.FRA])
            st.write(f"Extracted Text: {extracted_text}")

            # Split the extracted text into individual questions
            questions = re.split(r'\?+', extracted_text)

            # Process each question and generate responses
            st.write("Responses:")
            for i, question in enumerate(questions):
                question = question.strip()
                if question:
                    response = get_response0(question + '?', df, vectorizer, X)
                    st.write(f"Question {i + 1}: {question}?")
                    st.write(f"Response {i + 1}: {response}")

    if user_input:
        user_input = user_input.lower()
        if user_input != 'bye':
            if user_input in ['thanks', 'thank you']:
                st.write("Historical chatbot: That's my job! Any other questions?")
            else:
                if greeting(user_input) is not None:
                    st.write("Historical chatbot: " + greeting(user_input))
                else:
                    st.write("Historical chatbot:\n" + get_response0(user_input, df, vectorizer, X))
        else:
            st.write("Historical chatbot: Bye! Have a great time!")


if __name__ == "__main__":
    main()
