# Extractive Chatbot with Image to Text Conversion

## Introduction

This project aims to create an intelligent chatbot capable of answering historical questions through both text and image inputs. The chatbot leverages natural language processing (NLP) techniques to interpret questions and provide accurate responses. It also includes an image-to-text feature that allows users to upload images containing questions, which the chatbot then processes and answers. The chatbot is trained on an enhanced dataset that includes information about Moroccan and French history (I added some data to the history.yml file in the English corpus of my chatbot; it was not included in the repository), making it well-versed in these areas. Additionally, the chatbot uses a CSV file containing historical events to provide precise answers when specific keywords are detected.

## Project Structure

The project is divided into several key components:

### 1. Image to Text Conversion

This part of the project involves converting text from uploaded images into a format that the chatbot can process. It uses the Tesseract OCR (Optical Character Recognition) engine to extract text from images. The extracted text is then cleaned and prepared for further processing.


### 2. Training the Chatbot

The chatbot is built using the ChatterBot library and is trained on the English corpus. The corpus has been modified to include specific information about Moroccan and French history, providing the chatbot with a rich knowledge base in these areas. Additionally, a CSV file named 'World Important Dates.csv' containing historical events is used to enhance the chatbot's responses.


### 3. Utilizing the CSV Dataset

The chatbot uses the 'World Important Dates.csv' file to provide accurate historical information based on specific keywords in user queries. When a user asks a question that includes keywords such as "date," "country," "type," or "impact," the chatbot searches the CSV dataset to give a precise response from the data, ensuring accuracy and relevance.

![image](https://github.com/abdou00000/extractive_chatbot_image_to_text/assets/162928474/96e88004-86cf-4df9-9e9b-92126cee5db9)

### 4. Integrating the Chatbot

The integration process combines the image-to-text conversion functionality with the trained chatbot. This allows the chatbot to handle both text inputs and image-based queries seamlessly. The integration ensures that the chatbot can provide accurate and relevant historical information regardless of the input method.

![image](https://github.com/abdou00000/extractive_chatbot_image_to_text/assets/162928474/12e731eb-b4e6-4719-b3ae-79c771169c8b)

![image](https://github.com/abdou00000/extractive_chatbot_image_to_text/assets/162928474/8058da80-07cf-4ab9-8b08-f32b8affab3e)


### 5. Deploying with Streamlit

The project is deployed using Streamlit, an open-source app framework that allows for the easy creation of custom web applications. Streamlit provides an interactive interface where users can input text queries or upload images, and receive responses from the chatbot in real time.

## Conclusion

This project demonstrates the integration of multiple NLP and machine learning techniques to create a versatile and intelligent chatbot. By combining text and image inputs, the chatbot can handle a wide range of user queries, making it a powerful tool for historical information retrieval. The use of a CSV dataset ensures precise and accurate responses to specific historical questions. The deployment with Streamlit ensures an accessible and user-friendly experience, bringing the capabilities of advanced NLP directly to users' fingertips.

### N.B:
To test each part of the code separately, there is a folder called 'Examples', and there are also some test images in the folder called 'test_images'.
To run the project, first run the `part1.py` script, then in your terminal run the following command:
```sh
streamlit run part2.py
```
