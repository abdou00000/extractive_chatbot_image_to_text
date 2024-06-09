from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

# Create a new instance of ChatBot
chatbot = ChatBot('MyChatBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on English language corpus data
trainer.train('chatterbot.corpus.english')


# Example interaction
while True:
    user_input = input("You: ")
    response = chatbot.get_response(user_input)
    print("Bot:", response)