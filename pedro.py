import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load the model and tokenizer
def load_model():
    global chatbot_pipeline, conversation_history
    model_name = "google/flan-t5-base"  # Use "flan-t5-large" if hardware allows
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    chatbot_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    conversation_history = ""  # Initialize conversation history

# Function to handle user input and generate bot responses
def send_message():
    global conversation_history

    # Get user input from the entry box
    user_input = user_entry.get("1.0", tk.END).strip()
    if not user_input:
        return

    # Add the user input to the conversation history
    conversation_history += f"User: {user_input}\n"

    # Display user input in the chat window
    chat_window.insert(tk.END, f"You: {user_input}\n", "user")
    user_entry.delete("1.0", tk.END)

    # Add a guiding prompt
    inputs = f"The following is a friendly and helpful conversation between a user and PedroLopesGPT.\n{conversation_history}Bot:"
    response = chatbot_pipeline(inputs, max_length=200, num_return_sequences=1, truncation=True)[0]["generated_text"]

    # Extract the bot's new response
    bot_response = response.strip()
    conversation_history += f"Bot: {bot_response}\n"

    # Display the bot's response in the chat window
    chat_window.insert(tk.END, f"PedroLopesGPT: {bot_response}\n", "bot")
    chat_window.see(tk.END)  # Auto-scroll to the latest message

# Load the model at startup
load_model()

# Create the main application window
root = tk.Tk()
root.title("PedroLopesGPT Chatbot")

# Configure window size
root.geometry("600x600")

# Chat window (scrollable text area)
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Add text styling for user and bot
chat_window.tag_config("user", foreground="blue", font=("Arial", 12, "bold"))
chat_window.tag_config("bot", foreground="green", font=("Arial", 12))

# User input box
user_entry = tk.Text(root, height=3, font=("Arial", 12))
user_entry.pack(padx=10, pady=(0, 10), fill=tk.X)

# Send button
send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12), bg="lightblue")
send_button.pack(pady=(0, 10))

# Add Enter key binding
root.bind('<Return>', lambda event: send_message())

# Run the application
root.mainloop()
