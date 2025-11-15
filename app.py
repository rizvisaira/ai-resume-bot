import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
from chatbot.main import Me
import gradio as gr

if __name__ == "__main__":
    me = Me()
    interface = gr.ChatInterface(me.chat, type="messages")
    interface.launch()