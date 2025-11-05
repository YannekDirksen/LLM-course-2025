from fasthtml.common import *
import google.generativeai as genai
import strip_markdown
import configparser
from dotenv import load_dotenv
import os
from pathlib import Path



load_dotenv()
API_KEY = os.environ.get("helsinki")
genai.configure(api_key=API_KEY)
LLM = "gemini-2.5-flash"
model = genai.GenerativeModel(LLM)

# --- Read system prompts from config file safely ---
from pathlib import Path

prompts = configparser.ConfigParser()

# Get the absolute path of prompts.env (same folder as this script)
cfg_path = Path(__file__).resolve().parent / "prompts.env"

# Try to read it and show a helpful message if not found
read_files = prompts.read(cfg_path)
if not read_files:
    raise FileNotFoundError(f"⚠️ Could not find prompts.env at: {cfg_path}")

# Load values safely with fallbacks
topic = prompts.get("TEMPLATES", "TOPIC", fallback="the provided text")
number = prompts.get("TEMPLATES", "NUMBER", fallback="5")

# Set system prompt
# system_prompt = prompts.get("SYSTEM_PROMPTS", "IT_HELPDESK")
system_prompt = f"Summarize the following text about {topic} in {number} bullet points:"


# Set up the app, including daisyui and tailwind for the chat component
hdrs = (picolink, Script(src="https://cdn.tailwindcss.com"),
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"))
app = FastHTML(hdrs=hdrs, cls="p-4 max-w-lg mx-auto")

# Chat message component (renders a chat bubble)
def ChatMessage(msg, user):
    bubble_class = "chat-bubble-primary" if user else 'chat-bubble-secondary'
    chat_class = "chat-end" if user else 'chat-start'
    return Div(cls=f"chat {chat_class}")(
               Div('user' if user else 'assistant', cls="chat-header"),
               Div(msg, cls=f"chat-bubble {bubble_class}"),
               Hidden(msg, name="messages")
           )

# The input field for the user message.
def ChatInput():
    return Input(name='msg', id='msg-input', placeholder="Type a message",
                 cls="input input-bordered w-full", hx_swap_oob='true')

# The main screen
@app.get
def index():
    page = Form(hx_post=send, hx_target="#chatlist", hx_swap="beforeend")(
           Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
               Div(cls="flex space-x-2 mt-2")(
                   Group(ChatInput(), Button("Send", cls="btn btn-primary"))
               )
           )
    return Titled('Simple chatbot demo', page)

# Handle the form submission
@app.post
def send(msg:str, messages:list[str]=None):
    if not messages: messages = [system_prompt]
    messages.append(msg.rstrip())
    r = model.generate_content(messages).text
    return (ChatMessage(msg, True),    # The user's message
            ChatMessage(strip_markdown.strip_markdown(r.rstrip()), False), # The chatbot's response
            ChatInput()) # And clear the input field

serve()

