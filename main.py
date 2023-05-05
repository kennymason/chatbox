# hello_world.py

import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.document_loaders import ObsidianLoader
from langchain.indexes import VectorstoreIndexCreator
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from requests import Session

os.environ['OPENAI_API_KEY'] = ""
OBSIDIAN_PATH = ""

'''
A chatbot with:
    ChatGPT agent with extra tooling
    Code completion agent
    Obsidian Lookup
'''

name = "user"
server = Session()

# Mode
isChat = True
isCode = False
isObs = False
# Chat logs per mode
chatMsgs = []
codeMsgs = []
obsMsgs = []

# Create llm agent
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
tools = load_tools(
    ["llm-math"], 
    llm=llm
)
# initialize conversational memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
agent = initialize_agent(
    agent='chat-conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)

# Load Obsidian Vault
loader = ObsidianLoader(OBSIDIAN_PATH)
docs = loader.load()
# Create Vectorstore index
index = VectorstoreIndexCreator().from_loaders([loader])

# GUI:
app = QApplication([])

text_area = QPlainTextEdit()
text_area.setReadOnly(True)
text_area.setStyleSheet('background-color: #444f54; color: white; border-radius: 4px')
# text_area.setFocusPolicy(Qt.NoFocus)
message = QLineEdit()
message.setStyleSheet('background-color: #444f54; color: white; border-radius: 4px')

layout = QVBoxLayout()

button_layout = QHBoxLayout()  # Create a horizontal layout for the buttons
chatButton = QPushButton('Chat')
chatButton.setStyleSheet('QPushButton:focus { outline: none; }')
button_layout.addWidget(chatButton)
codeButton = QPushButton('Code')
codeButton.setStyleSheet('QPushButton:focus { outline: none; }')
# button_layout.addWidget(codeButton)
obsButton = QPushButton('Obsidian')
obsButton.setStyleSheet('QPushButton:focus { outline: none; }')
button_layout.addWidget(obsButton)
layout.addLayout(button_layout)

layout.addWidget(text_area)
layout.addWidget(message)
window = QWidget()
window.setLayout(layout)

window.setGeometry(100, 100, 600, 500)
window.show()

# Query Agents
def queryChat():
    return agent(message.text())

def queryObs():
    return index.query_with_sources(message.text())

# Event handlers:
def send_message():
    text_area.appendPlainText(f"user:\n {message.text()}\n")

    answer = ""

    if isChat:
        res = queryChat()
        answer = res['output']
        chatMsgs.append({ 'user': message.text(), 'asst': answer})
    if isCode:
        res = queryCode()
        answer = res['output']
        codeMsgs.append({ 'user': message.text(), 'asst': answer})
    if isObs:
        res = queryObs()
        answer = res['answer']
        obsMsgs.append({ 'user': message.text(), 'asst': answer})

    if res:
        text_area.appendPlainText(f"assistant:\n {answer}\n")
    message.clear()

# Switch Modes
def activateChat():
    global isChat
    global isCode
    global isObs
    isChat = True
    isCode = False
    isObs = False

    chatHistory = []

    for msg in chatMsgs:
        chatHistory.append(f"user:\n {msg['user']}")
        chatHistory.append(f"assistant:\n{msg['asst']}")

    text_area.setPlainText('\n'.join(chatHistory))

def activateCode():
    global isChat
    global isCode
    global isObs
    isChat = False
    isCode = True
    isObs = False

    chatHistory = []

    for msg in codeMsgs:
        chatHistory.append(f"user:\n {msg['user']}")
        chatHistory.append(f"assistant:\n{msg['asst']}")

    text_area.setPlainText('\n'.join(chatHistory))

def activateObs():
    global isChat
    global isCode
    global isObs
    isChat = False
    isCode = False
    isObs = True

    chatHistory = []

    for msg in obsMsgs:
        chatHistory.append(f"user:\n {msg['user']}")
        chatHistory.append(f"assistant:\n{msg['asst']}")

    text_area.setPlainText('\n'.join(chatHistory))

# Signals:
message.returnPressed.connect(send_message)
chatButton.clicked.connect(activateChat)
codeButton.clicked.connect(activateCode)
obsButton.clicked.connect(activateObs)

# set sys prompt for agent
def initAgent():
    sys_msg = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself

    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    """
    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = new_prompt
    
initAgent()
app.exec()
