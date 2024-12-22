import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from openai import OpenAIError

st.set_page_config(page_title="Chat with the Hmone's personal assistant",  layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with Eaint Lay Hmone's Career Assistant")
st.markdown(
    """
    Welcome to my personal career assistant!  
    Here, you can ask questions about my career journey, educational background, key projects, and professional skills.  
    Whether you're a recruiter, collaborator, or just curious, feel free to ask me anything!
    """
)
# Prompt the user to enter their OpenAI API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()
#openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    # Set API key
    openai.api_key = api_key

    # Test the API by making a dummy call (list models)
    openai.Model.list()
except OpenAIError as e:
   st.error("The API key is invalid. Please enter a valid OpenAI API key.")
st.stop()
   


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "I am Hmone personal assistant.Ask me a question about Eaint Lay Hmone",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are my personal assistant and your job is 
        to answer my biography for recuriter. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
