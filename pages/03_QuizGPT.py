import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


# Sidebar for API Key and GitHub Repo link
with st.sidebar:
    api_key = st.text_input("Enter your OpenAI API Key")
    difficulty = st.selectbox("Select Difficulty", ["Very Easy","Easy","Normal","Hard","Hell"])
    topic = st.text_input("Search Wikipedia...")
    # print(api_key)
    # print(difficulty)

# Use the provided API key for the ChatOpenAI instance
llm = ChatOpenAI(
    # api_key=api_key,
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)




# difficulty = get_difficulty_level()



prompt = PromptTemplate.from_template(
    """            
    Please create a quiz based on the following criteria:

    Topic: {subject}
    Number of Questions: 10
    Difficulty Level: Level-{difficulty}

    The quiz should be well-structured with clear questions and correct answers.
    Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
    The quiz format should be multiple-choice,
    and each question should be accompanied by four possible answers, with only one correct option.
    """,
)

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(topic, difficulty):
    print(api_key)
    print(difficulty)
    chain = prompt | llm
    return chain.invoke(
        {
            "subject": topic,
            "difficulty": difficulty,
        }
    )



if not topic:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(topic,difficulty)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()