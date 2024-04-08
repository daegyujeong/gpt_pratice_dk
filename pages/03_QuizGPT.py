import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        print("Parsing!!!")
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    The difficulty of questions is {difficulty}

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    
    Any of answers can not be duplicated to other questions which means, once the anwser has been used from one of qeustions,
    the answer can not be used from the other qeustions.


    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(topic, difficulty_level):
    print(difficulty_level)
    doc = format_docs(docs)
    # print(doc)
    questions_prompt = template.format_messages(
        difficulty=difficulty_level,
        context=doc,
    )
    # Use the provided API key for the ChatOpenAI instance
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )    
    # print(questions_prompt)
    questions_chain = llm.invoke(questions_prompt)
    print(questions_chain)

    formatting_chain = formatting_prompt | llm | output_parser
    test = formatting_chain.invoke({"context":questions_chain})
    # formatting_chain = formatting_prompt.format_messages(context=questions_chain)
    # test = llm.invoke(formatting_chain)
    # questions_chain = questions_prompt | llm 
    # print(questions_prompt)

    # formatting_prompt = formatting_prompt.format_messages(
    #     context=questions_chain,
    # )    
    # print(formatting_prompt)
    # formatting_chain = formatting_prompt | llm
    # test=formatting_chain.invoke({"context":questions_chain})
    print(test)   
    # parsed_test = output_parser.parse(text=test)
    # print(parsed_test)
    # test = test | output_parser
    return test

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

# Sidebar for API Key and GitHub Repo link
with st.sidebar:
    api_key = st.text_input("Enter your OpenAI API Key")
    difficulty_level = st.selectbox("Select Difficulty", ["Very Easy","Easy","Normal","Hard","Hell"])
    topic = st.text_input("Search Wikipedia...")
    if topic:
        docs = wiki_search(topic)
    # print(api_key)
    # print(difficulty)



def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



if st.sidebar.button("Generate Quiz"):    
    if topic:
        count = 0
        response = run_quiz_chain(topic,difficulty_level)
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index= None,
                    key = count,
                )
                count = count+1
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()

    if not topic:
            st.markdown(
                """
                Please Type Valid Input
            """
            )     
if not topic:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles to test your knowledge and help you study.
                
    Get started by searching on Wikipedia in the sidebar.
    """
    )        