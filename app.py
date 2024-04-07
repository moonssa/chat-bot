import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, output_parser

@st.cache_data(show_spinner="퀴즈 생성중...")
def run_quiz_chain(*, title, count,  difficulty):
    chain = prompt | llm
    return chain.invoke(
        {
            "quiz_title": title,
            "quiz_count": count,
            "quiz_difficulty": difficulty,
        }
    )


st.set_page_config(
    page_title="QuizGPT| 챌린지",
    page_icon="🎴",
)

st.title(" QuizGPT ")
with st.expander("과제 내용 보기", expanded=True):
  st.markdown(
        """
  QuizGPT를 구현하되 다음 기능을 추가합니다:

  - :white_check_mark: 함수 호출을 사용합니다.
  - :white_check_mark: 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
  - :white_check_mark: 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
  - :white_check_mark: 만점이면 st.ballons를 사용합니다.
  - :white_check_mark: 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
  - :white_check_mark: st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
    """
  )

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)
    
output_parser = JsonOutputParser()

api_key = st.sidebar.text_input("API KEY", type="password")
st.session_state["api_key"] = api_key
os.environ['OPENAI_API_KEY'] = api_key

with st.sidebar:

  st.divider()
  st.markdown(
  """
    Github Link
    
    https://github.com/moonssa/chat-bot/commit/039b215d177a5e6c8ba9510c8f929b7a94556d18
"""
)

function = {
  "name": "create_quiz", 
  "description":"function that takes list of questions and answers and returns a quiz",
  "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },

        "required":["questions"],
},
}




prompt = PromptTemplate.from_template("""Make a quiz about {quiz_title}
                                      
            Number of Questions: {quiz_count}
            Difficulty Level: Level-{quiz_difficulty}/10""")

st.session_state["quiz_title"]=None

if not api_key:
  st.error("⚠️  Please Enter Your :red[OpenAI API Key]")
else:
    llm = ChatOpenAI(temperature=0.1).bind(
    function_call = {
    "name": "create_quiz"
    },
    functions=[
    function
    ]
    )

    with st.form("settings_form"):
        quiz_title = st.text_input("퀴즈 제목")
        st.session_state["quiz_title"] = quiz_title
        quiz_count=st.slider("문항수를 선택하세요", 1,10,5)
        st.session_state["quiz_count"] = quiz_count
        quiz_difficulty=st.slider("난이도를 선택하세요", 1,10,3)
        st.session_state["quiz_difficulty"] = quiz_difficulty
        button = st.form_submit_button("퀴즈 만들기", use_container_width=True)

    if(st.session_state["quiz_title"]):
        try:
            response = run_quiz_chain(
                title = st.session_state["quiz_title"],
                count =st.session_state["quiz_count"],
                difficulty = st.session_state["quiz_difficulty"]
            )
        
  
            response = response.additional_kwargs["function_call"]["arguments"]
            response = json.loads(response)
            # st.write(response["questions"])

            with st.form("questions_form"):
                correct=0
                for idx, question in enumerate(response["questions"]):
                    st.write(f"{idx+1}.  ", question["question"])
                    value = st.radio(
                        f"Select an option",
                        [answer["answer"] for answer in question["answers"]],
                        key=f"{idx}_radio",
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        correct +=1
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong")
                st.form_submit_button("**:blue[제출하기]**", use_container_width=True)

                st.write(correct)
                if correct ==st.session_state["quiz_count"]:
                    st.balloons()
        except Exception as e:
            st.error("Please Check API_KEY ")

  
