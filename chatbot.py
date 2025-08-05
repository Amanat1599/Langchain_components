from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()

llm_local = HuggingFacePipeline.from_model_id(  
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  
            task="text-generation",  
            )
model = ChatHuggingFace(llm=llm_local)


while True:
    user_input = input("You : ")

    if user_input == 'exit':
        break

    out = model.invoke(user_input)
    print(f"AI : {out.content}")
