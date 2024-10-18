# import getpass
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI


# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     # google_api_key = GOOGLE_API_KEY,
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
# )


# messages = [
#     (
#         "system",
#         "you are a helpful chatbot for basic science questions",
#     ),
#     ("human", "what is plant? "),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)

import langchain_core.messages.tool
print(dir(langchain_core.messages.tool))
