from dotenv import load_dotenv
from openai import OpenAI
import json
import gradio as gr

from chatbot.src.vector_store import load_vector_store, retrieve_similar_documents
from chatbot.src.agent_tools import AgentTools


load_dotenv(override=True)

class Me:
    def __init__(self):
        self.name = "Saira Rizvi"
        self.openai_client = OpenAI()
        self.agent_tools = AgentTools()

        self.vectors_store = load_vector_store(persist_directory="./chroma_store")
        self.documents = ""
        with open("profile_data/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
        particularly questions related to {self.name}'s career, background, skills and experience. \
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
        You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## Relevant Documents:\n{self.documents}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = getattr(self.agent_tools, tool_name, None)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def retrieve_documents(self, query, k=3):
        similar_docs = retrieve_similar_documents(self.vectors_store, query, k)
        self.documents = "\n".join([doc.page_content for doc in similar_docs])
        return
    
    def chat(self, message, history):
        self.retrieve_documents(message)
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=self.agent_tools.tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
