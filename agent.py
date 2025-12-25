
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")



PDF_FOLDER_PATH = "data"
VECTORSTORE_PATH = "data/vector_index"
def get_policy_retriever():
    """
    Load or create FAISS vectorstore and return a retriever.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if index already exists
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        #print("[Policy Tool] Loading existing FAISS vectorstore...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("[Policy Tool] FAISS index not found. Building vectorstore from PDFs...")

        # Load all PDFs
        all_docs = []
        for filename in os.listdir(PDF_FOLDER_PATH):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["source_file"] = filename
                all_docs.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        final_documents = text_splitter.split_documents(all_docs)

        # Create vectorstore
        vectorstore = FAISS.from_documents(documents=final_documents, embedding=embedding_model)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("[Policy Tool] Vectorstore created and saved.")

    return vectorstore.as_retriever(k=4, score_threshold=0.5)

pdf_retriever = get_policy_retriever()

from typing  import  TypedDict, List, Optional
from langgraph.graph import StateGraph
#from langgraph.graph.message import add_messages
#from langchain_core.messages import  HumanMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel

import json

class PolicyResponse(BaseModel):
    answer: str
    action: str
    action_input: Optional[str]

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    action: str
    action_input: Optional[str]


llm = ChatOpenAI(model = "llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3, base_url='https://api.groq.com/openai/v1')
# did not use structured_output because groq based LLMs did not support Json formatting
llm_structured = llm.with_structured_output(PolicyResponse)

from langchain_core.prompts import PromptTemplate

prompt = """
You are a return, refund, and cancellation policy decision assistant.

Your role is to READ the provided policy context, DECIDE whether a policy applies to
the user's question, and RESPOND accordingly.

You must first decide:
- Does the policy explicitly cover this question?
- If yes, which specific policy category applies?
- If no, clearly state that the information is not available.

Decision Rules:
Rules:
1. Use ONLY the provided context. Do not use prior knowledge or assumptions.
2. Do NOT invent procedures, steps, timelines, fees, or contact details.
3. If the policy explicitly covers the question:
   - Provide a concise, factual answer grounded in the policy text.
   - Set action = "APPLY_POLICY".
   - Set action_input to EXACTLY ONE of the following values
     (do NOT create new labels under any circumstance):
       - return_window
       - brand_specific_return
       - defective_item
       - cancellation_policy
       - refund_timeline
4. Use this guidance when selecting action_input:
   - return_window → general return eligibility or time limits
   - brand_specific_return → rules applying only to specific brands (e.g., Apple)
   - defective_item → damaged, defective, or not-working items
   - cancellation_policy → order cancellation fees or cancellation rules
   - refund_timeline → time taken to receive refunds
5. If the policy does NOT explicitly cover the question:
   - Set answer = "This information is not available in the provided documents."
   - Set action = "NO_INFO".
   - Set action_input = out_of_scope.
6. Do NOT add examples, opinions, or inferred instructions.
7. Output VALID JSON only. Do not include any text outside the JSON object.

Output format:
{{
  "answer": "<string>",
  "action": "APPLY_POLICY" | "NO_INFO",
  "action_input": "<policy_category>"
}}

Context:
{context}

Question:
{question}
"""


custom_rag_prompt = PromptTemplate.from_template(prompt)

def retrieve(state: State):
    retrieved_docs = pdf_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    prompt_message = custom_rag_prompt.invoke({
        "question": state["question"],
        "context": docs_content
    })

    raw_response = llm.invoke(prompt_message).content

    output = {
        "answer": "This information is not available in the provided documents.",
        "action": "NO_INFO",
        "action_input": None
    }

    try:
        parsed = json.loads(raw_response)

        output["answer"] = parsed.get("answer", output["answer"])
        output["action"] = parsed.get("action", output["action"])
        output["action_input"] = parsed.get("action_input", output["action_input"])

    except Exception:
        pass

    return output


workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")

agent= workflow.compile()