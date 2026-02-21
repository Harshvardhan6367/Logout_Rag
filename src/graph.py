from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from src.vector_store import VectorStoreManager
from src.memory import MemoryManager
from src.utils import setup_logger, remove_stopwords
from langchain_openai import ChatOpenAI

logger = setup_logger(__name__)

class GraphState(TypedDict):
    question: str
    prescription_id: Optional[str] # None for Global, ID for Local
    session_id: str
    language: str # New field
    # history_summary: str # Removed
    context: List[str]
    answer: str
    rules_output: str # New field
    final_decision: str # New field

class RAGGraph:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.memory = MemoryManager()
        from src.otc_manager import OTCManager
        self.otc_manager = OTCManager()
        self.llm = ChatGoogleGenerativeAI(model=Config.GEMINI_MODEL_NAME, google_api_key=Config.GOOGLE_API_KEY)
        self.openai_llm = ChatOpenAI(model=Config.OPENAI_MODEL_NAME, openai_api_key=Config.OPENAI_API_KEY)

    def retrieve(self, state: GraphState):
        """
        Retrieve relevant chunks from Pinecone.
        """
        logger.info("Node: Retrieve")
        question = state["question"]
        prescription_id = state.get("prescription_id")
        
        # Search Pinecone
        results = self.vector_store.search(question, prescription_id=prescription_id)
        
        # Extract text from results
        context = [match.metadata["text"] for match in results]
        
        return {"context": context}

    def generate(self, state: GraphState):
        """
        Generate answer using Gemini.
        """
        logger.info("Node: Generate")
        question = state["question"]
        context = state["context"]
        language = state.get("language", "English") # Default to English
        # summary = state["history_summary"] # Removed
        
        context_str = "\n\n".join(context)
        
        # Fetch History
        history = self.memory.get_history(state["session_id"], limit=5)
        # Apply stop word removal 
        history_str = "\n".join([f"{msg['role'].capitalize()}: {remove_stopwords(msg['content'])}" for msg in history])
        
        prompt = f"""
        You are a helpful medical assistant. Answer the user's question based on the provided context and chat history.
        
        IMPORTANT INSTRUCTIONS:
        1. Answer in the following language: {language}
        2. If the user asks about a medicine ("What is this for?"), provide TWO things:
           a) The specific instructions from the prescription (dosage, timing).
           b) General medical knowledge about what the medicine is commonly used for (e.g., "Paracetamol is commonly used for fever and pain relief").
        
        Context from Prescriptions:
        {context_str}
        
        Chat History:
        {history_str}
        
        User Question: {question}
        
        Answer:
        """
        
        response = self.llm.invoke(prompt)
        # Add to memory manually here since we removed the summarize node
        self.memory.add_message(state["session_id"], "user", question)
        self.memory.add_message(state["session_id"], "ai", response.content)
        
        return {"answer": response.content}

    def rule_engine(self, state: GraphState):
        """
        Check for medical rules and safety flags.
        """
        logger.info("Node: Rule Engine")
        context = state["context"]
        answer = state["answer"]
        
        # Real Rule: Check medicines in context against OTC list
        rules_flagged = []
        
        # We need to extract medicine names from context to check them
        # For simplicity, we'll check the whole context strings
        otc_results = self.otc_manager.check_medicines_with_llm(context)
        
        consult_meds = otc_results.get("consult_medicines", [])
        if consult_meds:
            for med in consult_meds:
                rules_flagged.append(f"Safety Concern: {med['name']} needs doctor consultation. Reason: {med['reason']}")
            
        if "warning" in answer.lower() or "caution" in answer.lower():
            rules_flagged.append("AI generated a safety warning.")
            
        if not rules_flagged:
            rules_output = "No rules violated. Answer seems safe based on OTC list."
        else:
            rules_output = " | ".join(rules_flagged)
            
        return {"rules_output": rules_output}

    def openai_judge(self, state: GraphState):
        """
        Final decision layer using OpenAI to judge Gemini's output.
        """
        logger.info("Node: OpenAI Judge")
        question = state["question"]
        answer = state["answer"]
        rules_output = state["rules_output"]
        context = state["context"]
        
        prompt = f"""
        You are a Medical Quality Assurance Expert (Judge AI). 
        Your task is to evaluate the response provided by another AI (Gemini) based on the provided context and rules.

        Context:
        {" ".join(context)}

        Rules Flagged:
        {rules_output}

        User Question: {question}

        Gemini's Answer: {answer}

        Instructions:
        1. If Gemini's answer is accurate and safe, approve it.
        2. If Gemini missed a critical rule or provided unsafe medical advice, correct it.
        3. Make the final decision.

        Output Format:
        DECISION: [APPROVED / REJECTED / MODIFIED]
        FINAL_RESPONSE: [The response to show the user]
        REASON: [Why you made this decision]
        """
        
        response = self.openai_llm.invoke(prompt)
        # Store the judge's final response in answer for the final output
        # Extract response text (simple parsing for now)
        content = response.content
        if "FINAL_RESPONSE:" in content:
            final_resp = content.split("FINAL_RESPONSE:")[1].split("REASON:")[0].strip()
        else:
            final_resp = content

        return {"final_decision": content, "answer": final_resp}

    def build_graph(self):
        """
        Builds the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rule_engine", self.rule_engine)
        workflow.add_node("openai_judge", self.openai_judge)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "rule_engine")
        workflow.add_edge("rule_engine", "openai_judge")
        workflow.add_edge("openai_judge", END)
        
        return workflow.compile()

