# WyzeAssist Question Expander API - Final Version
# Complete workflow: Similarity Search → Smart User Detection → LLM Processing → Recommendations
# Full CRUD operations for abbreviation management

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import os
import json
import uuid
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import warnings
warnings.filterwarnings('ignore')

# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    question: str
    user_name: Optional[str] = None

class SuccessfulExpansion(BaseModel):
    expanded_question: str
    message: str = "Question expanded successfully"
    recommendations: Optional[List[str]] = None

class UnclearQuestion(BaseModel):
    expanded_question: Optional[str] = None
    message: str
    recommendations: List[str]

class QuestionResponse(BaseModel):
    original_question: str
    result: Union[SuccessfulExpansion, UnclearQuestion]
    user_context: Optional[str] = None

class AbbreviationRequest(BaseModel):
    abbreviation: str
    definition: str

class AbbreviationResponse(BaseModel):
    abbreviation: str
    definition: str

class AbbreviationUpdate(BaseModel):
    definition: str

class MessageResponse(BaseModel):
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="WyzeAssist Question Expander API",
    description="Complete workflow for expanding questions with smart recommendations and abbreviation management",
    version="3.0.0"
)

# Global abbreviations storage
ABBREVIATIONS_FILE = "abbreviations.json"
CHROMA_PERSIST_DIR = "./chroma_db"

# WyzeAssist Legal Knowledge Base - Complete abbreviations dictionary
DEFAULT_ABBREVIATIONS = {
    "TK": "Timekeeper",
    "TTK": "Timekeeper",
    "tk": "Timekeeper",
    "ttk": "Timekeeper",
    "tkpr": "Timekeeper",
    "My": "Refers to the user's own records and roles",
    "TK#": "Timekeeper Number ",
    "TTK#": "Timekeeper Number",
    "#": "Timekeeper Number",
    "Atty": "Attorney",
    "Associate": "Attorney",
    "Assoc": "Attorney",
    "Sr Associate": "Senior Attorney",
    "Para": "Paralegal",
    "LC": "Law Clerk",
    "Equity Partner": "Partner",
    "Non-Equity Partner": "Partner",
    "Off Mgr": "Office Manager",
    "Loc Mgr": "Office Manager",
    "PG Lead": "Practice Group Leader",
    "Sect Lead": "Practice Group Leader",
    "Open": "Status - Active",
    "Closed": "Status - Terminated or Inactive",
    "Off": "Office",
    "Loc": "Office",
    "Dept": "Department",
    "Sect": "Section",
    "PG": "Practice Group",
    "Practice Group": "Section",
    "CL": "Client",
    "Client #": "Client Number",
    "CL #": "Client Number",
    "RLF": "Client Name - Richard Leyton Fingers (example alternative name)",
    "City": "Client Address - City",
    "State": "Client Address - State",
    "Country": "Client Address - Country",
    "MT": "Matter",
    "MT #": "Matter Number",
    "Matter #": "Matter Number",
    "MT Name": "Matter Name",
    "MT Off": "Matter Office",
    "MT Dept": "Matter Department",
    "MT PG": "Matter Practice Group",
    "Matter PG": "Matter Practice Group",
    "MT Type": "Matter Type",
    "batty": "Billing Attorney",
    "billing atty": "Billing Attorney",
    "supaty": "Supervising Attorney",
    "supatty": "Supervising Attorney",
    "respaty": "Responsible Attorney",
    "respatty": "Responsible Attorney",
    "orig atty": "Originating Attorney",
    "orig": "Originating Attorney",
    "pro atty": "Proliferating Attorney",
    "proatty": "Proliferating Attorney",
    "Phase Code": "Task & Phase Code",
    "Phase Desc": "Task & Phase Description",
    "Task Code": "Task Code",
    "Task Desc": "Task Description",
    "Act": "Activity",
    "Act Code": "Activity Code",
    "Phase": "Phase",
    "Task": "Task",
    "Activity": "Activity",
    "ABA List": "ABA List",
    "Billed": "Time Billed",
    "tbillamt": "Time Billed Amount",
    "Collected": "Time Collected",
    "Fee Rev": "Fee Revenue",
    "wk hours": "Worked Hours",
    "wk hrs": "Worked Hours",
    "bill hrs": "Billed Hours",
    "coll hrs": "Collected Hours",
    "wk rt": "Work Rate",
    "FF": "Flat Fees",
    "Fixed Fees": "Flat Fees",
    "Trust Bal": "Trust Balance",
    "TR BAL": "Trust Balance",
    "MTD": "Month to Date",
    "YTD": "Year to Date",
    "R12": "Rolling 12 Months",
    "PMTD": "Prior Month to Date",
    "PYTD": "Prior Year to Date",
    "LYTD": "Last Year to Date",
    "Electronic billed": "Ebilled",
    "e-billed": "Ebilled",
    "NB": "NonBillable",
    "Work in Progress": "WIP Fees",
    "UNB": "WIP Fees",
    "WIP": "Total WIP",
    "UNB Costs & Charges": "Total WIP",
    "Accounts Receivable": "AR Fees",
    "AR": "AR Fees",
    "AR Costs": "AR Costs",
    "AR Total": "AR Costs",
    "WOFF": "Write Off",
    "time written off": "Write Off",
    "UNALLOC": "Unallocated Cash",
    "TK Budget": "Timekeeper Budget or Target",
    "Budget": "Timekeeper Budget or Target",
    "Target": "Timekeeper Budget or Target",
    "GBNF": "Gone but not forgotten",
    "Doubtful": "Doubtful",
    "Reserve": "Reserver",
    "Fee Revenue": "Fee Revenue (configurable based on jurisdiction country and accounting rules)",
    "GMV": "Gross Merchandise Value - Total value of goods sold",
}


class QuestionExpanderAPI:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        self.vector_db = None
        self.abbreviations = self._load_abbreviations()
        self.doc_ids = {}
        self._setup_vector_store()
    
    def _load_abbreviations(self) -> Dict[str, str]:
        try:
            with open(ABBREVIATIONS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._save_abbreviations(DEFAULT_ABBREVIATIONS)
            return DEFAULT_ABBREVIATIONS.copy()
    
    def _save_abbreviations(self, abbreviations: Dict[str, str]):
        with open(ABBREVIATIONS_FILE, 'w') as f:
            json.dump(abbreviations, f, indent=2)
        self.abbreviations = abbreviations
    
    def _setup_vector_store(self):
        # Initialize ChromaDB with current abbreviations using unique UUIDs
        if self.vector_db:
            try:
                self.vector_db.delete_collection()
            except:
                pass
        
        documents = []
        self.doc_ids = {}
        
        for abbrev, definition in self.abbreviations.items():
            concept_id = str(uuid.uuid4())
            doc_id = f"concept_{concept_id}"
            
            doc = Document(
                page_content=f"{abbrev}: {definition}",
                metadata={
                    "abbreviation": abbrev, 
                    "definition": definition,
                    "id": doc_id
                }
            )
            documents.append(doc)
            self.doc_ids[abbrev.upper()] = doc_id
        
        if documents:
            self.vector_db = Chroma.from_documents(
                documents, 
                embedding=self.embeddings, 
                collection_name="firm_abbreviations",
                persist_directory=CHROMA_PERSIST_DIR,
                ids=[doc.metadata["id"] for doc in documents]
            )
            print(f"Vector store initialized with {len(documents)} abbreviations")
        else:
            print("No abbreviations to load into vector store")
    
    def _detect_personal_keywords(self, question: str) -> bool:
        # Smart detection of personal keywords
        personal_keywords = ["my", "mine", "i", "me", "myself", "our", "ours", "we", "us"]
        question_lower = question.lower()
        return any(keyword in question_lower.split() for keyword in personal_keywords)
    
    def expand_question(self, question: str, user_name: Optional[str] = None) -> Dict:
        # Complete workflow: Similarity Search → Smart User Detection → LLM Processing
        if not self.vector_db:
            return {
                "expanded_question": None,
                "message": "System not properly initialized",
                "recommendations": []
            }
        
        try:
            # Step 1: Similarity Search - Find relevant abbreviations
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 15})
            relevant_docs = retriever.get_relevant_documents(question)
            
            abbreviations = []
            for doc in relevant_docs:
                abbreviations.append(doc.page_content)
            
            # Step 2: Smart User Detection
            has_personal_keywords = self._detect_personal_keywords(question)
            user_context = ""
            if has_personal_keywords and user_name:
                user_context = f"for {user_name}"
            
            # Step 3: LLM Processing with Single System Prompt
            system_prompt = f"""
You are WyzeAssist, a legal/business analytics assistant. Your primary task is to transform the user's short question into a detailed, actionable query. Even if the question seems simple, expand it with as much relevant detail as possible. Only provide question recommendations if the question is completely unintelligible.

CONTEXT:
- Abbreviations and definitions (use these to expand questions): 
{chr(10).join(abbreviations[:10]) if abbreviations else "No relevant abbreviations found"}
- User context: {user_context if user_context else "No specific user context"}

INSTRUCTIONS:
1. Carefully read the user's question.
2. **Always attempt to expand the question into a detailed, actionable query.**
   - Use the available abbreviations and user context.
   - **IMPORTANT: If user context is provided (e.g., "for lynn"), include the user's name in the expanded query.**
   - Add relevant timeframes, metrics, and filters.
   - Focus on legal billing context.
   - Even for short questions, provide a detailed expansion.
   - Replace personal pronouns (my, mine, I, me) with the specific user name when user context is available.
3. **Only if the question is COMPLETELY UNINTELLIGIBLE:**
   - Respond with: "I couldn't fully understand your question. Here are some questions you might want to ask:"
   - Generate exactly 3 practical, actionable, and relevant question suggestions that the user might actually want to ask, based on the input and context.
   - List the 3 suggestions, each on a new line, numbered 1 to 3.
   - After the suggestions, add this message: "If none of these questions match your intent, please rephrase or provide more details so I can assist you better."

RESPONSE FORMAT:
- **If EXPANDABLE:** [Expanded Query with user name included if user context is provided]
- **If COMPLETELY UNINTELLIGIBLE:**
I couldn't fully understand your question. Here are some questions you might want to ask:
1. [First suggested question]
2. [Second suggested question]
3. [Third suggested question]
If none of these questions match your intent, please rephrase or provide more details so I can assist you better.

EXAMPLES:
- If user asks "List my clients" and user context is "for lynn" → "List all clients for lynn, including client names, contact information, active matters, and total fees paid to date."
- If user asks "TK hours" and no user context → "Show Timekeeper worked hours and billed hours for all active Timekeepers this month."

USER QUESTION: {question}
"""
            
            response = self.llm([HumanMessage(content=system_prompt)])
            llm_response = response.content.strip()
            
            # Parse LLM response
            if "I haven't understood your question clearly" in llm_response or "SUGGESTIONS:" in llm_response:
                # Question is unclear - extract recommendations
                parts = llm_response.split("SUGGESTIONS:")
                message = parts[0].strip()
                
                recommendations = []
                if len(parts) > 1:
                    suggestion_text = parts[1].strip()
                    # Extract numbered suggestions
                    lines = suggestion_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                            # Clean up the suggestion
                            clean_suggestion = line
                            for prefix in ['1.', '2.', '3.', '1)', '2)', '3)', '-', '•']:
                                if clean_suggestion.startswith(prefix):
                                    clean_suggestion = clean_suggestion[len(prefix):].strip()
                                    break
                            if clean_suggestion:
                                recommendations.append(clean_suggestion)
                
                # Ensure we have exactly 3 recommendations
                if len(recommendations) < 3:
                    default_suggestions = [
                        "Show me all active matters for my top 5 clients",
                        "What are my total billed hours and revenue for this quarter?", 
                        "Display accounts receivable aging report for overdue invoices"
                    ]
                    recommendations.extend(default_suggestions[len(recommendations):])
                
                return {
                    "expanded_question": None,
                    "message": message if message else "I haven't understood your question clearly, sir. Here are some questions you might want to ask:",
                    "recommendations": recommendations[:3]
                }
            else:
                # Question is clear - return expanded query
                return {
                    "expanded_question": llm_response,
                    "message": "Question expanded successfully",
                    "recommendations": None
                }
            
        except Exception as e:
            print(f"Error in expand_question: {str(e)}")
            return {
                "expanded_question": None,
                "message": "I haven't understood your question clearly, sir. Here are some questions you might want to ask:",
                "recommendations": [
                    "Show me all active matters for my top 5 clients",
                    "What are my total billed hours and revenue for this quarter?",
                    "Display accounts receivable aging report for overdue invoices"
                ]
            }
    
    def add_abbreviation(self, abbrev: str, definition: str):
        abbrev_upper = abbrev.upper()
        self.abbreviations[abbrev_upper] = definition
        self._save_abbreviations(self.abbreviations)
        self._setup_vector_store()
        print(f"Added abbreviation: {abbrev_upper} = {definition}")
    
    def update_abbreviation(self, abbrev: str, definition: str):
        abbrev_upper = abbrev.upper()
        if abbrev_upper not in self.abbreviations:
            raise ValueError(f"Abbreviation '{abbrev}' not found")
        
        old_definition = self.abbreviations[abbrev_upper]
        self.abbreviations[abbrev_upper] = definition
        self._save_abbreviations(self.abbreviations)
        self._setup_vector_store()
        print(f"Updated abbreviation: {abbrev_upper} from '{old_definition}' to '{definition}'")
    
    def delete_abbreviation(self, abbrev: str):
        abbrev_upper = abbrev.upper()
        if abbrev_upper not in self.abbreviations:
            raise ValueError(f"Abbreviation '{abbrev}' not found")
        
        deleted_definition = self.abbreviations[abbrev_upper]
        del self.abbreviations[abbrev_upper]
        self._save_abbreviations(self.abbreviations)
        self._setup_vector_store()
        print(f"Deleted abbreviation: {abbrev_upper} = {deleted_definition}")
    
    def get_abbreviation(self, abbrev: str) -> Optional[str]:
        return self.abbreviations.get(abbrev.upper())
    
    def get_all_abbreviations(self) -> Dict[str, str]:
        return self.abbreviations.copy()


# Initialize the expander
try:
    expander = QuestionExpanderAPI()
    print("WyzeAssist Question Expander API initialized successfully")
except ValueError as e:
    print(f"Error initializing API: {e}")
    expander = None

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "WyzeAssist Question Expander API - Final Version",
        "version": "3.0.0",
        "status": "running" if expander else "error",
        "workflow": "Similarity Search → Smart User Detection → LLM Processing → Recommendations",
        "endpoints": "/docs for API documentation",
        "description": "Complete question expansion with smart recommendations and abbreviation management"
    }

@app.post("/expand", response_model=QuestionResponse)
async def expand_question(request: QuestionRequest):
    # Main question expansion endpoint with complete workflow
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    
    try:
        result = expander.expand_question(request.question, request.user_name)
        
        # Determine user context
        user_context = None
        if expander._detect_personal_keywords(request.question) and request.user_name:
            user_context = f"for {request.user_name}"
        
        # Create appropriate response based on result
        if result["expanded_question"]:
            # Successful expansion
            response_result = SuccessfulExpansion(
                expanded_question=result["expanded_question"],
                message=result["message"]
            )
        else:
            # Unclear question with recommendations
            response_result = UnclearQuestion(
                expanded_question=result["expanded_question"],
                message=result["message"],
                recommendations=result["recommendations"]
            )
        
        return QuestionResponse(
            original_question=request.question,
            result=response_result,
            user_context=user_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error expanding question: {str(e)}")

# Abbreviation Management Endpoints (Full CRUD)

@app.get("/abbreviations", response_model=Dict[str, str])
async def get_all_abbreviations():
    # Get all abbreviations
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    return expander.get_all_abbreviations()

@app.get("/abbreviations/{abbrev}", response_model=AbbreviationResponse)
async def get_abbreviation(abbrev: str):
    # Get specific abbreviation
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    
    definition = expander.get_abbreviation(abbrev)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Abbreviation '{abbrev}' not found")
    
    return AbbreviationResponse(abbreviation=abbrev.upper(), definition=definition)

@app.post("/abbreviations", response_model=MessageResponse)
async def add_abbreviation(request: AbbreviationRequest):
    # Add new abbreviation
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    
    try:
        if expander.get_abbreviation(request.abbreviation):
            raise HTTPException(
                status_code=400, 
                detail=f"Abbreviation '{request.abbreviation}' already exists. Use PUT to update."
            )
        
        expander.add_abbreviation(request.abbreviation, request.definition)
        return MessageResponse(
            message=f"Abbreviation '{request.abbreviation.upper()}' added successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding abbreviation: {str(e)}")

@app.put("/abbreviations/{abbrev}", response_model=MessageResponse)
async def update_abbreviation(abbrev: str, request: AbbreviationUpdate):
    # Update existing abbreviation
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    
    try:
        expander.update_abbreviation(abbrev, request.definition)
        return MessageResponse(
            message=f"Abbreviation '{abbrev.upper()}' updated successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating abbreviation: {str(e)}")

@app.delete("/abbreviations/{abbrev}", response_model=MessageResponse)
async def delete_abbreviation(abbrev: str):
    # Delete abbreviation
    if not expander:
        raise HTTPException(status_code=500, detail="API not properly initialized")
    
    try:
        expander.delete_abbreviation(abbrev)
        return MessageResponse(
            message=f"Abbreviation '{abbrev.upper()}' deleted successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting abbreviation: {str(e)}")

# Utility Endpoints

@app.get("/health")
async def health_check():
    # Health check endpoint
    if not expander:
        return {
            "status": "unhealthy",
            "error": "API not properly initialized",
            "abbreviations_count": 0
        }
    
    return {
        "status": "healthy",
        "abbreviations_count": len(expander.abbreviations),
        "service": "WyzeAssist Question Expander",
        "version": "3.0.0",
        "workflow": "Complete question expansion with smart recommendations"
    }

@app.get("/examples")
async def get_examples():
    # Show example question expansions and API usage
    return {
        "successful_expansions": [
            {
                "input": "My new clients",
                "output": "List all new Clients and associated Matters opened this month or in the last 21 days, whichever period is longer, for [user]"
            },
            {
                "input": "TK hours",
                "output": "Show Timekeeper worked hours and billed hours for all active Timekeepers this month"
            },
            {
                "input": "AR aging report",
                "output": "Show Accounts Receivable aging report with outstanding balances by Client and Matter"
            }
        ],
        "unclear_question_response": {
            "message": "I haven't understood your question clearly, sir. Here are some questions you might want to ask:",
            "recommendations": [
                "Show me all active matters for my top 5 clients",
                "What are my total billed hours and revenue for this quarter?",
                "Display accounts receivable aging report for overdue invoices"
            ]
        },
        "api_usage": {
            "expand_question": "POST /expand with {'question': 'My new clients', 'user_name': 'John'}",
            "add_abbreviation": "POST /abbreviations with {'abbreviation': 'EBITDA', 'definition': 'Earnings Before Interest, Taxes, Depreciation, and Amortization'}",
            "update_abbreviation": "PUT /abbreviations/EBITDA with {'definition': 'Updated definition'}",
            "delete_abbreviation": "DELETE /abbreviations/EBITDA",
            "get_abbreviation": "GET /abbreviations/TK"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting WyzeAssist Question Expander API - Final Version...")
    print("Complete workflow: Similarity Search → Smart User Detection → LLM Processing → Recommendations")
    print("Full CRUD operations for abbreviation management")
    print("Make sure OPENAI_API_KEY environment variable is set")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
