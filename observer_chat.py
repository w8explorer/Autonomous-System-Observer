import os
import sys
import logging
from langchain_openai import ChatOpenAI
from core.vector_store import CodeVectorStore

# Standard colors
B = "\033[1m"
R = "\033[0m"
CY = "\033[96m"
MA = "\033[95m"
YL = "\033[93m"
GR = "\033[92m"
DM = "\033[2m"

# Project Constants
VECTOR_DB = "/home/ubuntu/observer_vectors"
LLM_URL = "http://localhost:1234/v1"

logging.basicConfig(level=logging.ERROR)

def _divider(title="", colour=CY):
    text = f" {title} " if title else ""
    print(f"\n{colour}{text:─^64}{R}")

def start_chat():
    _divider("Observer Expert Consultation Mode", MA)
    print(f"{DM}   Loading semantic memory and reasoning engine...{R}")
    
    # 1. Initialize LLM (Local Port 1234)
    llm = ChatOpenAI(
        base_url=LLM_URL,
        api_key="not-needed",
        model_name="llama-3.2-1b-instruct",
        temperature=0.1
    )

    # 2. Initialize Vector Store
    store = CodeVectorStore(device="cpu")
    if os.path.exists(VECTOR_DB):
        store.load(VECTOR_DB)
    else:
        print(f"{YL}⚠️  Semantic memory not found. Run a scan first.{R}")
        return

    # 3. Setup Consultant
    from engine_adapter import AdaptiveObserverConsultant
    consultant = AdaptiveObserverConsultant(llm, store)

    print(f"{GR}✅ System Ready.{R}")
    print(f"{DM}Type your questions below. Use {B}/exit{R}{DM} to leave.{R}")

    while True:
        try:
            print(f"\n{MA}{B}🕵️‍♂️ Observer {CY}>{R} ", end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/exit", "exit", "quit", "/quit"]:
                print(f"{MA}Consultation closed. Goodbye!{R}")
                break
                
            if user_input.lower() in ["/clear", "clear"]:
                os.system('clear')
                _divider("Observer Expert Consultation Mode", MA)
                continue

            print(f"{DM}   Reasoning...{R}")
            
            # Run the adaptive loop
            result = consultant.ask(user_input)
            
            # Show "Thinking" (Intermediate steps)
            for step in result.get("intermediate_steps", []):
                print(f"   {DM}{step}{R}")
            
            # Final Answer
            _divider(colour=DM)
            print(f"\n{GR}{B}🎯 Response:{R}\n")
            print(result.get("final_answer", "I could not formulate an answer."))
            _divider(colour=DM)

        except KeyboardInterrupt:
            print(f"\n{MA}Consultation closed.{R}")
            break
        except Exception as e:
            print(f"\n{YL}⚠️  Analysis Error: {str(e)}{R}")

if __name__ == "__main__":
    start_chat()
