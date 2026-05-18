import sys
import src.vectorstore as vs
from src.rag_chain import RAGChain

# ANSI Color Codes
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_help():
    print(f"\n{YELLOW}Available Commands:{RESET}")
    print("  quit  - Exit the chat")
    print("  clear - Clear conversation history")
    print("  help  - Show this help message")
    print("  stats - Show database statistics")
    print()

def main():
    print(f"{GREEN}🤖 Starting Free RAG Terminal Chat...{RESET}")
    rag = RAGChain()
    
    print(f"{YELLOW}Type 'help' for commands. Type 'quit' to exit.{RESET}\n")
    
    while True:
        try:
            user_input = input(f"{BLUE}You: {RESET}").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"{GREEN}Goodbye!{RESET}")
                break
                
            if user_input.lower() == 'help':
                print_help()
                continue
                
            if user_input.lower() == 'clear':
                rag.clear_history()
                print(f"{YELLOW}History cleared.{RESET}")
                continue
                
            if user_input.lower() == 'stats':
                stats = vs.get_stats()
                print(f"{YELLOW}Database Stats: {stats}{RESET}")
                continue
                
            # Ask RAG
            result = rag.ask(user_input)
            
            print(f"\n{GREEN}AI: {RESET}{result['answer']}")
            
            # Print sources
            if result['sources']:
                print(f"\n{YELLOW}📚 Sources:{RESET}")
                for i, src in enumerate(result['sources']):
                    score = src.get('fusion_score', src.get('score', 0))
                    print(f"  [{i+1}] {src.get('filename')} (Page {src.get('page')}) - Score: {score:.4f}")
            
            print(f"{YELLOW}⏱️ Latency: {result['latency']}s{RESET}\n")
            
        except KeyboardInterrupt:
            print(f"\n{GREEN}Goodbye!{RESET}")
            break
        except Exception as e:
            print(f"{YELLOW}Error: {e}{RESET}")

if __name__ == "__main__":
    main()
