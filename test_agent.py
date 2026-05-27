import os
from dotenv import load_dotenv
from agent import get_agent_executor

load_dotenv()

def test():
    try:
        print("Initializing Agent Executor...")
        executor = get_agent_executor()
        
        print("\nTesting: Asking a simple question...")
        response = executor.invoke({
            "messages": [("user", "I am getting cataract surgery. Is it covered?")]
        })
        final_output = response["messages"][-1].content
        print(f"\nResponse: {final_output}")
        print("\nAgent test successful!")
    except Exception as e:
        import traceback
        print(f"Error during agent execution:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test()
