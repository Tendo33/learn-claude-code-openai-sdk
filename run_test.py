import sys
from pathlib import Path

# Fix pythonpath
sys.path.append(str(Path(__file__).resolve().parent))

from agents import s_full

def test():
    print("Testing s_full...")
    
    # Send a prompt to the agent
    prompt = """
    Please do the following:
    1. Create a background task that sleeps for 2 seconds and echoes 'bg_done'
    2. While waiting, spawn a teammate named 'tester' with role 'QA'.
    3. Have the teammate create a task on the task board.
    4. Then ask the teammate to voluntarily shutdown.
    5. Check the background task status.
    """
    
    history = [{"role": "user", "content": prompt}]
    
    try:
        s_full.agent_loop(history)
        print("Agent loop completed.")
    except Exception as e:
        print(f"Agent loop failed: {e}")

if __name__ == "__main__":
    test()
