from agent import agent

evaluation_questions = [
    "What is the return window for most items?",
    "What is the return deadline for Apple products purchased in December?",
    "How does the policy handle items that arrive damaged or defective?",
    "What should a customer do if a purchased product is not working on arrival?",
    "Is there a cancellation fee for orders?",
    "Does Samsung follow the same return policy as Apple?",
    "What items are eligible for return?",
    "refund timeline for credit cards"
]

def run_evaluation():
    for q in evaluation_questions:
        result = agent.invoke({"question": q})
        print("Question:", q)
        print("Answer:", result["answer"])
        print("Action:", result["action"])
        print("Action Input:", result["action_input"])
        print("-" * 50)

if __name__ == "__main__":
    run_evaluation()



