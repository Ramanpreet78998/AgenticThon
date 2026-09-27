from agent_tools import calculator

question = input("Ask something: ")

if any(symbol in question for symbol in ["+", "-", "*", "/"]):
    result = calculator(question)
    print("🤖 Agent used Calculator Tool")
    print("Answer:", result)

else:
    print("🤖 Agent did not use Calculator Tool")
    print("Question:", question)