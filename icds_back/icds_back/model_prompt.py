import ollama

def prompt_model(model, prompt):
    ollama.chat(
        model=model,
        messages=[{
            "user": "user", 'content':prompt,
        }],
        options=[{'temperature': 0.1}]
    )



