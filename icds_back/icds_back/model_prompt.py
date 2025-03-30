import ollama
from app import pcd_to_llm_text

def prompt_model(model, prompt):
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user", 
            'content':prompt
        }],
        options=[{'temperature': 0.1}]
    )

    return response['message']['content']

if __name__ == "__main__":
    model = "gemma3:latest"

    data = pcd_to_llm_text("CIE.pcd", max_points=1000)
    prompt = f"{data} Here's a point cloud data. it consists of 3D points and rbg colors. Are you able to identfy the objects within the point cloud?"

    print(prompt_model(model, prompt))
