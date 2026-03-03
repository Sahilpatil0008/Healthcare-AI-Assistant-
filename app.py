from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

print("Loading AI model... Please wait.")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if torch.cuda.is_available() else -1
)

print("Model Loaded Successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["msg"]

    # Strong structured medical instruction
    prompt = f"""
You are a professional medical assistant.

Answer the question in detailed and structured format.

Provide:
1. Definition
2. Symptoms
3. Causes
4. Treatment options
5. When to see a doctor

Explain clearly in simple language.

Question: {user_message}

Answer:
"""

    response = generator(
        prompt,
        max_length=400,
        min_length=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    answer = response[0]["generated_text"]

    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(debug=True)