import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

def predict(text):
    
    prediction = pipe(text, truncation=True, max_length=128)[0]["summary_text"]
    return prediction

interface = gr.Interface(
    fn=predict, 
    inputs=gr.Textbox(lines=10, placeholder="Paste your dialogue here..."), 
    outputs="text",
    title="Pegasus Text Summarizer",
    description="Fine-tuned on the Samsum dataset for dialogue summarization."
)

if __name__ == "__main__":
    interface.launch()