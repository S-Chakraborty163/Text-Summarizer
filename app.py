import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "artifacts/model_trainer/pegasus-samsum-model"
tokenizer_path = "artifacts/model_trainer/tokenizer"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

def summarize_dialogue(text):
    if ":" not in text and not any(x in text for x in ["Person A", "Person B"]):
        
        return "Error: This model expects dialogue format. Please input a conversation with speakers."
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    
    output_tokens = model.generate(
        inputs["input_ids"], 
        max_length=128, 
        min_length=30, 
        num_beams=8,          
        length_penalty=0.8,    
        early_stopping=True
    )
    
    summary = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return summary

demo = gr.Interface(
    fn=summarize_dialogue,
    inputs=gr.Textbox(lines=10, label="Input Dialogue", placeholder="Enter chat log here..."),
    outputs=gr.Textbox(label="Generated Summary"),
    title="Dialogue Summarizer"
)

if __name__ == "__main__":
    demo.launch(share=True)

