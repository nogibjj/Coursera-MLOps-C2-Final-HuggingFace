from transformers import TFAutoModel
import gradio as gr


#uncomment to download model
#from transformers import pipeline
#model = pipeline(
#    "summarization",
#    model="sshleifer/distilbart-cnn-12-6",
#    revision="a4f8f3e",
#)

model = TFAutoModel.from_pretrained("summarizeApp", from_pt=True)


def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary


# create an interface for the model
with gr.Interface(predict, "textbox", "text") as interface:
    interface.launch()
