import gradio as gr
import pickle

# Load the model & vectorizer
with open("spam_classifier.pkl", "rb") as model_file:
    NB_classifier = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

def classify_message(message):
    transformed_text = vectorizer.transform([message])
    prediction = NB_classifier.predict(transformed_text)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Interface setup with interactive features
interface = gr.Interface(
    fn=classify_message,
    inputs=gr.Textbox(
        label="Enter Message", 
        placeholder="Type your message here...", 
        lines=2, 
        interactive=True,  # Textbox becomes interactive
        elem_id="input_message"
    ),
    outputs=gr.Textbox(
        label="Prediction", 
        interactive=False,  # Read-only output
        elem_id="output_text"
    ),
    title="Interactive Spam Classifier",
    description="This model detects whether a message is spam or not. Type a message, and it will classify it as 'Spam' or 'Not Spam'.",
    theme="compact",  
    examples=[
        ["Free offer, click here now!"], 
        ["Hello, how are you?"], 
        ["Claim your prize now!", "Hurry, limited time offer!"],  
        ["Let's meet for coffee tomorrow.", "Are you free this weekend?"]
    ],
    live=True,  # Enable live updates for prediction as the user types
    allow_flagging="never",  # Optional: Disable flagging for simplicity
    css="""
        #gradio-container {
            font-family: 'Arial', sans-serif;
        }
        #input_message {
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            transition: border-color 0.3s ease;
        }
        #input_message:focus {
            border-color: #4CAF50;
        }
        #gradio-container input[type='submit'] {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            cursor: pointer;
        }
        #gradio-container input[type='submit']:hover {
            background-color: #45a049;
        }
        #output_text {
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
            background-color: #424242;
            text-align: center;
            color: #333;  
            transition: color 0.3s ease;
        }
    """
)

interface.launch()
