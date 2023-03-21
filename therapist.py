from time import sleep

import gradio as gr
import openai, subprocess, os
from pathlib import Path
from dotenv import load_dotenv
import os



# Obtain the Environment Variables from .env file
dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)
openai.api_key = os.getenv('OPENAI_API_KEY')

messages = [{"role": "system", "content": 'You are a therapist. Respond to all input in 25 words or less.'}]

def transcribe(audio):
    global messages

    os.rename(audio, audio+".wav")
    audio_file = open(audio+".wav", "rb")
    # print(audio_file)
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    # print(f"user transcription: {transcript}")

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    # print(f"chatGPT response: {response}")

    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    subprocess.call(["say", system_message['content']])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch()
ui.launch()