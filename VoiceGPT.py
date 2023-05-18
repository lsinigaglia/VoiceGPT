import os
import openai
import gradio as gr
from pydub.playback import play
import time
from huggingface_hub import HfApi
from google.cloud import texttospeech
from google.oauth2 import service_account

google_credentials = {
  "type": "service_account",
  "project_id": os.environ['GOOGLE_PROJECT_ID'],
  "private_key_id": os.environ['GOOGLE_PRIVATE_KEY_ID'],
    #to do: encript correctly the private key
  "private_key": "-----BEGIN PRIVATE KEY-----\n" + os.environ['GOOGLE_PRIVATE_KEY'] + "\n" + os.environ['GOOGLE_PRIVATE_KEY1'] + "\nGT4WWAeo0Yle8deKhWpLtzP6axQWsQGcyZgnBpkFCrjoTUA6Agm6scaxo/KdErRb\nkAxPWGGBiE9o8jA4n9hxd6F3O0PHEDJ+pPCTPP4nBEICfaRkzw2U7FHoPS/OS+g5\nqzHCVxJQTfYkB4EQFsStbVEhjYGXJT0UVLCiAHjN/8ls+DLsWG7uPHElvjm9PWp+\n94ihML4SztSTcEQw7CEI/4xD7OCKMwbeQnN+TXMNZO8VjiATFpPuqeuf2caU7uwT\nlZ6xDd13AgMBAAECggEADh/u2qQRynP80t4UNhuIP/TuBrOj2FEI7JqXEWe4VvIt\n0gqH1Co1SwAVVvtOCQ8+73hVZc/kaQUiJGqKdK01+Jdq64ATtIFPe53knVVNz5PV\nuQsU/KanVP5oz4vTqFklkzfDioZ+DrT7YZ00YfrgNw1OkUBB6QccrpXAG/6Ri+kG\ntNcYrZH932RE272TxMhg3R4hBEf4oKjIV3hmq72bZRcQ94M4+olnKAnMycN/VjS8\nWjAGpbvV84lVB7UbhJryh4dPOOKHjPeO553CS6swYMwRJqxZ61EmRh8lbM0/0Wrm\npCJwMTpiSgQS6OvY6Hxw3fhIjlWNivKhe6FyH7bT8QKBgQD8dA4T4G9j/pjIVcL1\neAokAhuqHWBamlvMtRil/Zy/8foirgPFmI5mhbCx/fQUWGc41bx+9dI2uK12Gdif\nnGpViOSnzNcnDU26eQtJsFojvDWI+aXnJw+Zyl1a6MG0q2J/sxr4OwmPlFmzk5Po\ntRgNPsgGG5IqlefhK5XSAF4+0QKBgQC2m7fURhrYg5kBZ+ITOvLlXZQduYhZw/2a\neM6ytJp+DwWcgIm4o+kyCOmTSImCrJLpSANMZw7sob7DUwRSwpKiOWbjdUBkwkHq\nLnoulbqWdFt+zwb3VloL00R3Op2B8UE5TbY+WSx03uSXQ/zdPILHaElbI89/FkA3\nltpJe4a5xwKBgQCNtMtOJvX/yCKUmWdFCGuQyMoklDbIxMpwvtqmGhTJvZctrkYZ\nvUd5juOfFbDTVsgiI/+ZuHyWENX1bA6nkVIyzxOiiR3gItyVpmpKo74FPxlB2Phi\nJw/fwLLkW7CXrHguvCeQXPtB6ADuBxHIVxa4IJyAqStD0j+FqkR/y4sbcQKBgAUK\nXVOlr74EO+f3Bx2CxgugvqLnaSUZLNnjtcjnBVCvd3cvcR3AoII6DsB7AxixTMjV\nrQmh2p6bhFl95COorUV/EiD7LpDZb9pX+BVrGqBmi9P/QPD42Dl1VnF4E7rvft5n\n" + os.environ['GOOGLE_PRIVATEKEY2'] + "\n0Jj48FyXqaADO89ckkxm7TUvm+20ROpnEkEd41gKPpg2ZaKpB1bItRzfbOqv6H0f\nlV2DtIyta86j+GK9WNFrjbKGfurYYgjfuAD+KwzwAx8wkuYvrKWmeIypSZUDD4+t\nw4cRmICLk3Vcp//UR8xrM0Jr\n-----END PRIVATE KEY-----\n",
  "client_email": os.environ['GOOGLE_EMAIL'],
  
  "client_id": os.environ['GOOGLE_CLIENT_ID'],
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/luca-voicegpt%40studious-rhythm-383600.iam.gserviceaccount.com"
}

openai.api_key = os.environ['OPENAI_API_KEY']

messages=[{"role": "system", "content": "Divide the answer in 2 sections. In the first you act like an ideal conversation partner. In the second you correct my english"},]
#messages=[{"role": "system", "content": "You are my english teacher. Your task is to make me an english lesson"},]

total_tokens_used = 0

def transcribe(audio):
    global messages
    global total_tokens_used
    if audio is None:
        print("Audio input is not received.")
        return "No audio input."
    with open(audio, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})
    first_two_messages = messages[:2]
    last_three_messages = messages[-3:]
    combined_messages = first_two_messages + last_three_messages
    
    # Measure the time taken for the ChatGPT API call
    chatgpt_start_time = time.time()
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = combined_messages)
    #response = openai.ChatCompletion.create(model="gpt-4-0314", messages = combined_messages) 
    chatgpt_end_time = time.time()
    chatgpt_duration = chatgpt_end_time - chatgpt_start_time
    print(f"Time taken for ChatGPT API call: {chatgpt_duration} seconds")
    
    system_message = response["choices"][0]["message"]
    response_text = system_message["content"]
    
    # Measure the time taken for the Google Text-to-Speech API call
    tts_start_time = time.time()
    synthesized_audio_path = synthesize_text(response_text)
    tts_end_time = time.time()
    tts_duration = tts_end_time - tts_start_time
    print(f"Time taken for Google Text-to-Speech API call: {tts_duration} seconds")
    
    messages.append(system_message)
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    tokens_this_call = response["usage"]["total_tokens"]
    total_tokens_used += tokens_this_call


    return synthesized_audio_path, response_text
#role_input = [{"role": "system", "content":gr.inputs.Textbox(lines=1, label='Role')},]

def synthesize_text(text):
    credentials = service_account.Credentials.from_service_account_info(google_credentials)    
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    with open("output.wav", "wb") as out:
        out.write(response.audio_content)

    return 'output.wav'

example_texts = "<h2><strong>Try with one of these sentences to start with!</strong></h2>Let's talk about traveling. What's your favorite destination?<br>Help me rephrase this: I'd be very happy if I can go to the party tonight.<br>Correct my English: Me has been studying English since two years."

ui = gr.Interface(
    fn = transcribe, 
    inputs = 
        gr.Audio(source="microphone", type="filepath"),
    outputs = [
        gr.Audio(label="answer"), 
        "text"],
    live = True,
    title = 'Talk & Learn with ChatGPTeacher: Your AI Language Companion<br>Click on "Record from microphone" and start talking!',
    article= example_texts,
    theme = gr.themes.Soft(),
    
    )

#ui.launch(share = True, auth = [('jacopo','SHINIGARIA'), ('Daniele', 'Gallucci')], auth_message= "Ciao Frank!")

ui.launch()
