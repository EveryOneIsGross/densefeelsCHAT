import sys
from xifSEARCH_02 import main
import ollama
import torch
import sounddevice as sd
import time
import numpy as np
import re
import os
import threading
import keyboard
from BABEL_preprocessTTS_08 import preprocess
import xml.etree.ElementTree as ET

def load_ttsmodel(params):
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])
    return model

def preprocess_ssml(text):
    def replace_tag(m):
        tag = m.group(1)
        attr = m.group(2)
        value = m.group(3)
        content = m.group(4)
        
        if tag == 'b':
            if attr == 'time':
                return f'<break time="{value}"/>{content}'
            elif attr == 'strength':
                return f'<break strength="{value}"/>{content}'
            else:
                return f'{content}'
        elif tag == 'prosody' and attr in ['rate', 'pitch']:
            if (attr == 'rate' and value in ['slow', 'medium', 'fast']) or \
               (attr == 'pitch' and value in ['low', 'medium', 'high']):
                return f'<prosody {attr}="{value}">{content}</prosody>'
            else:
                return f'{content}'
        elif tag in ['p', 's']:
            return f'<{tag}>{content}</{tag}>'
        else:
            # Strip other tags that are not part of the supported SSML tags
            return content

    text = re.sub(r'<speak>(.*?)</speak>', r'\1', text, flags=re.DOTALL)
    text = preprocess(text)
    text = re.sub(r'<(\w+)\s+(\w+)="([^"]+)">(.*?)</\1>', replace_tag, text, flags=re.DOTALL)
    return f'<speak>{text}</speak>'

def split_text_into_chunks(text, max_length=128, threshold=20):
    chunks = []
    current_chunk = ""

    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            while len(current_chunk + ' ' + sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                if len(sentence) > max_length:
                    part, sentence = split_at_punctuation(sentence, max_length, threshold)
                    chunks.append(part.strip())
                else:
                    current_chunk = sentence
                    sentence = ""

            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_at_punctuation(sentence, max_length, threshold):
    for i in range(max_length - threshold, min(max_length, len(sentence))):
        if sentence[i] in ".!?,":
            return sentence[:i+1].strip(), sentence[i+1:].strip()

    last_space = sentence.rfind(' ', 0, max_length)
    if last_space != -1:
        return sentence[:last_space], sentence[last_space+1:]
    
    return sentence[:max_length].strip(), sentence[max_length:].strip()

def generate_audio_from_text(model, text, params, max_length=512, padding_duration=0.5):
    if not isinstance(text, str):
        text = str(text)

    def generate_chunk(chunk):
        try:
            waveform = model.apply_tts(ssml_text=chunk, sample_rate=params['sample_rate'], speaker=params['speaker'])
            audio_np = waveform.squeeze(0).cpu().numpy()
            padding_samples = int(padding_duration * params['sample_rate'])
            padding = np.zeros(padding_samples, dtype=np.float32)
            audio_np = np.concatenate((audio_np, padding))
            return audio_np
        except ValueError as e:
            print(f"ValueError: {e}")
            print(f"Problematic text: {chunk}")
            return None

    text = re.sub(r'<speak>(.*?)</speak>', r'\1', text, flags=re.DOTALL)
    chunks = split_text_into_chunks(text, max_length, params['threshold'])
    audio_chunks = []
    for chunk in chunks:
        chunk = f'<speak>{chunk}</speak>'
        audio_chunk = generate_chunk(chunk)
        if audio_chunk is not None:
            audio_chunks.append(audio_chunk)

    return np.concatenate(audio_chunks) if audio_chunks else None

stop_playback = False
skip_chunk = False

def play_audio(audio, sample_rate):
    global stop_playback
    sd.play(audio, samplerate=sample_rate)
    time.sleep(len(audio) / sample_rate)
    sd.stop()
    stop_playback = False

def add_silence_and_fadeout(audio, silence_duration, fadeout_duration, sample_rate):
    silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
    fadeout_samples = int(fadeout_duration * sample_rate)
    fadeout_curve = np.linspace(1, 0, fadeout_samples)
    audio[-fadeout_samples:] *= fadeout_curve
    return np.concatenate((audio, silence))

def get_user_input():
    print("Please provide the path to the input document:")
    file_path = input()
    print("Please provide the path to the system prompt file:")
    system_prompt_path = input()
    print("Please enter your search query (or type 'quit' to exit):")
    query = input()
    return file_path, system_prompt_path, query

def generate_assistant_prompt(search_results, query, conversation_history):
    prompt = "REFLECTIONS:\n\nummm...\n\n"
    for i, (user_query, assistant_response) in enumerate(conversation_history, start=1):
        prompt += f"USER: {user_query}\n"
        prompt += f"ASSISTANT: {assistant_response}\n\n"
    prompt += "THOUGHTS:\n\nHmmm...\n\n"
    for i, (chunk, score, sentiment) in enumerate(search_results, start=1):
        prompt += f"IDEA {i}. {chunk}\n"
        prompt += f"\nIMPORTANCE: {score:.4f}\n"
        prompt += f"\nFEELING: {sentiment:.4f}\n\n"
    prompt += f"USER:\n {query}\n\n"
    return prompt

def generate_assistant_response(prompt, query, params, model, system_prompt, file_path=None):
    global stop_playback, skip_chunk
    if query.lower().startswith('read'):
        file_name = query.split(' ', 1)[1].strip()
        if os.path.isfile(file_name):
            with open(file_name, 'r') as file:
                file_content = file.read()
            chunks = split_text_into_chunks(file_content, params['max_chunk_size'])
            for chunk in chunks:
                if stop_playback:
                    stop_playback = False
                    skip_chunk = False
                    return None
                if skip_chunk:
                    skip_chunk = False
                    continue
                preprocessed_chunk = preprocess_ssml(f'<speak>{chunk}</speak>')
                audio = generate_audio_from_text(model, preprocessed_chunk, params)
                if audio is not None:
                    silence_duration = 0.5
                    fadeout_duration = 0.5 / 1000
                    audio_with_silence_and_fadeout = add_silence_and_fadeout(audio, silence_duration, fadeout_duration, params['sample_rate'])
                    play_thread = threading.Thread(target=play_audio, args=(audio_with_silence_and_fadeout, params['sample_rate']))
                    play_thread.start()
                    while play_thread.is_alive():
                        if keyboard.is_pressed('s'):
                            skip_chunk = True
                            sd.stop()
                            print("Skipping the current chunk...")
                            break
                        if keyboard.is_pressed('q'):
                            stop_playback = True
                            sd.stop()
                            print("Stopping playback...")
                            return None
                        time.sleep(0.1)
            return file_content
        else:
            print(f"File not found: {file_name}")
            return None
    else:
        print(prompt)
        response = ollama.chat(model='phi3:14b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'{prompt} {query}'},
        ])
        assistant_response = response['message']['content']
        
        chunks = split_text_into_chunks(assistant_response, params['max_chunk_size'], params['threshold'])
        
        for chunk in chunks:
            if stop_playback:
                stop_playback = False
                skip_chunk = False
                return None
            if skip_chunk:
                skip_chunk = False
                continue
            preprocessed_chunk = preprocess_ssml(f'<speak>{chunk}</speak>')
            audio = generate_audio_from_text(model, preprocessed_chunk, params)
            if audio is not None:
                silence_duration = 0.5
                fadeout_duration = 0.5 / 1000
                audio_with_silence_and_fadeout = add_silence_and_fadeout(audio, silence_duration, fadeout_duration, params['sample_rate'])
                play_thread = threading.Thread(target=play_audio, args=(audio_with_silence_and_fadeout, params['sample_rate']))
                play_thread.start()
                while play_thread.is_alive():
                    if keyboard.is_pressed('s'):
                        skip_chunk = True
                        sd.stop()
                        print("Skipping the current chunk...")
                        break
                    if keyboard.is_pressed('q'):
                        stop_playback = True
                        sd.stop()
                        print("Stopping playback...")
                        return None
                    time.sleep(0.1)
        
        return assistant_response

def load_conversation_history(file_path):
    if not file_path:
        return []
    xml_file = file_path.replace('.txt', '.xml')
    if os.path.isfile(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        history = []
        for conversation in root.findall('conversation'):
            user_query = conversation.find('user_query').text
            assistant_response = conversation.find('assistant_response').text
            history.append((user_query, assistant_response))
        return history
    else:
        return []

def save_conversation_history(file_path, conversation_history):
    if not file_path:
        return
    xml_file = file_path.replace('.txt', '.xml')
    root = ET.Element('conversation_history')
    for user_query, assistant_response in conversation_history:
        conversation = ET.SubElement(root, 'conversation')
        user_query_elem = ET.SubElement(conversation, 'user_query')
        user_query_elem.text = user_query
        assistant_response_elem = ET.SubElement(conversation, 'assistant_response')
        assistant_response_elem.text = assistant_response
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)

def handle_read_query(query, params, model, conversation_history):
    file_content = generate_assistant_response(None, query, params, model)
    if file_content:
        conversation_history.append((query, file_content))

def handle_search_query(query, file_path, params, model, conversation_history):
    search_results = main(file_path, query)
    assistant_prompt = generate_assistant_prompt(search_results, query, conversation_history)
    assistant_response = generate_assistant_response(assistant_prompt, query, params, model)
    if assistant_response is not None:
        print("\nassistant Response:")
        print(assistant_response)
        conversation_history.append((query, assistant_response))

if __name__ == '__main__':
    params = {
        'language': 'en',
        'model_id': 'v3_en',
        'sample_rate': 48000,
        'device': 'cuda',
        'speaker': 'en_61',
        'max_chunk_size': 768,
        'threshold': 32
    }

    model = load_ttsmodel(params)
    file_path = None
    system_prompt_path = None
    while True:
        if not file_path or not system_prompt_path:
            file_path, system_prompt_path, query = get_user_input()
            conversation_history = load_conversation_history(file_path)
            with open(system_prompt_path, 'r') as file:
                system_prompt = file.read().strip()
        else:
            print("Please enter your search query (or type 'quit' to exit):")
            query = input()

        if query.lower() == 'quit':
            save_conversation_history(file_path, conversation_history)
            break

        if query.lower().startswith('read'):
            file_content = generate_assistant_response(None, query, params, model, system_prompt)
            if file_content:
                conversation_history.append((query, file_content))
        else:
            search_results = main(file_path, query)
            assistant_prompt = generate_assistant_prompt(search_results, query, conversation_history)
            assistant_response = generate_assistant_response(assistant_prompt, query, params, model, system_prompt)
            if assistant_response is not None:
                print("\nassistant Response:")
                print(assistant_response)
                conversation_history.append((query, assistant_response))

        if len(conversation_history) > 5:
            conversation_history.pop(0)