import sys
import ollama
import torch
import sounddevice as sd
import time
import numpy as np
import re
import os
import random
import threading
import keyboard

import xml.etree.ElementTree as ET

from BABEL_preprocessTTS import preprocess
from xifSEARCH import main, Config

# Name ollama model
llm_model = 'PHRASE-2:latest'

# define the n results and size of grounded text
searchCONFIG = Config(topk_results=8, max_tokens=16)

# Grounding rules for the agent
sysGUIDANCE = '''\n\n
AGENT you are to ACT strictly in character according to the previous instructions.\n
Keep your IMPORTANCE, THOUGHTS, and FEELING private.\n
RESPOND to the USER directly.\n
Assume you have everything you need to PROCEED.\n\n'''

# Humanise the gaps in the conversation
filler_phrases = {
    "reflections": [
        "Let me think...",
        "Hmm, let's see...",
        "Just a moment...",
        "Okay, let me gather my thoughts...",
        "Let me ponder this...",
    ],
    "thoughts": [
        "Interesting...",
        "Alright, here's what I've got...",
        "After considering the information...",
        "Based on the given context...",
        "Taking everything into account...",
    ],
}

# Initialize TTS model parameters
params = {
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 8000,
    'device': 'cuda',
    'speaker': 'en_61',
    'max_chunk_size': 768,
    'threshold': 32,
    'rate': 'medium',
    'pitch': 'medium'
}

def load_ttsmodel(params):
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])
    return model

def escape_xml_chars(text):
    replacements = {
        '&': 'and',
        '"': '',
        "'": '',
        '<': ' ',
        '>': ' ',
        'é': 'e',
        'è': 'e',
        'à': 'a',
        'ç': 'c',
        'ô': 'o',
        'ü': 'u',
        'ä': 'a',
        'ö': 'o',
        'ß': 'ss',
        'ñ': 'n',
        '¿': '',
        '¡': '',
        '*': ' ',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

# Handle the XML for the TTS model, certain to break
def preprocess_ssml(text):
    def replace_tag(m):
        tag = m.group(1)
        attr = m.group(2)
        value = escape_xml_chars(m.group(3))
        content = escape_xml_chars(m.group(4))
        
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
            return f'{content}'

    text = re.sub(r'<speak>(.*?)</speak>', r'\1', text, flags=re.DOTALL)
    text = escape_xml_chars(text)
    text = re.sub(r'<(\w+)\s+(\w+)="([^"]+)">(.*?)</\1>', replace_tag, text, flags=re.DOTALL)
    return f'<speak>{text}</speak>'

def split_text_into_chunks(text, max_length=256, threshold=64):
    if len(text) <= max_length:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Post-process chunks to ensure proper punctuation and length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            subchunks = []
            start = 0
            while start < len(chunk):
                end = min(start + max_length, len(chunk))
                subchunk = chunk[start:end].strip()
                if not subchunk.endswith(('.', '!', '?')):
                    last_punctuation = max(subchunk.rfind('.'), subchunk.rfind('!'), subchunk.rfind('?'))
                    if last_punctuation != -1:
                        subchunk = subchunk[:last_punctuation + 1]
                subchunks.append(subchunk)
                start = end
            final_chunks.extend(subchunks)

    return chunks

def split_at_punctuation(sentence, max_length, threshold):
    for i in range(max_length - threshold, min(max_length, len(sentence))):
        if sentence[i] in ".!?,":
            return sentence[:i+1].strip(), sentence[i+1:].strip()

    last_space = sentence.rfind(' ', 0, max_length)
    if last_space != -1:
        return sentence[:last_space], sentence[last_space+1:]
    
    return sentence[:max_length].strip(), sentence[max_length:].strip()

def clean_text(text):
    # Only remove specific unwanted tags without converting to XML entities
    text = re.sub(r'<\/?[^>]+>', '', text)
    return text

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
        #clean xml tags using clean_text function
        chunk = clean_text(chunk)
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

# Contextualize the search results
def generate_assistant_prompt(search_results, query, conversation_history):
    prompt = "AGENT REFLECT ON OUR HISTORY:\n\n"
    if not conversation_history:
        prompt += random.choice(filler_phrases["reflections"]) + "\n\n"
    else:
        for i, (user_query, assistant_response) in enumerate(conversation_history, start=1):
            prompt += f"USER: {user_query}\n"
            prompt += f"AGENT: {assistant_response}\n\n"
    
    prompt += "AGENT THESE ARE YOUR IMMEDIATE THOUGHTS:\n\n"
    if not search_results:
        prompt += random.choice(filler_phrases["thoughts"]) + "\n\n"
    else:
        for i, (chunk, score, sentiment) in enumerate(search_results, start=1):
            prompt += f"AGENT MEMORY {i}. {chunk}\n"
            prompt += f"\nMEMORY SIGNIFICANCE: {score:.4f}\n"
            prompt += f"\nAGENT FEELING: {sentiment:.4f}\n\n"
    
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
        
        response = ollama.chat(model=llm_model, messages=[
            {'role': 'system', 'content': system_prompt + sysGUIDANCE},
            {'role': 'user', 'content': f'{prompt} {query}'},
        ])
        assistant_response = response['message']['content']
        preprocessed_response = preprocess(assistant_response)
        chunks = split_text_into_chunks(preprocessed_response, params['max_chunk_size'], params['threshold'])
        
        for chunk in chunks:
            if stop_playback:
                stop_playback = False
                skip_chunk = False
                return None
            if skip_chunk:
                skip_chunk = False
                continue
            # pre process the chunk
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
        return [], ""
    xml_file = file_path.replace('.txt', '.xml')
    if os.path.isfile(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        history = []
        system_prompt = ""
        for element in root:
            if element.tag == 'system_prompt':
                system_prompt = element.text
            elif element.tag == 'conversation':
                user_query = element.find('user_query').text
                assistant_response = element.find('assistant_response').text
                history.append((user_query, assistant_response))
        return history, system_prompt
    else:
        return [], ""

def save_conversation_history(file_path, conversation_history, system_prompt):
    if not file_path:
        return
    xml_file = file_path.replace('.txt', '.xml')
    
    # Check if the XML file already exists
    if os.path.isfile(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
    else:
        # If the file does not exist, create a new root element
        root = ET.Element('conversation_history')

    # Update or create the system prompt element
    system_prompt_elem = root.find('system_prompt')
    if system_prompt_elem is not None:
        system_prompt_elem.text = system_prompt
    else:
        system_prompt_elem = ET.SubElement(root, 'system_prompt')
        system_prompt_elem.text = system_prompt
    
    # Add new conversation elements directly, without deleting old ones
    for user_query, assistant_response in conversation_history:
        conversation = ET.SubElement(root, 'conversation')
        user_query_elem = ET.SubElement(conversation, 'user_query')
        user_query_elem.text = user_query
        assistant_response_elem = ET.SubElement(conversation, 'assistant_response')
        assistant_response_elem.text = assistant_response

    # Write the updated tree to the file
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)


def handle_read_query(query, params, model, conversation_history, system_prompt):
    file_name = query.split(' ', 1)[1].strip()  # Extract filename from the command
    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            file_content = file.read()

        # Append full content to the conversation history
        conversation_history.append((query, file_content))

        # Generate assistant's prompt without "FEELING" and "IMPORTANCE" scores
        assistant_prompt = f"USER:\n{query}\n\nAGENT REFLECT ON THE CONTENT:\n{file_content}\n\nUSER:\nPlease provide a summary and analysis of the content.\n\n"
        
        assistant_response = generate_assistant_response(assistant_prompt, file_content, params, model, system_prompt)
        
        # Append the assistant's response to the conversation history
        conversation_history.append(("Assistant's analysis of read content", assistant_response))

        # Print the assistant's response
        print("Assistant's response to the file content:")
        print(assistant_response)
    else:
        print(f"File not found: {file_name}")

def handle_search_query(query, file_path, params, model, conversation_history, include_memories):
    search_results = main(file_path, query, searchCONFIG)
    assistant_prompt = generate_assistant_prompt(search_results, query, conversation_history, include_memories)
    assistant_response = generate_assistant_response(assistant_prompt, query, params, model)
    if assistant_response is not None:
        print("\nAssistant Response:")
        print(assistant_response)
        conversation_history.append((query, assistant_response))

if __name__ == '__main__':

    model = load_ttsmodel(params)
    
    file_path, system_prompt_path = None, None

    while True:
        if not file_path or not system_prompt_path:
            file_path, system_prompt_path, query = get_user_input()
            conversation_history, system_prompt = load_conversation_history(file_path)
            # Load system prompt just once when file paths are obtained
            with open(system_prompt_path, 'r') as file:
                system_prompt = file.read().strip()
        else:
            query = input("Please enter your search query (or type 'quit' to exit):\n")

        # Command processing
        query_lower = query.lower()
        if query_lower == 'quit':
            save_conversation_history(file_path, conversation_history, system_prompt)
            break

        # Handling queries
        if query_lower.startswith('read'):
            handle_read_query(query, params, model, conversation_history, system_prompt)
        else:
            search_results = main(file_path, query, searchCONFIG)
            assistant_prompt = generate_assistant_prompt(search_results, query, conversation_history)
            assistant_response = generate_assistant_response(assistant_prompt, query, params, model, system_prompt)
            if assistant_response:
                print("\nAssistant Response:")
                print(assistant_response)
                conversation_history.append((query, assistant_response))

        # Maintain a fixed size for conversation history
        if len(conversation_history) > 5:
            conversation_history.pop(0)
