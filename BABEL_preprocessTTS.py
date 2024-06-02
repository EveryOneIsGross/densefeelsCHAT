import re
import num2words

punctuation = r'\s,.?!/\)\]\[\<>\(\):;"'
alphabet_map = {
    "A": " ah ",
    "B": " bee. ",
    "C": " see. ",
    "D": " dee. ",
    "E": " eee. ",
    "F": " eff. ",
    "G": " jee. ",
    "H": " haich. ",
    "I": " eye ",
    "J": " jay. ",
    "K": " kay. ",
    "L": " ell. ",
    "M": " emm. ",
    "N": " enn. ",
    "O": " ohh. ",
    "P": " pee. ",
    "Q": " queue. ",
    "R": " are. ",
    "S": " ess. ",
    "T": " tee. ",
    "U": " you. ",
    "V": " vee. ",
    "W": " double you. ",
    "X": " ex. ",
    "Y": " why? ",
    "Z": " zed. ",
    "a": " a ",
    "b": " bee. ",
    "c": " see. ",
    "d": " dee. ",
    "e": " eee. ",
    "f": " eff. ",
    "g": " jee. ",
    "h": " haich. ",
    "i": " eye ",
    "j": " jay. ",
    "k": " kay. ",
    "l": " ell. ",
    "m": " emm. ",
    "n": " enn. ",
    "o": " ohh. ",
    "p": " pee. ",
    "q": " queue. ",
    "r": " are. ",
    "s": " ess. ",
    "t": " tee. ",
    "u": " you. ",
    "v": " vee. ",
    "w": " double you. ",
    "x": " ex. ",
    "y": " why? ",
    "z": " zed. "
}

known_abbreviations_map = {
    "USA": "U S A",
    "UK": "U K",
    "NASA": "NASA",
    "FBI": "F B I",
    "CIA": "C I A",
    "GPT": "Gee Pee Tee",
    "AI": "Aye. Eye. ",
    "LLM": "Owl Owl Emm",
    "OS": "Oh Ess",
    "iOS": "Eye Oh Ess",
    "GPT-3": "Gee Pee Tee Three",
    "GPT-4": "Gee Pee Tee Four",
    "ABC": "A B C",
    "abc": "a b c",
    "ai": " Aye. Eye! ",
    "llm": "owl, owl, emm, ",
    "os": "oh, ess, ",
    "gpt": "gee pee tee",
    "CLI ": "command line interface",
    " cli ": "command line interface",
    "ChatGPT": "Chat Gee Pee Tee",
    "chatGPT": "chat Gee Pee Tee",
}

def preprocess(string):
    print("Original:", string)  # Debugging
    string = re.sub(r'\n', ' .  .  .  ', string)  # Replace newline characters with multiple pause tokens
    #print("After replacing newlines:", string)  # Debugging
    string = remove_surrounded_chars(string)
    #print("After removing surrounded chars:", string)  # Debugging
    string = string.replace('"', '')
    string = string.replace('\u201D', '').replace('\u201C', '')  # right and left quote
    string = string.replace('\u201F', '')  # italic looking quote
    #print("After replacing quotes:", string)  # Debugging
    string = convert_num_locale(string)
    #print("After converting number locale:", string)  # Debugging
    string = replace_negative(string)
    #print("After replacing negatives:", string)  # Debugging
    string = replace_roman(string)
    #print("After replacing roman numerals:", string)  # Debugging
    string = hyphen_range_to(string)
    #print("After replacing hyphen ranges:", string)  # Debugging
    string = convert_years_to_words(string)  # Convert years to their spoken equivalents
    #print("After converting years to words:", string)  # Debugging
    string = num_to_words(string)
    #print("After converting numbers to words:", string)  # Debugging
    string = replace_abbreviations(string)
    #print("After replacing abbreviations:", string)  # Debugging
    string = replace_single_letters(string)
    #print("After replacing single letters:", string)  # Debugging
    #string = re.sub(rf'\s*([{re.escape(punctuation)}])\s*', r' \1 ', string)
    #print("After adding space around punctuation:", string)  # Debugging
    string = string.strip() # Remove leading and trailing whitespace
    string = ' '.join(string.split())  # Ensure single spaces between words
    print("Final preprocessed string:", string)  # Debugging
    return string

def remove_surrounded_chars(string):
    if re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL):
        m = re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL)
        string = m.group(0)
    return re.sub(r'\*[^*]*?(\*|$)', '', string)

def convert_num_locale(text):
    pattern = re.compile(r'(?:\s|^)\d{1,3}(?:\.\d{3})+(,\d+)(?:\s|$)')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break
        start = match.start()
        end = match.end()
        result = result[0:start] + result[start:end].replace('.', '').replace(',', '.') + result[end:len(result)]
    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', result)
    return result

def replace_negative(string):
    return re.sub(rf'(\s)(-)(\d+)([{re.escape(punctuation)}])', r'\1negative \3\4', string)

def replace_roman(string):
    pattern = re.compile(rf'\s[IVXLCDM]{{2,}}[{re.escape(punctuation)}]')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break
        start = match.start()
        end = match.end()
        result = result[0:start + 1] + str(roman_to_int(result[start + 1:end - 1])) + result[end - 1:len(result)]
    return result

def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val

def hyphen_range_to(text):
    pattern = re.compile(r'(\d+)[-–](\d+)')
    result = pattern.sub(lambda x: x.group(1) + ' to ' + x.group(2), text)
    return result

def num_to_words(text):
    pattern = re.compile(r'\d+\.\d+|\d+')
    result = pattern.sub(lambda x: num2words.num2words(float(x.group())), text)
    return result

def replace_abbreviations(string):
    known_abbreviations = list(known_abbreviations_map.keys())
    pattern = re.compile(rf'(\b(?:{"|".join(map(re.escape, known_abbreviations))})\b)')
    result = pattern.sub(lambda x: known_abbreviations_map.get(x.group(1), replace_abbreviation(x.group(1))), string)
    return result

def replace_single_letters(string):
    # Match single letters that are truly isolated by spaces or punctuation, excluding those preceded by an apostrophe
    pattern = re.compile(rf'(?<![a-zA-Z\'])\b([A-Za-z])\b(?![a-zA-Z])')
    result = pattern.sub(lambda x: match_mapping(x.group(1)), string)
    #print("After replacing isolated single letters:", result)  # Debugging
    # Match single letters followed by numbers and insert a space between them,
    # except when enclosed in quotation marks or part of a variable assignment or expression
    pattern_letter_number = re.compile(r'(?<!["\'])(?<![=(])([A-Za-z])(\d+)(?!["\'])(?![)])')
    result = pattern_letter_number.sub(lambda x: ' '.join(list(x.group(1) + x.group(2))), result)
    #print("After replacing letters followed by numbers:", result)  # Debugging
    return result

def convert_years_to_words(text):
    def year_to_words(match):
        year = match.group(0)
        if 1000 <= int(year) <= 1999:
            return num2words.num2words(int(year[:2])) + " " + num2words.num2words(int(year[2:]))
        elif 2000 <= int(year) <= 2099:
            if year.endswith("00"):
                return num2words.num2words(int(year))
            else:
                return num2words.num2words(int(year[:2])) + " " + num2words.num2words(int(year[2:]))
        else:
            return num2words.num2words(int(year))
    pattern = re.compile(r'\b(1\d{3}|20[0-9]{2})\b')
    result = pattern.sub(year_to_words, text)
    return result

def replace_lowercase_abbreviations(string):
    pattern = re.compile(rf'(^|[\s(.\'\[<])(([a-z]\.){{1,4}})([{re.escape(punctuation)}]|$)')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break
        start = match.start()
        end = match.end()
        result = result[0:start] + replace_abbreviation(result[start:end].upper()) + result[end:len(result)]
    return result

def replace_abbreviation(string):
    if string in known_abbreviations_map:
        return known_abbreviations_map[string]
    result = ""
    for char in string:
        result += match_mapping(char)
    return result

def match_mapping(char):
    return alphabet_map.get(char, char)

# Testing the preprocess function
#original_text = "Hi again ABC! Drift with me, it's great to chat with you again, rather than a GPT! How are you doing today? Didn't you find some relaxation time or try out some of those calming techniques I mentioned earlier?"
#preprocessed_text = preprocess(original_text)
#print("Final preprocessed string:", preprocessed_text)
