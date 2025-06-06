import stanza

nlp = stanza.Pipeline('en', processors='tokenize,ner')

BLACKLIST = ['abbot', 'toro']


def cleaner(text):
    doc = nlp(text)

    matches = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.type == "PERSON"]
    cleaned = []
    for match in matches:
        if match[0][0].upper() != match[0][0] or match[0].lower() in BLACKLIST:
            continue
        rem = ''
        for i in text[match[1]:match[2]]:
            if i == ' ':
                rem += ' '
            else:
                rem += "ӛ"
        text = text[:match[1]] + rem + text[match[2]:]
        cleaned.append(match[0])
    return text.split(' '), cleaned
