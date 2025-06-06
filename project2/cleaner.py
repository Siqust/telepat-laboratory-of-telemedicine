from natasha import NamesExtractor, MorphVocab, AddrExtractor
from cleaner2 import cleaner as cleaner_eng
from blacklist import BLACKLIST

morph_vocab = MorphVocab()
names_extractor = NamesExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)

alphabet = [chr(i) for i in range(ord("а"), ord("я") + 1)] + [chr(i) for i in range(ord("a"), ord("z") + 1)]


def cleaner(text):
    s = []
    for i in text:
        if i.isdigit() and int(i) > 10 ** 4:
            s.append('ӛ' * len(i))
        else:
            s.append(i)

    text = ' '.join(s)

    print("CLEANING:", text)
    cleaned = []
    for match in names_extractor(text):
        words = [match.fact.first, match.fact.middle, match.fact.last]

        for word in words:
            if word is not None:
                if word.lower() not in BLACKLIST and len(word) != 2 and word[0].upper() == word[0]:
                    cleaned.append(match)
                    rem = ""
                    for i in text[match.start:match.stop]:
                        if i == ' ':
                            rem += ' '
                        else:
                            rem += "ӛ"
                    text = text[:match.start] + rem + text[match.stop:]
                    break
                else:
                    try:
                        if len(word) == 1 and text[match.stop + 1] == '.' and word.lower() in alphabet and len(
                                [i for i in words if i not in [None, word]]) == 0 and word[0].upper() == word[0]:
                            cleaned.append(match)
                            text = text[:match.start] + "ӛ" * (
                                    match.stop - match.start - text[match.start:match.stop].count(' ')) + text[
                                                                                                          match.stop:]
                            break
                    except:
                        pass
                    break
    text, cleaned2 = cleaner_eng(text)
    return text, cleaned + cleaned2

# import stanza
# def english_cleaner():
