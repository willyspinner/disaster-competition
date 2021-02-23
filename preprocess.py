import nltk
import re
import string
import nltk
from nltk.corpus import stopwords

from abbv import abbreviations

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

url_re = re.compile(r'https?://\S+|www\.\S+')
html_re = re.compile(r'<.*?>')
at_re = re.compile(r'@\S+')
hashtag_re = re.compile(r'#(?P<t>\S+)')
punct_space_re = re.compile(r'[-_]')
multispace_re = re.compile(r'\s{2,}')
punct_table = str.maketrans('', '', string.punctuation)

def remove_URL(text):
    return url_re.sub('', text)

def remove_HTML(text):
    return html_re.sub('', text)

def remove_mention(text):
    return at_re.sub('', text)


def remove_non_ascii_chars(text):
    return ''.join([c for c in text if c in string.printable])


# NOTE: can't remove hashtags since they may contain useful keywords.
# e.g. #man #airport #airplane #aircraft #aeroplane #runway #accident #freaky
def remove_hashtag(text):
    return hashtag_re.sub('\g<t>', text)


def remove_punctuations(text):
    text = punct_space_re.sub(' ', text)
    return text.translate(punct_table)


def sub_abbreviations(text):
    def abbrev_word(w):
        return abbreviations[w] if w in abbreviations.keys() else w
    return ' '.join([abbrev_word(w) for w in text.split(' ')])

def remove_stopwords(text):
    return ' '.join([w for w in text.split(' ') if w not in stopwords_set])


def remove_multiple_space(text):
    return multispace_re.sub(' ', text)


def preprocess_text(text):
    text = text.lower()
    text = remove_URL(text)
    text = remove_HTML(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    text = remove_non_ascii_chars(text)
    text = sub_abbreviations(text)
    text = remove_multiple_space(text)
    return text



