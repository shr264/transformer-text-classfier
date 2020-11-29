import re
import nltk
import string
import logging
import inflect
import wordninja
import unicodedata as ud

from typing import List
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import RegexpTokenizer, word_tokenize, sent_tokenize

# Use inflect for some operations on text
p = inflect.engine()

logger = logging.getLogger(__name__)


def remove_non_ascii(word: str) -> str:
    """Remove non-ASCII characters from list of tokenized words"""
    return ud.normalize('NFKD', word).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(word: str) -> str:
    """Convert all characters to lowercase from list of tokenized words"""
    return word.lower()


def remove_punctuation(word: str) -> str:
    """Remove punctuation from list of tokenized words"""

    if any(char.isdigit() for char in word):
        return word

    return word.translate(str.maketrans('', '', string.punctuation))


def replace_numbers(word: str) -> str:
    """Replace all interger occurrences in list of tokenized words with textual representation"""

    # Remove dollar signs if they are still present
    for char in ['$', '&', '#', '@', ';', '.', ':', ',']:
        word = word.replace(char, '') if char in word else word

    # Catches measurements, ex. - 34x30 -> thirty four thirty
    if 'x' in word:
        size_measurement = True
        words = word.split('x')
        if all(_word.isdigit() for _word in words):
            _words = []
            for _word in words:
                try:
                    _word = p.number_to_words(_word)
                    _word = _word.replace(',', '').replace('-', ' ')
                    _words.append(_word)
                except:
                    _words.append(word)
            return " ".join(_words)

    # Convert numbers to words
    if word.isdigit():
        try:
            word = p.number_to_words(word)
            word = word.replace(',', '').replace('-', ' ')
            return word
        except:
            return word
    else:
        return word


def remove_stopwords(words: List[str]) -> List[str]:
    """Remove stop words from list of tokenized words"""
    new_words = [
        word for word in words if word not in stopwords.words('english')
    ]
    return new_words


def stem_words(words: List[str]) -> List[str]:
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = [stemmer.stem(word) for word in words]
    return stems


def lemmatize_verbs(words: List[str]) -> List[str]:
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return lemmas


def stem_and_lemmatize(words: List[str]) -> (List[str], List[str]):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


def standardize(word: str, replace_nums: bool = True) -> str:
    word = word.strip('\n')
    word = remove_non_ascii(word)
    word = to_lowercase(word)
    if replace_nums:
        word = replace_numbers(word)  # numbers to text
    word = remove_punctuation(word)
    return word


def normalize(words: str, word_split: bool = True, replace_nums: bool = True) -> str:
    """Performs a series of cleaning / splitting steps on input words"""
    """words should be a string that can be tokenized with ' ' separator"""

    normalized = []
    for word in words.split(" "):

        # python normalize strings first
        word = ud.normalize('NFC', word)

        try:
            normal = standardize(word, replace_nums=replace_nums)
        except Exception as E:
            logger.info(
                f'Unable to perform vanilla normalization of word - {word}')
            logger.warning(f'Error traceback - {E}')
            normal = word.lower().replace('&', 'and').strip(",").strip("/")
        except:
            normalized.append(word)
            continue

        # command to ignore word-splitting
        if not word_split:
            normalized.append(normal)
            continue

        # if a recognized word, dont try to split
        if wordnet.synsets(normal):
            normalized.append(normal)
            continue

        # try to split the word with ninja
        splits = wordninja.split(normal)

        # clean the split words
        normalized_splits = list(map(standardize, splits))

        # check if the split words are real words
        valid_words = [
            werd for werd in normalized_splits if wordnet.synsets(werd)]

        # only accept the split if all words are real words
        if len(valid_words) == len(splits):
            normalized += normalized_splits
        else:
            normalized.append(normal)

    if not len(normalized):
        return words
    else:
        return " ".join(normalized)
