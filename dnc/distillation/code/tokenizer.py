import re

from razdel import tokenize as razdel_tokenizer

from dnc.distillation.code.embeddings import CustomEmbeddings
from dnc.distillation.code.log import get_logger

logger = get_logger(__name__)


# TODO: rewrite with vocab when emb=None
# TODO: rewrite to use it in collator
class Tokenizer:
    def __init__(
        self,
        emb,
        # unk_token="<unk>",
        # pad_token="<pad>",
        stopwords=None,
        isalpha=True,
        lower=True,
    ):
        if not emb:
            logger.warning("NO EMBEDDINGS!")
        self.emb = self._load_emb(emb)
        if stopwords:
            with open(stopwords, encoding="utf-8") as file:
                stopwords = {x.strip() for x in file.readlines()}
                self.stopwords = stopwords
        else:
            self.stopwords = False
        self.isalpha = isalpha
        self.isalpha_re = re.compile(
            "[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]", flags=re.IGNORECASE
        )
        self.lower = lower

    def tokenize(self, text):
        return [_.text for _ in razdel_tokenizer(text)]

    def __call__(self, text):
        if self.lower:
            text = text.lower()
        text = self.tokenize(text)
        if self.stopwords:
            text = self.remove_stopwords(text)
        if self.isalpha:
            text = self.remove_non_alpha(text)
        return self.emb(text)

    def embed(self, seq):
        return self.emb(seq)

    def remove_stopwords(self, seq):
        return [t for t in seq if t.lower() not in self.stopwords]

    def remove_non_alpha(self, seq):
        return [t for t in seq if self.isalpha_re.search(t)]

    def _load_emb(self, path):
        return CustomEmbeddings(path)
