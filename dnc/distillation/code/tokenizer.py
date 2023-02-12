from razdel import tokenize as razdel_tokenizer

from dnc.distillation.code.embeddings import CustomEmbeddings
from dnc.distillation.code.log import get_logger

logger = get_logger(__name__)


# TODO: rewrite with vocab when emb=None
# TODO: rewrite to use it in collator
class Tokenizer:
    def __init__(self, emb, unk_token="<unk>", pad_token="<pad>"):
        if not emb:
            logger.warning("NO EMBEDDINGS!")
        self.emb = self._load_emb(emb)
    
    def tokenize(self, text):
        return [_.text for _ in razdel_tokenizer(text)]
    
    def __call__(self, text):
        return self.emb([_.text for _ in razdel_tokenizer(text)])
    
    def _load_emb(self, path):
        return CustomEmbeddings(path)