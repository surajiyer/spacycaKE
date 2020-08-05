from .about import __version__
from spacybert import BertInference
from spacy.tokens import Doc
from spacy.matcher import Matcher
from spacy.util import filter_spans
import torch
from typing import Tuple


class BertKeyphraseExtraction:
    """
    Based on the paper "Simple Unsupervised Keyphrase Extraction using Sentence Embedding"
    """

    name = 'bert_keyphrase_extraction'
    noun_chunk_pattern = [
        {'POS': 'ADJ', 'OP': '*'},
        {'POS': 'NOUN', 'OP': '+'}]

    def __init__(
            self, nlp, *, from_pretrained: str,
            attr_names: Tuple[str] = ('bert_repr', 'noun_phrases', 'extracted_phrases'),
            force_extension: bool = True, top_k: int = 5, mmr_lambda: float = .5,
            **kws):
        """
        Keyword arguments only after first argument!

        Params
        ------
        nlp: spacy Language
            Spacy language object

        from_pretrained: str, None
            Path to Bert model directory or name of HuggingFace transformers
            pre-trained Bert weights, e.g., 'bert-base-uncased'

        attr_names: Tuple[str]
            In order:
                1. Name of the BERT embedding attribute, default = '._.bert_repr'
                2. Name of the candidate phrases attribute, default = '._.noun_phrases'
                3. Name of the top-k extracted phrases attribute, default = '._.extracted_phrases'

        force_extension: bool
            A boolean value to create the same 'Extension Attribute' upon being
            executed again

        top_k: int
            Select the top-k candidate phrases

        mmr_lambda: float [0..1]
            Lambda parameter for the maximum marginal relevance re-ranking of keyphrases

        kws:
            More keywords arguments to supply to spacybert.BertInference()
        """
        assert len(attr_names) == 3
        assert kws.get('pooling_strategy', 0) is not None
        assert isinstance(top_k, int) and top_k > 0
        assert 0. <= mmr_lambda <= 1.
        self.attr_names = attr_names
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

        # Load noun chunks parser
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("noun_chunk", None, self.noun_chunk_pattern)

        # Load bert inference spacy extensions
        if from_pretrained:
            BertInference(
                from_pretrained=from_pretrained,
                attr_name=attr_names[0],
                set_extension=True, force_extension=force_extension, **kws)
        else:
            import warnings
            warnings.warn(
                'from_pretrained not supplied. Will continue assuming'
                + f'Doc._.{attr_names[0]} is available.', RuntimeWarning)

        Doc.set_extension(attr_names[1], getter=self._get_candidate_phrases, force=force_extension)
        Doc.set_extension(attr_names[2], default=None, force=force_extension)

    def _get_candidate_phrases(self, doc: Doc):
        return filter_spans([doc[start:end] for _, start, end in self.matcher(doc)])

    def __call__(self, doc: Doc):
        # Compute similarity between document and phrases
        phrases = self._get_candidate_phrases(doc)
        doc_embedding = getattr(doc._, self.attr_names[0])
        phrases_embeddings = torch.stack(list(
            map(lambda p: getattr(p._, self.attr_names[0]), phrases)))
        doc_phrase_similarity = torch.matmul(doc_embedding, phrases_embeddings.transpose(0, 1))

        # Rank phrases based on similarity ascending
        # and limit to top-k phrases
        indices = doc_phrase_similarity.argsort(descending=True)
        phrases_embeddings = phrases_embeddings[indices]
        phrases = [phrases[i] for i in indices.tolist()]
        R = list(range(len(phrases)))
        S = []

        # Re-rank based on Maximum Marginal Relevance score
        while len(R) > 0:
            first_part = torch.matmul(doc_embedding, phrases_embeddings[R].transpose(0, 1))
            second_part = torch.matmul(
                phrases_embeddings[R],
                phrases_embeddings[S].transpose(0, 1)).max(dim=1).values
            scores = (self.mmr_lambda * first_part) - ((1 - self.mmr_lambda) * second_part)
            phrase_to_add = R[scores.argmax()]
            R.remove(phrase_to_add)
            S.append(phrase_to_add)

        doc._.set(self.attr_names[2], [phrases[i] for i in S[:self.top_k]])

        return doc
