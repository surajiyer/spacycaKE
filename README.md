# spacycaKE: Keyphrase Extraction for spaCy
[spaCy v2.0](https://spacy.io/usage/v2) extension and pipeline component for Keyphrase Extraction methods meta data to `Doc` objects.

## Installation
`spacycaKE` requires `spacy` v2.0.0 or higher and `spacybert` v1.0.0 or higher.

## Usage
```
import spacy
from spacycake import BertKeyphraseExtraction as bake
nlp = spacy.load('en')
```

Then use `bake` as part of the spacy pipeline,
```
cake = bake(nlp, from_pretrained='bert-base-cased', top_k=3)
nlp.add_pipe(cake, last=True)
```

Extract the keyphrases.
```
doc = nlp("This is a test but obviously you need to place a bigger document here to extract meaningful keyphrases")
print(doc._.extracted_phrases)  # <-- List of 3 keyphrases
```

## Available attributes
The extension sets attributes on the `Doc` object. You can change the attribute names on initializing the extension.
| | | |
|-|-|-|
| `Doc._.bert_repr` | `torch.Tensor` | Document BERT embedding |
| `Doc._.noun_phrases` | `List[str]` | List of the candidate phrases from the document |
| `Doc._.extracted_phrases` | `List[str]` | List of the final extracted keyphrases |

## Settings
On initialization of `bake`, you can define the following:

| name | type | default | description |
|-|-|-|-|
| `nlp` | `spacy.lang.(...)` | - | Only used to get the language vocabulary to initialize the phrase matcher |
| `from_pretrained` | `str` | `None` | Path to Bert model directory or name of HuggingFace transformers pre-trained Bert weights, e.g., `bert-base-cased` |
| `attr_names` | `Tuple[str]` | `('bert_repr', 'noun_phrases', 'extracted_phrases')` | Name of the various available attributes set to the `._` property (in order) |
| `force_extension` | `bool` | `True` | A boolean value to create the same 'Extension Attribute' upon being executed again |
| `top_k` | `int` | 5 | Max number of extracted phrases |
| `mmr_lambda` | `float` | .5 | Maximum Marginal Relevance lambda parameter. Used to control diversity of extracted keyphrases. Closer to 1., the more diverse the results. Closer to 0., the more similar the extracted phrases will be to the source document. |
| `kws` | `kwargs` | - | More keyword arguments to supply to `spacybert.BertInference()` |

## Roadmap
This extension is still experimental. Possible future updates include:
* Adding other keyphrase extraction methods.