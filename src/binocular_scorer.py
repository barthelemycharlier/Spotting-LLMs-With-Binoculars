import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityCalculator:
    def __init__(self, performer_name, observer_name=None, device=None, dtype=torch.float16, max_length=1024):
        """
        Class for computing perplexity and cross-perplexity.
        performer_name: model used to generate text
        observer_name: model used to evaluate text (if None, same as performer)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Performer model (generative)
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype
        )
        self.performer_model.eval()

        # Observer model (evaluation)
        if observer_name is None:
            self.observer_model = self.performer_model
        else:
            self.observer_model = AutoModelForCausalLM.from_pretrained(
                observer_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype
            )
        self.observer_model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(performer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length


    def tokenize(self, texts):
        """
        Tokenizes a single text or a list of texts into tensors.
        Ensures all sequences in a batch have the same length with padding.
        """
        if isinstance(texts, str):
            # single text
            return self.tokenizer(texts, return_tensors="pt").to(self.device)
        elif isinstance(texts, list):
            # batch of texts
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,       # pad sequences to the same length
                truncation=True,    # truncate sequences that are too long
                max_length=self.max_length     # limit max length
            ).to(self.device)
        else:
            raise ValueError("Input should be a string or a list of strings")

    @torch.inference_mode()
    def perplexity(self, text):
        """
        Compute the standard perplexity of text under the performer model.
        """
        inputs = self.tokenize(text)
        input_ids = inputs["input_ids"]

        outputs = self.performer_model(input_ids, labels=input_ids)
        loss = outputs.loss  # average cross-entropy over tokens (computed by the transformers library)
        return torch.exp(loss).item()

    @torch.inference_mode()
    def cross_perplexity(self, text):
        """
        Compute the cross-perplexity of text: performer output evaluated by observer model.
        """
        inputs = self.tokenize(text)
        input_ids = inputs["input_ids"]

        outputs = self.observer_model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

    @torch.inference_mode()
    def binoculars_score(self, texts):
        """
        Compute the Binoculars normalized score for a single text or a batch of texts.
        Returns a list of scores.
        """
        if isinstance(texts, str):
            texts = [texts]  # make it a list for uniform processing

        scores = []
        for text in texts:
            ppl = self.perplexity(text)
            x_ppl = self.cross_perplexity(text)
            score = torch.log(torch.tensor(ppl)) / torch.log(torch.tensor(x_ppl))
            scores.append(score.item())  # convert scalar tensor to float

        return scores

