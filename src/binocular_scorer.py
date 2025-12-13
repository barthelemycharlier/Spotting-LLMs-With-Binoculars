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
    def binoculars_score(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        # tokenize batch
        inputs = self.tokenize(texts)
        input_ids = inputs["input_ids"]

        # --- Performer PPL ---
        with torch.no_grad():
            logits = self.performer_model(input_ids).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_ids.size(0), -1).mean(dim=1)
            ppl = torch.exp(loss)

        # --- Observer cross-PPL ---
        with torch.no_grad():
            logits = self.observer_model(input_ids).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_ids.size(0), -1).mean(dim=1)
            x_ppl = torch.exp(loss)

        scores = torch.log(ppl) / torch.log(x_ppl)
        return scores.tolist()
