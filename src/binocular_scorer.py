import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityCalculator:
    def __init__(self, performer_name, observer_name=None, device=None, dtype=torch.float16, max_length=512):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_length = max_length

        # Load models
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.performer_model.eval()

        if observer_name is None:
            self.observer_model = self.performer_model
        else:
            self.observer_model = AutoModelForCausalLM.from_pretrained(
                observer_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True
            )
        self.observer_model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(performer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loss and softmax on GPU
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax = torch.nn.Softmax(dim=-1)

    def tokenize(self, batch):
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(self.device)
        return encodings

    @torch.inference_mode()
    def get_logits(self, encodings):
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        torch.cuda.synchronize()
        return observer_logits, performer_logits

    def perplexity(self, encoding, logits):
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

        loss = -(token_log_probs * shifted_attention_mask)
        ppl = (loss.sum(-1) / shifted_attention_mask.sum(-1)).exp()

        return ppl.detach().cpu().numpy()

    def cross_perplexity(self, observer_logits, performer_logits, encoding):
        shifted_observer_logits = observer_logits[..., :-1, :].contiguous()
        shifted_performer_logits = performer_logits[..., :-1, :].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        performer_probs = torch.softmax(shifted_performer_logits, dim=-1)
        observer_log_probs = torch.log_softmax(shifted_observer_logits, dim=-1)

        cross_entropy = -(performer_probs * observer_log_probs).sum(-1)
        xppl = (cross_entropy * shifted_attention_mask).sum(-1) / shifted_attention_mask.sum(-1)

        return xppl.exp().detach().cpu().numpy()

    def binocular_score(self, text):
        batch = [text] if isinstance(text, str) else text
        encodings = self.tokenize(batch)
        observer_logits, performer_logits = self.get_logits(encodings)
        ppl = self.perplexity(encodings, observer_logits)
        xppl = self.cross_perplexity(observer_logits, performer_logits, encodings)

        scores = (ppl / xppl).squeeze().tolist()
        return scores
