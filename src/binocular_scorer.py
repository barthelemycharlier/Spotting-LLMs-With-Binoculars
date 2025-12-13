import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityCalculator:
    def __init__(self, performer_name, observer_name=None, device=None, dtype=torch.bfloat16):
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

    def tokenize(self, text):
        """Tokenize text and move to the appropriate device."""
        return self.tokenizer(text, return_tensors="pt").to(self.device)

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
    def binoculars_score(self, text):
        """
        Compute the Binoculars normalized score:
        Score = log(PPL_performer) / log(X-PPL_performer->observer)
        """
        ppl = self.perplexity(text)
        x_ppl = self.cross_perplexity(text)
        return torch.log(torch.tensor(ppl)) / torch.log(torch.tensor(x_ppl))



performer_name = "tiiuae/falcon-7b"
observer_name = "tiiuae/falcon-7b-instruct"

calc = PerplexityCalculator(performer_name, observer_name)

prompt = "Can you write a few sentences about a capybara that is an astrophysicist?"

# Compute standard perplexity
ppl = calc.perplexity(prompt)

# Compute cross-perplexity
x_ppl = calc.cross_perplexity(prompt)

# Compute normalized Binoculars score
score = calc.binoculars_score(prompt)

print("Perplexity (performer):", ppl)
print("Cross-perplexity (performer -> observer):", x_ppl)
print("Binoculars score:", score.item())
