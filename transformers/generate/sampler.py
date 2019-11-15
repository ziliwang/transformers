import torch
import torch.nn.functional as F
from tqdm import trange


class Sampler(object):
    def __init__(self, temperature=1.0, k=0, p=0, repetition_penalty=1.0):
        self.k = k
        self.p = p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        self.do_apply_temperature = True if temperature > 0 else False
        self.do_apply_repetition_penalty = True if repetition_penalty > 1 else False

    def generate(self, model, sequence_length=1.0, prompt=[]):
        """ Generate a sequence of `length` tokens starting from the
        provided `prompt`.
        """
        prompt = torch.tensor(prompt, dtype=torch.long, device=None)
        generated_sequence = prompt
        with torch.no_grad():
            for _ in trange(sequence_length):
                outputs = model(input_ids=generated_sequence)
                next_token_logits = outputs[0][:, -1, :]
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits, generated_sequence
                )
                next_token = self.generate_one_token(next_token_logits)
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        return generated_sequence

    def generate_one_token(self, next_token_logits):
        logits = self.apply_temperature(next_token_logits)
        logits = self.apply_top_k_filter(logits)
        logits = self.apply_nucleus_filter(logits)
        return self.sample_one_token(logits)

    def apply_repetition_penalty(self, logits, generated_sequence):
        """ Apply a penalty to tokens that appear more than once in the
        generated sequence.

        .. Keskar, Nitish Shirish, et al. "Ctrl: A conditional transformer
           language model for controllable generation." arXiv preprint
           arXiv:1909.05858 (2019).
        """
        if self.do_apply_repetition_penalty:
            generated_token_idx = set(generated_sequence.to_list())
            for token_idx in generated_token_idx:
                logits[token_idx] /= self.repetition_penalty
        return logits

    def apply_temperature(self, logits):
        """ Shape the tokens' distribution through temperature. The higher the value
        of the temperature, the more skewed towards high probability events the
        distribution is.

        .. Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning.
        MIT press, 2016.
        """
        if self.do_apply_temperature:
            return logits / self.temperature
        return logits

    def apply_top_k_filter(self, logits):
        """ Use the probability distribution of the tokens to determine the set
        to be sampled from.  Specifically we select the set of size k such that
        the sum of its items' probabilities is maximum.

        .. Fan, Angela, Mike Lewis, and Yann Dauphin. "Hierarchical neural
        story generation." arXiv preprint arXiv:1805.04833 (2018).
        """
        if self.k > 0:
            indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        return logits

    def apply_nucleus_filter(self, logits):
        """ Use the probability distribution of the tokens to determine the set
        to be sampled from. Specifically, choose the smallest set such that the
        sum of its items' probabilities is greater than a number p in [0,1].

        .. Holtzman, Ari, et al. "The curious case of neural text
           degeneration." arXiv preprint arXiv:1904.09751 (2019).
        """
        if self.p > 0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        return logits

    def sample_one_token(self, logits):
        if self.do_apply_temperature:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return torch.argmax(logits, dim=-1).unsqueeze(-1)


class SamplerForXLM(Sampler):
    pass


class SamplerForXLNet(Sampler):
    pass


class SamplerForEncoderDecoder(Sampler):
    pass
