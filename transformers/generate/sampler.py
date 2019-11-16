from collections import namedtuple

import torch
import torch.nn.functional as F
from tqdm import trange


SamplerConfig = namedtuple("SamplerConfig", ["temperature", "k", "p", "repetition_penalty"])


def new_sampler(model, temperature=1.0, k=0, p=0, repetition_penalty=1.0, device=torch.device("cpu")):
    """ Factory function that returns the appropriate sampler with regards
    to the model passed as a parameter.

    Only single stacks are currently supported.
    """

    MATCH_MODEL_SAMPLER = {
        "XLNetLMHeadModel": SamplerForXLNet,
        "XLMWithLMHeadModel": SamplerForXLM,
    }

    sampler_config = SamplerConfig(
        temperature=temperature, k=k, p=p, repetition_penalty=repetition_penalty
    )

    sampler = MATCH_MODEL_SAMPLER.get(model.__class__.__name__, None)
    if not sampler:
        sampler = SamplerSingleStack

    return sampler(model, sampler_config, device)


class Sampler(object):
    def __init__(self, model, config, device):
        self.k = config.k
        self.p = config.p
        self.temperature = config.temperature
        self.repetition_penalty = config.repetition_penalty

        self.do_apply_temperature = True if config.temperature > 0 else False
        self.do_apply_repetition_penalty = True if config.repetition_penalty > 1 else False

        self.model = model
        self.device = device

    def generate_sequence(self, length=1, prompt=[], **model_kwargs):
        """ Generate a sequence of `length` tokens starting from the
        provided `prompt`.
        """
        raise NotImplementedError

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


# -------------------------------------------------------------
# Samplers for single stack models
# -------------------------------------------------------------


class SamplerSingleStack(Sampler):
    def __init__(self, model, config, device):
        super(SamplerSingleStack, self).__init__(model, config, device)

    def generate_sequence(self, length=1, prompt=[], **model_kwargs):
        prompt = torch.tensor(prompt, dtype=torch.long, device=self.device)
        generated_sequence = prompt
        with torch.no_grad():
            for _ in trange(length):
                outputs = self.forward_pass(generated_sequence, **model_kwargs)
                next_token_logits = outputs[0][:, -1, :]
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits, generated_sequence
                )
                next_token = self.generate_one_token(next_token_logits)
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        return generated_sequence

    def forward_pass(self, input_ids, **model_kwargs):
        return self.model(input_ids, **model_kwargs)


class SamplerForXLM(SamplerSingleStack):
    def __init__(self, model, config, device):
        super(SamplerForXLM, self).__init__(model, config, device)

    def forward_pass(self, input_ids, **model_kwargs):
        mask_token = model_kwargs.get("mask_token", None)
        lang = model_kwargs.get("lang", None)
        input_ids = self._add_dummy_token(input_ids, mask_token)
        langs = self._create_langs(input_ids, lang)
        outputs = self.model(input_ids=input_ids, langs=langs)
        return outputs

    def _add_dummy_token(self, sequence, token_id):
        if token_id:
            return torch.cat(
                (sequence, torch.full((1, 1), token_id, dtype=torch.long, device=self.device)),
                dim=1,
            )
        return sequence

    def _create_langs(self, sequence, lang):
        if lang:
            return torch.tensor([lang] * sequence.shape[1], device=self.device).view(1, -1)
        return sequence


class SamplerForXLNet(SamplerSingleStack):
    def __init__(self, model, config, device):
        super(SamplerForXLNet, self).__init__(model, config, device)

    def forward_pass(self, model, input_ids, **model_kwargs):
        input_ids = self._add_dummy_token(input_ids)
        attention_mask = self._create_attention_mask(input_ids)
        target_mapping = self._create_target_mapping(input_ids)
        return model(input_ids, perm_mask=attention_mask, langs=target_mapping)

    @staticmethod
    def _add_dummy_token(sequence):
        return torch.cat((sequence, 0), dim=1)

    def _create_attention_mask(self, sequence):
        mask = torch.zeros(
            (1, sequence.shape[1], sequence.shape[1]), dtype=torch.float, device=self.device
        )
        mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        return mask

    def _create_target_mapping(self, sequence):
        target_mapping = torch.zeros(
            (1, 1, sequence.shape[1]), dtype=torch.float, device=self.device
        )
        target_mapping[0, 0, -1] = 1.0  # predict last token
        return target_mapping
