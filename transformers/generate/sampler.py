from collections import namedtuple
import warnings

import torch
import torch.nn.functional as F
from tqdm import trange

SamplerConfig = namedtuple("SamplerConfig", ["temperature", "k", "p", "repetition_penalty"])


def new_sampler(
    model, temperature=1.0, k=0, p=0, repetition_penalty=1.0, device=torch.device("cpu")
):
    """ Factory function that returns the appropriate sampler with regards
    to the model passed as a parameter.

    Only single stacks are currently supported.
    """

    sampler_config = SamplerConfig(
        temperature=temperature, k=k, p=p, repetition_penalty=repetition_penalty
    )

    # Single stack models
    MATCH_MODEL_SAMPLER = {
        "XLNetLMHeadModel": SamplerForXLNet,
        "XLMWithLMHeadModel": SamplerForXLM,
    }

    sampler = MATCH_MODEL_SAMPLER.get(model.__class__.__name__, None)
    if not sampler:
        sampler = SamplerSingleStack
    return sampler(model, sampler_config, device)


class Sampler(object):
    def __init__(self, config, device):
        self.k = config.k
        self.p = config.p
        self.temperature = config.temperature
        self.repetition_penalty = config.repetition_penalty

        self.do_apply_temperature = True if config.temperature > 0 else False
        self.do_apply_repetition_penalty = True if config.repetition_penalty > 1 else False

        if self.p > 1:
            warnings.warn(
                "p is a probability; its value must lie between 0 and 1, got {}. Ignored.".format(
                    self.p
                )
            )

        self.device = device

    def generate_sequence(self, length=1, prompt=[]):
        """ Generate a sequence of `length` tokens starting from the
        provided `prompt`.
        """
        raise NotImplementedError

    def generate_one_token(self, next_token_logits, past_sequence):
        logits = self.apply_repetition_penalty(next_token_logits, past_sequence)
        logits = self.apply_temperature(logits)
        logits = self.apply_top_k_filter(logits)
        logits = self.apply_nucleus_filter(logits)
        return self.sample_one_token(logits)

    def apply_repetition_penalty(self, logits, past_sequence):
        """ Apply a penalty to tokens that appear more than once in the
        generated sequence.

        .. Keskar, Nitish Shirish, et al. "Ctrl: A conditional transformer
           language model for controllable generation." arXiv preprint
           arXiv:1909.05858 (2019).
        """
        if self.do_apply_repetition_penalty:
            generated_token_idx = set(past_sequence[0].tolist())
            for token_idx in generated_token_idx:
                logits[0, token_idx] /= self.repetition_penalty
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
            try:
                indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")
            except RuntimeError:
                vocabulary_size = logits.size(-1)
                raise RuntimeError(
                    "the value of k provided ({}) must be smaller than the size of the vocabulary ({})".format(
                        self.k, vocabulary_size
                    )
                )

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
            sorted_probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

            # Remove tokens with cumulative probability above the threshold,
            # but keep the first token above the threshold.
            sorted_indices_to_remove = cumulative_probabilities > self.p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
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
        self.model = model
        super(SamplerSingleStack, self).__init__(config, device)

    def generate_sequence(self, length=1, prompt=[], **model_kwargs):
        prompt = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_sequence = prompt
        with torch.no_grad():
            for _ in trange(length):
                outputs = self.forward_pass(generated_sequence, **model_kwargs)
                next_tokens_logits = outputs[0][:, -1, :]
                next_tokens = self.generate_one_token(next_tokens_logits, generated_sequence)
                generated_sequence = torch.cat((generated_sequence, next_tokens), dim=1)

        return generated_sequence.squeeze(0).tolist()

    def forward_pass(self, input_ids, **model_kwargs):
        return self.model(input_ids, **model_kwargs)


class SamplerForXLM(SamplerSingleStack):
    def __init__(self, model, config, device):
        super(SamplerForXLM, self).__init__(model, config, device)

    def forward_pass(self, input_ids, **model_kwargs):
        mask_token = model_kwargs.pop("mask_token", None)
        language = model_kwargs.pop("language", None)
        input_ids = self._add_mask_token(input_ids, mask_token)
        language_embeddings = self._create_language_embeddings(input_ids, language)
        outputs = self.model(input_ids, langs=language_embeddings)
        return outputs

    def _add_mask_token(self, sequence, mask_token_id):
        """ Append a [MASK] token at the end of the sequence that the MLM model
        is going to try to predict.
        """
        if mask_token_id:
            return torch.cat(
                (
                    sequence,
                    torch.full((1, 1), mask_token_id, dtype=torch.long, device=self.device),
                ),
                dim=1,
            )
        return sequence

    def _create_language_embeddings(self, sequence, language):
        if language:
            return torch.tensor([language] * sequence.shape[1], device=self.device).view(1, -1)
        return None


class SamplerForXLNet(SamplerSingleStack):
    def __init__(self, model, config, device):
        super(SamplerForXLNet, self).__init__(model, config, device)

    def forward_pass(self, input_ids):
        input_ids = self._add_dummy_token(input_ids)
        attention_mask = self._create_attention_mask(input_ids)
        target_mapping = self._create_target_mapping(input_ids)
        return self.model(
            input_ids, perm_mask=attention_mask, target_mapping=target_mapping
        )

    def _add_dummy_token(self, sequence):
        dummy = torch.zeros((sequence.shape[0], 1), dtype=torch.long, device=self.device)
        return torch.cat((sequence, dummy), dim=1)

    def _create_attention_mask(self, sequence):
        mask = torch.zeros(
            (sequence.shape[0], sequence.shape[1], sequence.shape[1]), dtype=torch.float, device=self.device
        )
        mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        return mask

    def _create_target_mapping(self, sequence):
        target_mapping = torch.zeros(
            (sequence.shape[0], 1, sequence.shape[1]), dtype=torch.float, device=self.device
        )
        target_mapping[0, 0, -1] = 1.0  # predict last token
        return target_mapping
