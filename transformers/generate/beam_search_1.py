class Beam(object):
    def __init__(self, config, device):
        raise NotImplementedError

    def grow_one_step(self, log_probabilities):

        # The number of beams changes as some beams finish so we define _B
        vocab_size = log_probabilities.size(-1)
        _B = log_probabilities.size(0) // self.beam_size

        log_probabilities = self._enforce_min_length(log_probabilities)
        log_probabilities = self._block_repeating_trigrams(log_probabilities)
        raise NotImplementedError
    
    def _block_repeating_trigrams(self):
        raise NotImplementedError

    def _enforce_min_length(self):
        raise NotImplementedError

    def _enforce_max_length(self):
        raise NotImplementedError

    def _apply_length_penalty(self):
        raise NotImplementedError


class BeamSearchSingleStack(BeamSearch):
    pass

class BeamSearchEncoderDecoder(BeamSearch):
    pass
