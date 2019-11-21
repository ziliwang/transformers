# -------------------------------------------------------------
# Samplers for encoder-decoder models
# -------------------------------------------------------------


class SamplerEncoderDecoder(Sampler):
    """ Generic sampler for encoder-decoder models.
    """

    def __init__(self, model, config, device):
        self.encoder = model.encoder
        self.decoder = model.decoder
        super(SamplerEncoderDecoder, self).__init__(config, device)

    def generate_sequence(self, encoder_input, length=1, prompt=[], **model_kwargs):
        encoder_kwargs, decoder_kwargs = self._parse_model_kwargs(**model_kwargs)
        encoder_hidden_states = self.forward_pass_encoder(encoder_input, **encoder_kwargs)

        prompt = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_sequence = prompt
        with torch.no_grad():
            for _ in trange(length):
                outputs = self.forward_pass_decoder(
                    generated_sequence,
                    encoder_hidden_states=encoder_hidden_states,
                    **decoder_kwargs,
                )
                next_tokens_logits = outputs[0][:, -1, :]
                next_tokens = self.generate_one_token(
                    next_tokens_logits, generated_sequence
                )
                generated_sequence = torch.cat((generated_sequence, next_tokens), dim=1)

        return generated_sequence.squeeze(0).tolist()

    def forward_pass_decoder(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def forward_pass_encoder(self, input_ids, **kwargs):
        """ Passes through the encoder and returns the last
        hidden state layer.
        """
        outputs = self.encoder(input_ids, **kwargs)
        return outputs[0]

    @staticmethod
    def _parse_model_kwargs(kwargs):
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_") and not argument.startswith("decoder_")
        }
        decoder_kwargs = kwargs_common.copy()
        encoder_kwargs = kwargs_common.copy()
        encoder_kwargs.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        decoder_kwargs.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )
        decoder_kwargs["encoder_attention_mask"] = encoder_kwargs.get(
            "attention_mask", None
        )

        return encoder_kwargs, decoder_kwargs
