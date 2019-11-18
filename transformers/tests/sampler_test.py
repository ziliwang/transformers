# coding=utf-8
from collections import namedtuple
import unittest

import pytest

from transformers import is_torch_available

if is_torch_available():
    import torch

    from transformers import (generate,
                              XLMConfig,
                              XLMWithLMHeadModel,
                              XLNetConfig,
                              XLNetLMHeadModel)
    from transformers.generate.sampler import SamplerForXLM, SamplerForXLNet, SamplerSingleStack
else:
    pytestmark = pytest.mark.skip("Require Torch")


class SamplerFactoryTest(unittest.TestCase):
    ModelStub = namedtuple("ModelStub", [])

    def test_creation_of_xlm_sampler(self):
        model_config = XLMConfig()
        model = XLMWithLMHeadModel(model_config)
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerForXLM)

    def test_creation_of_xlnet_sampler(self):
        model_config = XLNetConfig()
        model = XLNetLMHeadModel(model_config)
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerForXLNet)

    def test_creation_of_single_stack_model(self):
        model = self.ModelStub()
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerSingleStack)


class SamplerSingleStackTest(unittest.TestCase):
    class ModelStub(object):
        def __init__(self, batch_size=1):
            self.batch_size = batch_size

        def __call__(self, _):
            return (0.5 * torch.ones((self.batch_size, 2, 5)),)

    def test_output_length_no_prompt_batch_size_1(self):
        expected_length = 5
        model = self.ModelStub()
        sampler = generate.new_sampler(model)
        output = sampler.generate_sequence(length=expected_length)
        self.assertEqual(len(output), expected_length)

    def test_output_length_with_prompt_batch_size_1(self):
        prompt = [1, 2]
        generated_length = 5
        expected_length = 7
        model = self.ModelStub()
        sampler = generate.new_sampler(model)
        output = sampler.generate_sequence(length=generated_length, prompt=prompt)
        self.assertEqual(len(output), expected_length)


if __name__ == "__main__":
    unittest.main()
