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


if __name__ == "__main__":
    unittest.main()
