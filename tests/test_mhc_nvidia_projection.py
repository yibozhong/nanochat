"""Run the same independent oracle/model gates for the NVIDIA backend."""
import pytest


@pytest.mark.parametrize('backend', ['nvidia', 'nvidia_precise'])
@pytest.mark.parametrize('width', [128, 512])
def test_nvidia_projection_against_fp64_and_official(backend, width):
    from test_mhc_torch_projection import test_projection_against_fp64_and_official
    test_projection_against_fp64_and_official(backend, width)


@pytest.mark.parametrize('backend', ['nvidia', 'nvidia_precise'])
def test_nvidia_model_causality_cache_and_all_gradients(backend):
    from test_mhc_torch_projection import test_torch_projection_model_causality_cache_and_all_gradients
    test_torch_projection_model_causality_cache_and_all_gradients(backend)
