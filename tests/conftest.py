"""Fixtures for test doubles at the native MASTSEL boundary."""

from ast import literal_eval

import pytest

from hybrid_ao_psf import _mastsel


@pytest.fixture
def stub_mastsel(monkeypatch):
    """Replace the native constructor and decoder together."""

    def install(constructor, *, decoder=literal_eval):
        monkeypatch.setattr(_mastsel, "_load_mavis_lo", lambda: constructor)
        monkeypatch.setattr(_mastsel, "_load_mastsel_decoder", lambda: decoder)

    return install
