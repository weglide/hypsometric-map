import pytest
from src.converter import WebMercator


def test_web_mercator():
    # test zoom level 0
    assert WebMercator.lat_factor(256, 0) == 1
    assert abs(WebMercator.lat_factor(0, 0) - 0.08627) <= 0.0001
    assert abs(WebMercator.lat_factor(512, 0) - 0.08627) <= 0.0001

    # test zoom level 1
    assert WebMercator.lat_factor(512, 1) == 1
    assert abs(WebMercator.lat_factor(0, 1) - 0.08627) <= 0.0001
    assert abs(WebMercator.lat_factor(256, 1) - 0.39854) <= 0.0001
    assert abs(WebMercator.lat_factor(768, 1) - 0.39854) <= 0.0001
    assert abs(WebMercator.lat_factor(1024, 1) - 0.08627) <= 0.0001

    # test zoom level 2
    assert WebMercator.lat_factor(1024, 2) == 1
    assert abs(WebMercator.lat_factor(0, 2) - 0.08627) <= 0.0001
    assert abs(WebMercator.lat_factor(256, 2) - 0.18787) <= 0.0001
    assert abs(WebMercator.lat_factor(512, 2) - 0.39854) <= 0.0001
    assert abs(WebMercator.lat_factor(768, 2) - 0.75494) <= 0.0001
    assert abs(WebMercator.lat_factor(1280, 2) - 0.75494) <= 0.0001
    assert abs(WebMercator.lat_factor(1536, 2) - 0.39854) <= 0.0001
    assert abs(WebMercator.lat_factor(1792, 2) - 0.18787) <= 0.0001
    assert abs(WebMercator.lat_factor(2048, 2) - 0.08627) <= 0.0001
