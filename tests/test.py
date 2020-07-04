import unittest
from converter import WebMercator

class TestWebMercator(unittest.TestCase):
    def test_lat_factor(self):

        # test zoom level 0
        self.assertAlmostEqual(WebMercator.lat_factor(0, 0), 0.08627, places=5)
        self.assertEqual(WebMercator.lat_factor(256, 0), 1)
        self.assertAlmostEqual(WebMercator.lat_factor(512, 0), 0.08627, places=5)

        # test zoom level 1
        self.assertAlmostEqual(WebMercator.lat_factor(0, 1), 0.08627, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(256, 1), 0.39854, places=5)
        self.assertEqual(WebMercator.lat_factor(512, 1), 1)
        self.assertAlmostEqual(WebMercator.lat_factor(768, 1), 0.39854, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(1024, 1), 0.08627, places=5)

        # test zoom level 2
        self.assertAlmostEqual(WebMercator.lat_factor(0, 2), 0.08627, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(256, 2), 0.18787, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(512, 2), 0.39854, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(768, 2), 0.75494, places=5)
        self.assertEqual(WebMercator.lat_factor(1024, 2), 1)
        self.assertAlmostEqual(WebMercator.lat_factor(1280, 2), 0.75494, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(1536, 2), 0.39854, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(1792, 2), 0.18787, places=5)
        self.assertAlmostEqual(WebMercator.lat_factor(2048, 2), 0.08627, places=5)
