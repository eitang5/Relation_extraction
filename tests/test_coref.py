import unittest
from unittest.mock import patch

from v1_pipeline.coref import coref_article


class CorefArticleTest(unittest.TestCase):
    def test_empty_predictions_return_no_clusters(self):
        class EmptyCoref:
            def predict(self, texts):
                return []

        with patch("v1_pipeline.coref._load", return_value=EmptyCoref()):
            self.assertEqual(coref_article("No clusters here."), [])


if __name__ == "__main__":
    unittest.main()
