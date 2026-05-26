import unittest

from v1_pipeline.st2_extract import _find_text_span, parse_rebel_triples


class RebelParserTest(unittest.TestCase):
    def test_parse_single_triple(self):
        generated = "<s><triplet> heavy rain <subj> flooding in the city <obj> cause</s>"

        self.assertEqual(
            parse_rebel_triples(generated),
            [
                {
                    "subject": "heavy rain",
                    "object": "flooding in the city",
                    "raw_relation": "cause",
                }
            ],
        )

    def test_parse_multiple_triples(self):
        generated = (
            "<s><triplet> rain <subj> floods <obj> cause "
            "<triplet> wind <subj> damage <obj> cause</s>"
        )

        self.assertEqual(len(parse_rebel_triples(generated)), 2)

    def test_parse_malformed_output(self):
        self.assertEqual(parse_rebel_triples("no useful output"), [])

    def test_find_span_is_case_insensitive(self):
        self.assertEqual(_find_text_span("Heavy rain caused flooding.", "heavy rain"), (0, 10))


if __name__ == "__main__":
    unittest.main()
