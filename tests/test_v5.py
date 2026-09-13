import unittest
from novelai_mcp.v5 import build_payload, sse_images


class V5Tests(unittest.TestCase):
    def test_character_positive_and_negative_prompts_stay_separate(self):
        p = build_payload('bookstore', characters=[{'prompt': 'white jacket', 'negative_prompt': 'blue shirt', 'x': .2, 'y': .5}, {'prompt': 'blue shirt', 'x': .8, 'y': .5}], seed=7)
        self.assertEqual(p['parameters']['v4_prompt']['caption']['char_captions'][0]['char_caption'], 'white jacket')
        self.assertEqual(p['parameters']['v4_negative_prompt']['caption']['char_captions'][0]['char_caption'], 'blue shirt')
        self.assertEqual(p['parameters']['seed'], 7)

    def test_unavailable_v5_features_are_refused_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, '不支持 infill'):
            build_payload('book', action='infill', image_path='/does/not/exist')
        with self.assertRaisesRegex(ValueError, '未开放'):
            build_payload('book', parameters={'director_reference_images': []})

    def test_v5_character_limit_and_dimensions(self):
        with self.assertRaises(ValueError):
            build_payload('book', characters=[{'prompt': 'person'}] * 23)
        with self.assertRaises(ValueError):
            build_payload('book', width=833)

    def test_stream_error_does_not_return_an_intermediate_preview(self):
        with self.assertRaises(RuntimeError):
            sse_images(b'event: intermediate\ndata: {"image":"iVBORw0KGgo="}\n\nevent: error\ndata: {"message":"failed"}\n')
        self.assertEqual(sse_images(b'event: intermediate\ndata: {"image":"iVBORw0KGgo="}\n'), [])


if __name__ == '__main__':
    unittest.main()
