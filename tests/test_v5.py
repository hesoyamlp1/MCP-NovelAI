import unittest
from novelai_mcp.v5 import build_payload, sse_images, reference_fields, precise_image
import tempfile
from pathlib import Path
from PIL import Image
import base64
import io


class V5Tests(unittest.TestCase):
    def test_character_positive_and_negative_prompts_stay_separate(self):
        p = build_payload('bookstore', characters=[{'prompt': 'white jacket', 'negative_prompt': 'blue shirt', 'x': .2, 'y': .5}, {'prompt': 'blue shirt', 'x': .8, 'y': .5}], seed=7)
        self.assertEqual(p['parameters']['v4_prompt']['caption']['char_captions'][0]['char_caption'], 'white jacket')
        self.assertEqual(p['parameters']['v4_negative_prompt']['caption']['char_captions'][0]['char_caption'], 'blue shirt')
        self.assertEqual(p['parameters']['seed'], 7)

    def test_unavailable_v5_features_are_refused_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, '不自动换模型'):
            build_payload('book', model='v5-curated', action='infill', image_path='/does/not/exist')
        with self.assertRaisesRegex(ValueError, '未开放'):
            build_payload('book', parameters={'director_reference_images': []})

    def test_inpaint_uses_dedicated_model_and_checks_mask_size(self):
        with tempfile.TemporaryDirectory() as d:
            image, mask = str(Path(d)/'image.png'), str(Path(d)/'mask.png')
            Image.new('RGB', (64,64)).save(image)
            Image.new('L', (64,64), 255).save(mask)
            for model, expected in [('v5-full','nai-diffusion-5-full-inpainting'),('v4.5-full','nai-diffusion-4-5-full-inpainting'),('v4.5-curated','nai-diffusion-4-5-curated-inpainting')]:
                p=build_payload('red book',model=model,action='infill',width=64,height=64,image_path=image,mask_path=mask)
                self.assertEqual(p['model'],expected)
                self.assertEqual(p['parameters']['img2img']['strength'],1.)
            Image.new('L',(128,64)).save(mask)
            with self.assertRaisesRegex(ValueError,'尺寸'):
                build_payload('book',action='infill',width=64,height=64,image_path=image,mask_path=mask)

    def test_precise_reference_uses_actual_pixels_and_modes(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'reference.png');Image.new('RGB',(64,96),'red').save(path)
            data=precise_image(path)
            with Image.open(io.BytesIO(base64.b64decode(data))) as im:self.assertEqual(im.size,(1024,1536))
            fields=reference_fields(data,'character',.8,.7)
            p=build_payload('sitting in a cafe',model='v4.5-full',parameters=fields)
            self.assertEqual(p['parameters']['director_reference_images'],[data])
            self.assertEqual(p['parameters']['director_reference_strength_values'],[.7])
            with self.assertRaisesRegex(ValueError,'不兼容'):
                build_payload('book',model='v4.5-full',parameters={**fields,'reference_image_multiple':['vibe']})

    def test_v5_character_limit_and_dimensions(self):
        with self.assertRaises(ValueError):
            build_payload('book', characters=[{'prompt': 'person'}] * 33)
        with self.assertRaises(ValueError):
            build_payload('book', width=833)

    def test_stream_error_does_not_return_an_intermediate_preview(self):
        with self.assertRaises(RuntimeError):
            sse_images(b'event: intermediate\ndata: {"image":"iVBORw0KGgo="}\n\nevent: error\ndata: {"message":"failed"}\n')
        self.assertEqual(sse_images(b'event: intermediate\ndata: {"image":"iVBORw0KGgo="}\n'), [])


if __name__ == '__main__':
    unittest.main()
