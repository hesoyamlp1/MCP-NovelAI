"""Bounded real MCP acceptance. Run only when explicitly requested by the developer/user."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['basic', 'img2img', 'img2img_stronger', 'infill', 'upscale', 'characters_alpha', 'text', 'stream', 'director_lineart', 'director_bg-removal', 'director_sketch', 'director_colorize', 'director_emotion', 'director_declutter', 'director_declutter-keep-bubbles', 'director_fixture', 'director_emotion_closeup', 'director_declutter_text', 'director_declutter-keep-bubbles_text'], default='basic')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / '.runtime/v5-e2e'
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for line in (root / '.runtime/novelai.env').read_text().splitlines():
        if line.startswith('NOVELAI_API_KEY='):
            env['NOVELAI_API_KEY'] = line.split('=', 1)[1]
    env.update(PYTHONPATH=str(root / 'src'), NOVELAI_SAVE_DIR=str(out))
    params = StdioServerParameters(command=sys.executable, args=['-m', 'novelai_mcp.server'], env=env, cwd=str(root))
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            catalog = await session.list_tools()
            (out / 'tools.json').write_text(catalog.model_dump_json(indent=2))
            for model in ['v5-full', 'v5-curated']:
                started = time.time()
                call = {'model': model, 'prompt': 'An adult woman, age 32, black hair tied in a low ponytail, beige shirt and dark apron, standing in a quiet seaside bookstore, holding a blue book. Detailed anime illustration, warm afternoon light, full body, natural proportions, no text.', 'negative_prompt': 'lowres, blurry, text, watermark', 'seed': 314159, 'width': 832, 'height': 1216}
                tool = 'generate_v5'
                if args.case == 'director_fixture':
                    call.update(width=1024, height=1024, prompt='A single clean manga panel, close-up front portrait of one adult man age 35, mature facial proportions, short black hair, neutral expression, closed mouth, plain collared shirt. A large white oval speech bubble with a clear black outline and tail next to his head says HELLO. Outside the bubble, big floating red letters in the upper left say DRAFT. Simple white background, crisp detailed anime linework, subtle flat colors. Text: HELLO. Text: DRAFT.')
                elif args.case == 'characters_alpha':
                    call.update(width=1216,height=832,prompt='2people, two adult friends standing side by side, full body, isolated on transparent background, no scenery, clean detailed anime illustration, no text.', characters=[{'prompt':'woman, age 32, long black hair, white jacket, black trousers', 'x':.25,'y':.5},{'prompt':'man, age 35, short brown hair, navy blue shirt, grey trousers','x':.75,'y':.5}], parameters={'straight_alpha':True,'tag_hint_transparent_background':True})
                elif args.case == 'text':
                    call.update(width=1024,height=1024,prompt='A clean hand-painted sign for a small seaside bookstore. Large clear English lettering in the center. Warm cream background with a tiny blue sailboat. Text: BOOKS', parameters={'image_format':'webp'})
                elif args.case == 'stream':
                    call.update(stream=True, width=1024, height=1024, prompt='A peaceful seaside bookstore at blue hour, warm lights in the windows, no people, detailed anime background illustration, no text.')
                elif args.case != 'basic':
                    prior = json.loads((out / (model + '-basic.receipt.json')).read_text())
                    image = prior['files'][0]['path']
                    if args.case in ('img2img', 'img2img_stronger'):
                        call.update(action='img2img', image_path=image, strength=.35, noise=0., prompt=call['prompt'].replace('blue book', 'red book').replace('afternoon', 'evening'))
                        if args.case == 'img2img_stronger':
                            call.update(strength=.7, prompt=call['prompt'] + ' The book cover is bright red. Cool blue evening light outside the windows.')
                    elif args.case == 'infill':
                        prep = await session.call_tool('prepare_image_v5', {'image_path': image, 'width': 832, 'height': 1216, 'mask_box': [200, 420, 630, 800]})
                        data = json.loads(next(c.text for c in prep.content if c.type == 'text'))
                        call.update(action='infill', image_path=data['path'], mask_path=data['mask_path'])
                    elif args.case.startswith('director_'):
                        operation = args.case[len('director_'):].removesuffix('_text').removesuffix('_closeup')
                        tool, call = 'director_image', {'image_path': image, 'operation': operation}
                        if args.case.endswith(('_text', '_closeup')):
                            fixture = json.loads((out / (model + '-director_fixture.receipt.json')).read_text())
                            call['image_path'] = fixture['files'][0]['path']
                        if args.case == 'director_colorize':
                            lineart = json.loads((out / (model + '-director_lineart.receipt.json')).read_text())
                            call.update(image_path=lineart['files'][0]['path'], prompt='black hair, beige shirt, brown apron, blue book, bookstore', defry=2)
                        elif args.case in ('director_emotion', 'director_emotion_closeup'):
                            call.update(prompt='happy;;open mouth, smiling, laughing' if args.case.endswith('_closeup') else 'happy;;', defry=0 if args.case.endswith('_closeup') else 1)
                    else:
                        tool, call = 'upscale_v5', {'model': model, 'image_path': image}
                result = await session.call_tool(tool, call, read_timeout_seconds=__import__('datetime').timedelta(seconds=240))
                (out / (model + '-' + args.case + '.tool-result.json')).write_text(result.model_dump_json(indent=2))
                if result.isError:
                    print(json.dumps({'model': model, 'case': args.case, 'ok': False, 'error': [c.text for c in result.content if c.type == 'text'], 'seconds': time.time()-started}, ensure_ascii=False), flush=True)
                    if args.case == 'basic':
                        raise RuntimeError('基础生成验收失败')
                    continue
                receipt = json.loads(next(c.text for c in result.content if c.type == 'text'))
                (out / (model + '-' + args.case + '.receipt.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
                print(json.dumps({'model': model, 'case': args.case, 'ok': True, 'files': receipt.get('files'), 'seconds': time.time()-started}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
