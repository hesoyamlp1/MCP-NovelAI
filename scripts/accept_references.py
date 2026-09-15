"""One explicitly selected real MCP reference test; originals and receipts retained."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True,choices=['v5_full_infill','v45_full_reference','v45_curated_reference','v45_curated_infill','v5_curated_img2img'])
    case=parser.parse_args().case
    root=Path(__file__).resolve().parents[1]
    out=root/'.runtime/reference-repair';out.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'.runtime/v5-e2e/v5-full-basic.receipt.json').read_text())['files'][0]['path']
    env=dict(os.environ)
    for line in (root/'.runtime/novelai.env').read_text().splitlines():
        if line.startswith('NOVELAI_API_KEY='):env['NOVELAI_API_KEY']=line.split('=',1)[1]
    env.update(PYTHONPATH=str(root/'src'),NOVELAI_SAVE_DIR=str(out))
    config=StdioServerParameters(command=sys.executable,args=['-m','novelai_mcp.server'],env=env,cwd=str(root))
    async with stdio_client(config) as (reader,writer):
        async with ClientSession(reader,writer) as session:
            await session.initialize()
            call={'prompt':'An adult woman age 32, black hair in a low ponytail, beige shirt, dark brown apron, dark green trousers, holding a blue book, seated at a wooden table in an outdoor seaside cafe, three-quarter view, detailed anime illustration, no text.', 'image_path':source,'width':832,'height':1216,'seed':314159,'negative_prompt':'text, watermark, lowres'}
            if case.endswith('infill'):
                prepared=await session.call_tool('prepare_image_v5',{'image_path':source,'width':832,'height':1216,'mask_box':[453,465,519,601]})
                if prepared.isError:raise RuntimeError(str(prepared))
                prepared=json.loads(next(c.text for c in prepared.content if c.type=='text'))
                call.update(reference_mode='infill',model='v5-full' if case.startswith('v5_') else 'v4.5-curated',image_path=prepared['path'],mask_path=prepared['mask_path'],strength=.8,
                            prompt='An adult woman age 32, black hair in a low ponytail, beige shirt and dark brown apron, standing in a bookstore holding a bright red hardcover book. The book cover is vivid red. Preserve the hand, pose and surrounding image. Detailed anime illustration, no text.')
            elif case.endswith('img2img'):
                call.update(reference_mode='img2img',model='v5-curated',strength=.5)
            else:
                call.update(reference_mode='precise',model='v4.5-full' if 'full' in case else 'v4.5-curated',reference_type='character',strength=.7,fidelity=.85)
            (out/(case+'.input.json')).write_text(json.dumps(call,ensure_ascii=False,indent=2))
            result=await session.call_tool('generate_reference',call,read_timeout_seconds=timedelta(seconds=240))
            (out/(case+'.tool-result.json')).write_text(result.model_dump_json(indent=2))
            if result.isError:
                print(json.dumps({'case':case,'ok':False,'errors':[c.text for c in result.content if c.type=='text']},ensure_ascii=False),flush=True)
                raise SystemExit(1)
            receipt=json.loads(next(c.text for c in result.content if c.type=='text'))
            (out/(case+'.receipt.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
            print(json.dumps({'case':case,'ok':True,'model':receipt.get('model'),'files':receipt['files']},ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(main())
