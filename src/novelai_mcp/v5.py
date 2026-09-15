"""NovelAI V5 tools built against the public image API, independent of legacy SDK enums."""
import base64
import hashlib
import io
import json
from pathlib import Path
import secrets
import uuid
import zipfile

import httpx
from PIL import Image
from mcp.types import ImageContent, TextContent
from mcp.server.fastmcp import Context

MODELS = {'v5-full': 'nai-diffusion-5-full', 'v5-curated': 'nai-diffusion-5-curated'}
REFERENCE_MODELS = {'v4.5-full': 'nai-diffusion-4-5-full', 'v4.5-curated': 'nai-diffusion-4-5-curated'}
GUIDANCE = '''NovelAI 美术工具。V5 优先使用 generate_v5，旧 generate_image/img2img 保留给旧模型。
先用 v5_capabilities 核对功能；用 suggest_tags_v5 查询模型对应的标签。标签计数不是训练数据或生成质量的保证。V5 同时支持标签和自然语言，场景构图可以用明确的自然语言描述。
人物固定外观放在独立 character prompt，场景和风格放在 base prompt；多角色可提供各自坐标与负面提示词。先保存满意的参考图和生成参数，再选择实际支持的图像输入方式保持连续性。
图生图需要 image_path、strength、noise；低 strength 倾向保留原图，高 strength 改动更大。明确描述要保留与要变化的内容，不能把图生图等同于精确身份复制。
实测 0.35 可保留大部分画面但不一定改掉目标细节，0.7 能产生更明显变化但也会改变姿态和服装细节。稳定外貌与必须保留的物件需要在提示词中明确描述。
官方当前 V5 尚未开放 Vibe Transfer/Precise Reference，不能混用 V4.5 的字段假装支持。V5 Full 局部重绘使用 nai-diffusion-5-full-inpainting；旧测试误用了普通生成模型，不能据此判定不支持。V5 Curated 官方网页的局部重绘实际回退 V4.5 Curated；本工具不静默回退，需要时明确选 V4.5 Curated。身份或风格参考使用 generate_reference 的 precise 模式及 V4.5。独立 upscale 已实测可用。2026-09-15 已通过 V5 Full infill API 与选区变化验证，但矩形选区结果有边缘色块，需要检查并优化遮罩；不要承诺无缝结果。V4.5 Full/Curated 的人物参考均已换场景实测，发型和服装可延续，但身材比例与画风仍可能改变。
prepare_image_v5 可准备画布/尺寸及遮罩。图像处理会保存新文件，原文件保留。API 限流或失败不自动重试。
使用 seed、采样和完整参数记录复现画面；prompt 的作用与 img2img strength 相互影响。PNG 支持透明度，透明背景还需要在正面描述中明确要求。
工具可预览最终 payload；预览不是实际生成。每次成功生成返回文件、尺寸、哈希与请求记录；用户可见结果需查看图片。'''


def resolve_model(name):
    if name in MODELS:
        return MODELS[name]
    if name in MODELS.values():
        return name
    raise ValueError('V5 工具只接受 v5-full / v5-curated')


def read_image(path):
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    return base64.b64encode(data).decode()


def sse_images(content):
    images = {}
    event_name = ''
    for line in content.decode().splitlines():
        if line.startswith('event:'):
            event_name = line[6:].strip()
            continue
        if not line.startswith('data:'):
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        label = str(event.get('type') or event.get('event') or event_name).lower()
        if 'error' in label:
            raise RuntimeError('SSE 生成失败：' + str(event.get('message', event.get('error', '见原始响应'))))
        if 'final' not in label:
            continue
        def visit(value):
            if not isinstance(value, dict):
                return
            for key in ('image', 'data'):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith(('iVBOR', 'UklGR')):
                    images[str(value.get('index', 0))] = base64.b64decode(candidate)
            for nested in value.values():
                if isinstance(nested, dict):
                    visit(nested)
                elif isinstance(nested, list):
                    for item in nested:
                        visit(item)
        visit(event)
    return list(images.items())


def build_payload(prompt, model='v5-full', action='generate', width=832, height=1216,
                  negative_prompt='', characters=None, image_path=None, mask_path=None,
                  strength=None, noise=0., seed=None, parameters=None):
    model_id = REFERENCE_MODELS.get(model) or (model if model in REFERENCE_MODELS.values() else resolve_model(model))
    if action == 'infill':
        if model_id == MODELS['v5-curated']:
            raise ValueError('V5 Curated 尚无已确认的原生局部重绘；请明确选择 v5-full 或 v4.5-curated，不自动换模型')
        model_id += '-inpainting'
    if strength is None:
        strength = 1. if action == 'infill' else .5
    if action not in ('generate', 'img2img', 'infill'):
        raise ValueError('action 必须是 generate / img2img / infill')
    if width < 64 or height < 64 or width % 64 or height % 64:
        raise ValueError('宽高必须为至少 64 的 64 倍数')
    if not 0 <= strength <= 1 or not 0 <= noise <= 1:
        raise ValueError('strength/noise 应在 0–1 之间')
    chars = characters or []
    limit = 32 if model_id.startswith('nai-diffusion-5-') else 6
    if len(chars) > limit:
        raise ValueError(f'当前模型最多 {limit} 个独立角色描述')
    positive, negative = [], []
    for char in chars:
        x, y = char.get('x', .5), char.get('y', .5)
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError('角色坐标应在 0–1 之间')
        centers = [{'x': x, 'y': y}]
        positive.append({'char_caption': char['prompt'], 'centers': centers})
        negative.append({'char_caption': char.get('negative_prompt', ''), 'centers': centers})
    actual_seed = seed if seed is not None else secrets.randbelow(2**32)
    params = {'params_version': 4, 'width': width, 'height': height, 'scale': 5., 'sampler': 'k_euler_ancestral', 'steps': 28, 'n_samples': 1, 'seed': actual_seed,
              'noise_schedule': 'karras', 'qualityToggle': False, 'ucPreset': 3, 'negative_prompt': negative_prompt,
              'v4_prompt': {'caption': {'base_caption': prompt, 'char_captions': positive}, 'use_coords': bool(chars), 'use_order': True},
              'v4_negative_prompt': {'caption': {'base_caption': negative_prompt, 'char_captions': negative}, 'use_coords': bool(chars), 'legacy_uc': False},
              'image_format': 'png', 'deliberate_euler_ancestral_bug': False, 'prefer_brownian': True}
    if action in ('img2img', 'infill'):
        if not image_path:
            raise ValueError('图像操作需要 image_path')
        params.update(image=read_image(image_path), strength=strength, noise=noise, extra_noise_seed=actual_seed)
    if action == 'infill':
        if not mask_path:
            raise ValueError('infill 需要 mask_path')
        params['mask'] = read_image(mask_path)
        params['img2img'] = {'strength': strength, 'noise': noise, 'extra_noise_seed': actual_seed}
    params.update(parameters or {})
    if any(not isinstance(params.get(k), int) or params[k] < 64 or params[k] % 64 for k in ('width', 'height')):
        raise ValueError('最终宽高必须是至少 64 的 64 倍数')
    if model_id.startswith('nai-diffusion-5-') and any(k.startswith('director_reference_') or k.startswith('reference_') for k in params):
        raise ValueError('官方当前 V5 未开放 Vibe Transfer/Precise Reference；请使用实际支持的 img2img 输入')
    if params.get('director_reference_images') and params.get('reference_image_multiple'):
        raise ValueError('Precise Reference 与 Vibe Transfer 不兼容，请选择一种')
    if action in ('img2img', 'infill'):
        expected = (params['width'], params['height'])
        for key in ('image', 'mask') if action == 'infill' else ('image',):
            with Image.open(io.BytesIO(base64.b64decode(params[key]))) as im:
                if im.size != expected:
                    raise ValueError(f'{key} 尺寸须等于生成画布 {expected}，请先 prepare_image_v5')
    if params.get('image_format') not in ('png', 'webp'):
        raise ValueError('image_format 必须是 png 或 webp')
    return {'input': prompt, 'model': model_id, 'action': action, 'parameters': params}


def reference_fields(image_b64, mode='character&style', fidelity=1., strength=1.):
    if mode not in ('character', 'style', 'character&style') or not 0 <= fidelity <= 1 or not 0 <= strength <= 1:
        raise ValueError('参考类型需为 character/style/character&style，strength/fidelity 需为 0–1')
    return {'director_reference_images': [image_b64],
            'director_reference_descriptions': [{'use_coords': False, 'use_order': False, 'legacy_uc': False, 'caption': {'base_caption': mode, 'char_captions': []}}],
            'director_reference_strength_values': [strength],
            'director_reference_secondary_strength_values': [1. - fidelity],
            'director_reference_information_extracted': [1.]}


def precise_image(path):
    from PIL import ImageOps
    with Image.open(path) as source:
        size = min(((1024, 1536), (1472, 1472), (1536, 1024)), key=lambda s: abs(s[0]/s[1]-source.width/source.height))
        fitted = ImageOps.contain(source.convert('RGB'), size)
        canvas = Image.new('RGB', size, 'black')
        canvas.paste(fitted, ((size[0]-fitted.width)//2, (size[1]-fitted.height)//2))
        out = io.BytesIO(); canvas.save(out, 'PNG')
    return base64.b64encode(out.getvalue()).decode()


class Client:
    def __init__(self, key, directory):
        self.key = key
        self.directory = Path(directory)

    async def get(self, path, params=None):
        async with httpx.AsyncClient(timeout=30, headers={'Authorization': 'Bearer ' + self.key, 'User-Agent': 'NovelAI-MCP/0.2.0'}) as client:
            response = await client.get('https://image.novelai.net' + path, params=params)
        response.raise_for_status()
        return response.json()

    async def generate(self, payload, endpoint='/ai/generate-image', progress=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        operation = uuid.uuid4().hex
        record_path = self.directory / (operation + '.request.json')
        record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15), headers={'Authorization': 'Bearer ' + self.key, 'User-Agent': 'NovelAI-MCP/0.2.0', 'Accept': 'text/event-stream' if endpoint.endswith('-stream') else 'application/json'}) as client:
                if endpoint.endswith('-stream'):
                    chunks, buffer = [], b''
                    async with client.stream('POST', 'https://image.novelai.net' + endpoint, json=payload) as stream:
                        async for chunk in stream.aiter_bytes():
                            chunks.append(chunk); buffer += chunk
                            while b'\n' in buffer:
                                line, buffer = buffer.split(b'\n', 1)
                                if progress and line.startswith(b'data:'):
                                    try:
                                        event = json.loads(line[5:])
                                        if isinstance(event, dict) and isinstance(event.get('step'), (int, float)):
                                            await progress(event['step'], payload['parameters'].get('steps'))
                                    except (ValueError, KeyError):
                                        pass
                        response = httpx.Response(stream.status_code, headers=stream.headers, content=b''.join(chunks), request=stream.request)
                else:
                    response = await client.post('https://image.novelai.net' + endpoint, json=payload)
        except httpx.RequestError as error:
            receipt = {'operation_id': operation, 'status': 'transport_error', 'request_path': str(record_path),
                       'model': payload.get('model'), 'action': payload.get('action'), 'files': [],
                       'outcome': 'not_submitted' if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)) else 'unknown',
                       'error': {'type': type(error).__name__, 'message': str(error), 'detail': repr(error)}}
            (self.directory / (operation + '.result.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            raise RuntimeError(json.dumps(receipt, ensure_ascii=False)) from error
        response_path = self.directory / (operation + '.response.bin')
        response_path.write_bytes(response.content)
        receipt = {'operation_id': operation, 'status': response.status_code, 'model': payload.get('model'), 'action': payload.get('action'), 'correlation_id': response.headers.get('x-correlation-id'), 'request_path': str(record_path), 'response_path': str(response_path), 'files': []}
        if not response.is_success:
            try:
                error, _ = json.JSONDecoder().raw_decode(response.text)
                receipt['error'] = error
            except ValueError:
                receipt['error'] = {'message': '非 JSON 错误响应，原文保存在 response_path'}
            (self.directory / (operation + '.result.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            raise RuntimeError(json.dumps(receipt, ensure_ascii=False))
        if 'text/event-stream' in response.headers.get('content-type', ''):
            images = sse_images(response.content)
        elif 'json' in response.headers.get('content-type', ''):
            images = [(str(i.get('index', n)), base64.b64decode(i['image'])) for n, i in enumerate(response.json()['images'])]
        else:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                images = [(str(n), archive.read(name)) for n, name in enumerate(archive.namelist()) if name.lower().endswith(('.png', '.webp'))]
        if not images:
            raise RuntimeError('响应未包含可用图片；原响应保存在 ' + str(response_path))
        for index, data in images:
            with Image.open(io.BytesIO(data)) as im:
                fmt, size, mode = im.format.lower(), im.size, im.mode
            path = self.directory / f'{operation}-{index}.{fmt}'
            path.write_bytes(data)
            receipt['files'].append({'path': str(path), 'width': size[0], 'height': size[1], 'mode': mode, 'sha256': hashlib.sha256(data).hexdigest()})
        (self.directory / (operation + '.result.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt


def register(mcp, key, directory):
    def client():
        return Client(key(), directory())

    @mcp.tool()
    async def import_image_v5(image_base64: str) -> dict:
        """导入调用方提供的参考图片原始字节，验证图像后保存独立文件；返回服务端路径供图生图使用。"""
        if len(image_base64) > 40_000_000:
            raise ValueError('参考图最大 30 MB')
        data = base64.b64decode(image_base64, validate=True)
        with Image.open(io.BytesIO(data)) as im:
            fmt, size = im.format.lower(), im.size
            im.verify()
        if fmt not in ('png', 'jpeg', 'webp'):
            raise ValueError('参考图需为 PNG、JPEG 或 WebP')
        folder = Path(directory()); folder.mkdir(parents=True, exist_ok=True)
        target = folder / (uuid.uuid4().hex + '-reference.' + fmt)
        target.write_bytes(data)
        return {'path': str(target), 'width': size[0], 'height': size[1], 'sha256': hashlib.sha256(data).hexdigest()}

    @mcp.tool()
    async def director_image(image_path: str, operation: str, prompt: str = '', defry: int = 0) -> list:
        """独立 Director Tools，并非 V5 原生模型功能。operation: bg-removal、lineart、sketch、colorize、emotion、declutter、declutter-keep-bubbles。prompt/defry 原样传给官方接口；emotion 的 prompt 使用 mood;;附加描述，例如 happy;;smile（安装 SDK 的格式），正面单人面孔效果更适合。colorize 可用标签描述颜色。保留全部返回图片与原始请求结果，不自动重试。"""
        if operation not in ('bg-removal', 'lineart', 'sketch', 'colorize', 'emotion', 'declutter', 'declutter-keep-bubbles'):
            raise ValueError('未知 Director Tools 操作')
        with Image.open(image_path) as im:
            width, height = im.size
        receipt = await client().generate({'image': read_image(image_path), 'width': width, 'height': height,
                                           'req_type': operation, 'prompt': prompt, 'defry': defry}, '/ai/augment-image')
        return [TextContent(type='text', text=json.dumps(receipt, ensure_ascii=False))] + [ImageContent(type='image', data=read_image(f['path']), mimeType='image/' + Path(f['path']).suffix[1:]) for f in receipt['files']]

    @mcp.tool()
    async def v5_capabilities() -> dict:
        """V5 模型及能力边界，使用前先读取。verified 状态另见本项目验收记录。"""
        return {'models': MODELS, 'independent_director_tools': {'tool': 'director_image', 'verified_on_v5_source_images': ['lineart', 'bg-removal', 'sketch', 'colorize', 'emotion', 'declutter', 'declutter-keep-bubbles'], 'effect_limits': '生成式处理可能同时改变文字、颜色、衣服和背景；必须保留原图并检查结果，非精确局部编辑。'}, 'documented': ['text_to_image', 'image_to_image', 'character_prompts_ui_limit_32', 'free_character_coordinates', 'natural_language_and_tags', 'text_rendering', 'transparent_background'],
                'not_available_per_current_official_docs': ['vibe_transfer', 'precise_reference'], 'verified_supported': ['text_to_image', 'image_to_image', 'upscale', 'sse_streaming', 'multi_character', 'transparent_background', 'png', 'webp', 'text_rendering'], 'verified_unsupported': [], 'inpainting': {'v5-full': {'model': 'nai-diffusion-5-full-inpainting', 'documented': True, 'verification': 'api_and_mask_effect_verified_2026-09-15', 'visual_limit': '矩形选区测试存在边缘色块，非无缝编辑保证'}, 'v5-curated': {'native': False, 'explicit_alternative': 'v4.5-curated'}}, 'reference_models': REFERENCE_MODELS, 'reference_tool': 'generate_reference', 'reference_verification': {'v4.5-full': 'precise_character_verified', 'v4.5-curated': 'precise_character_verified_and_infill_api_verified'}, 'official_client_source': 'https://novelai.net/_next/static/chunks/pages/_app-e14292a0bc1fd2c7.js',
                'prompt_limits_approx_tokens': {'v5-full': {'base': 1471, 'text': 750}, 'v5-curated': {'base': 703, 'text': 374}},
                'references': ['https://novelai.net/v5', 'https://docs.novelai.net/en/image/models/', 'https://image.novelai.net/docs/doc.json'], 'guide': GUIDANCE}

    @mcp.tool()
    async def v5_parameter_schema() -> dict:
        """读取官方图像 API 参数结构；共享字段不代表 V5 全部支持，结合 v5_capabilities 使用。"""
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.get('https://image.novelai.net/docs/doc.json')
            response.raise_for_status()
        definitions = response.json()['definitions']
        return {'source': 'https://image.novelai.net/docs/doc.json', 'schemas': {k: v for k, v in definitions.items() if k.startswith('image.')}, 'note': '这些是图像服务共享 schema；V5 Full 的 infill 需独立 inpainting 模型；V5 不支持 Precise Reference/Vibe Transfer，V4.5 参考用 generate_reference。'}

    @mcp.tool()
    async def suggest_tags_v5(query: str, model: str = 'v5-full', language: str = 'en') -> dict:
        """使用 NovelAI 自身的模型标签建议，保留原始 count/confidence，不把它当成生成质量评分。"""
        return await client().get('/ai/generate-image/suggest-tags', {'model': resolve_model(model), 'prompt': query, 'lang': language})

    @mcp.tool()
    async def account_v5() -> dict:
        """查询图像服务的真实订阅、usage 和 priority 返回，不自行推算未知费用或额度。"""
        c = client()
        return {'subscription': await c.get('/user/subscription'), 'priority': await c.get('/user/priority')}

    @mcp.tool()
    async def generate_reference(prompt: str, image_path: str, model: str = 'v4.5-full', reference_mode: str = 'precise',
                                 width: int = 832, height: int = 1216, reference_type: str = 'character',
                                 strength: float | None = None, fidelity: float = .8, noise: float = 0., reference_strength: float = 1.,
                                 mask_path: str | None = None, identity_reference_path: str | None = None,
                                 negative_prompt: str = '', characters: list[dict] | None = None,
                                 seed: int | None = None, parameters: dict | None = None, preview_only: bool = False) -> list:
        """带图生成，返回实际模型与原始请求/图片。precise 保持身份/风格并重新构图（仅 V4.5）；img2img 基于底图重绘（V5/V4.5）；infill 遮罩局部重绘（V5 Full/V4.5）。infill 可另给 identity_reference_path（仅 V4.5）。不静默换模型。"""
        if reference_mode not in ('precise', 'img2img', 'infill'):
            raise ValueError('reference_mode 需为 precise / img2img / infill')
        if strength is None:
            strength = 1. if reference_mode == 'infill' else (.7 if reference_mode == 'precise' else .5)
        is_precise = reference_mode == 'precise' or bool(identity_reference_path)
        if is_precise and model not in (*REFERENCE_MODELS, *REFERENCE_MODELS.values()):
            raise ValueError('Precise Reference 只支持 V4.5，请明确选择 v4.5-full 或 v4.5-curated')
        if identity_reference_path and reference_mode != 'infill':
            raise ValueError('额外身份参考用于 infill；precise 直接使用 image_path')
        params = dict(parameters or {})
        if is_precise:
            if any(k.startswith(('reference_', 'director_reference_')) for k in params):
                raise ValueError('请使用专用参考参数，避免混合或覆盖参考方式')
            params.update(reference_fields(precise_image(identity_reference_path or image_path), reference_type, fidelity, strength if reference_mode == 'precise' else reference_strength))
        payload = build_payload(prompt, model, 'generate' if reference_mode == 'precise' else reference_mode,
                                width, height, negative_prompt, characters, None if reference_mode == 'precise' else image_path,
                                mask_path, strength, noise, seed, params)
        if preview_only:
            return [TextContent(type='text', text=json.dumps({'preview_only': True, 'payload': payload}, ensure_ascii=False))]
        receipt = await client().generate(payload)
        receipt.update(model=payload['model'], reference_mode=reference_mode, reference_path=image_path)
        return [TextContent(type='text', text=json.dumps(receipt, ensure_ascii=False))] + [ImageContent(type='image', data=read_image(f['path']), mimeType='image/' + Path(f['path']).suffix[1:]) for f in receipt['files']]

    @mcp.tool()
    async def generate_v5(prompt: str, model: str = 'v5-full', action: str = 'generate', width: int = 832, height: int = 1216,
                          negative_prompt: str = '', characters: list[dict] | None = None, image_path: str | None = None, mask_path: str | None = None,
                          strength: float | None = None, noise: float = 0., seed: int | None = None, parameters: dict | None = None, preview_only: bool = False, stream: bool = False, ctx: Context = None) -> list:
        """V5 文生图/图生图，Full 支持 infill 并使用独立 inpainting 模型；Curated 不自动回退。characters 每项含 prompt、negative_prompt、x、y；parameters 为官方采样/步数/引导等设置。不会自动重试失败请求。"""
        resolve_model(model)
        payload = build_payload(prompt, model, action, width, height, negative_prompt, characters, image_path, mask_path, strength, noise, seed, parameters)
        if stream:
            payload['parameters']['stream'] = 'sse'
        if preview_only:
            return [TextContent(type='text', text=json.dumps({'preview_only': True, 'payload': payload}, ensure_ascii=False))]
        receipt = await client().generate(payload, '/ai/generate-image-stream' if stream else '/ai/generate-image', ctx.report_progress if ctx else None)
        return [TextContent(type='text', text=json.dumps(receipt, ensure_ascii=False))] + [ImageContent(type='image', data=base64.b64encode(Path(f['path']).read_bytes()).decode(), mimeType='image/' + Path(f['path']).suffix[1:]) for f in receipt['files']]

    @mcp.tool()
    async def inspect_image_v5(image_path: str, include_image: bool = True) -> list:
        """查看参考图尺寸、透明度及生成元数据；可同时返回原图给具备视觉能力的美术 Agent 检查。"""
        with Image.open(image_path) as im:
            metadata = {'path': image_path, 'width': im.width, 'height': im.height, 'mode': im.mode, 'metadata': im.info}
            if im.mode == 'RGBA':
                metadata['alpha_range'] = im.getchannel('A').getextrema()
            fmt = im.format.lower()
        result = [TextContent(type='text', text=json.dumps(metadata, ensure_ascii=False, default=str))]
        if include_image:
            result.append(ImageContent(type='image', data=read_image(image_path), mimeType='image/' + ('jpeg' if fmt == 'jpg' else fmt)))
        return result

    @mcp.tool()
    async def upscale_v5(image_path: str, model: str = 'v5-full', declared_blur_sigma: float = 0.) -> list:
        """官方独立放大接口；支持情况需按 V5 各模型真实返回确认，不假称是生成模型的能力。"""
        receipt = await client().generate({'image': read_image(image_path), 'model': resolve_model(model), 'declared_blur_sigma': declared_blur_sigma}, '/ai/upscale')
        return [TextContent(type='text', text=json.dumps(receipt, ensure_ascii=False))]

    @mcp.tool()
    async def prepare_image_v5(image_path: str, width: int, height: int, mode: str = 'contain', mask_box: list[int] | None = None, crop_box: list[int] | None = None, mask_polygons: list[list[list[int]]] | None = None) -> dict:
        """准备图像输入，保存新 PNG，保留原文件。crop_box=[左,上,右,下] 按原图坐标先裁切；contain 等比留边，cover 等比裁切，stretch 拉伸。mask_box 或 mask_polygons（多个 [[x,y],...] 轮廓，按输出画布坐标）生成白色选区/黑色背景遮罩，可合并多个选区。用于局部重绘，不必把周围背景一起框进去。"""
        from PIL import ImageOps, ImageDraw
        if width < 64 or height < 64 or width % 64 or height % 64 or width * height > 16_777_216:
            raise ValueError('画布宽高需为 64 倍数，最大 16MP')
        if mask_box is not None and (len(mask_box) != 4 or not 0 <= mask_box[0] < mask_box[2] <= width or not 0 <= mask_box[1] < mask_box[3] <= height):
            raise ValueError('mask_box 超出画布')
        if mask_polygons is not None:
            if not 1 <= len(mask_polygons) <= 32 or any(not 3 <= len(poly) <= 128 or any(len(point) != 2 or not all(type(n) is int for n in point) or not (0 <= point[0] < width and 0 <= point[1] < height) for point in poly) for poly in mask_polygons):
                raise ValueError('mask_polygons 需为 1–32 个多边形，每个 3–128 个画布内整像素坐标')
        with Image.open(image_path) as im:
            im = im.convert('RGBA')
            if crop_box is not None:
                if len(crop_box) != 4 or not 0 <= crop_box[0] < crop_box[2] <= im.width or not 0 <= crop_box[1] < crop_box[3] <= im.height:
                    raise ValueError('crop_box 超出原图')
                im = im.crop(crop_box)
            if mode == 'contain':
                out = Image.new('RGBA', (width, height), (255, 255, 255, 255)); fitted = ImageOps.contain(im, (width, height)); out.paste(fitted, ((width-fitted.width)//2, (height-fitted.height)//2))
            elif mode == 'cover':
                out = ImageOps.fit(im, (width, height))
            elif mode == 'stretch':
                out = im.resize((width, height), Image.Resampling.LANCZOS)
            else:
                raise ValueError('mode 必须是 contain / cover / stretch')
        folder = Path(directory()); folder.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        target = folder / (stem + '-prepared.png'); out.save(target)
        result = {'path': str(target), 'width': width, 'height': height, 'source': image_path, 'mode': mode, 'crop_box': crop_box}
        if mask_box is not None or mask_polygons:
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            if mask_box is not None: draw.rectangle(mask_box, fill=255)
            for poly in mask_polygons or []: draw.polygon([tuple(point) for point in poly], fill=255)
            mask_path = folder / (stem + '-mask.png'); mask.save(mask_path); result['mask_path'] = str(mask_path)
        return result
