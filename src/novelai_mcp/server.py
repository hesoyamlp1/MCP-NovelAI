#!/usr/bin/env python3
"""
NovelAI MCP Server
使用 novelai-python SDK 封装 NovelAI 图像生成 API 为 MCP 工具集。
支持 Vibe Transfer + Precise Reference（Character / Style / Character&Style）
"""


import base64
import io
import json
import os
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import TextContent, ImageContent
from PIL import Image as PILImage
from pydantic import SecretStr
from .v5 import register as register_v5, GUIDANCE as V5_GUIDANCE, reference_fields

from novelai_python import GenerateImageInfer, ApiCredential, ImageGenerateResp
from novelai_python.sdk.ai.generate_image import Character, Model, Sampler, UCPreset
from novelai_python.sdk.ai.generate_image.schema import CharacterPosition


# ---------------------------------------------------------------------------
# 全局变量（在 main() 中初始化）
# ---------------------------------------------------------------------------

credential: ApiCredential | None = None
API_KEY: str | None = None
DEFAULT_SAVE_DIR: str = str(Path.home() / "Pictures" / "NovelAI")
SAVE_DIR: str = DEFAULT_SAVE_DIR

# ---------------------------------------------------------------------------
# FastMCP 实例
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "NovelAI",
    instructions="",  # 在 main() 中动态设置
)

register_v5(mcp, lambda: API_KEY, lambda: SAVE_DIR)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "v4.5-full": Model.NAI_DIFFUSION_4_5_FULL,
    "v4.5-curated": Model.NAI_DIFFUSION_4_5_CURATED,
    "v4-full": Model.NAI_DIFFUSION_4_FULL,
    "v4-curated": Model.NAI_DIFFUSION_4_CURATED_PREVIEW,
    "v3": Model.NAI_DIFFUSION_3,
}

AVAILABLE_MODELS = ", ".join(MODEL_MAP.keys())

DANBOORU_CATEGORY_MAP = {
    0: "general",     # 通用标签（姿势、服装、场景等）
    1: "artist",      # 画师
    3: "copyright",   # 版权/作品名
    4: "character",   # 角色名
    5: "meta",        # 元数据
}


def resolve_model(model_name: str) -> Model:
    """将用户友好的模型名称解析为 SDK 模型枚举"""
    key = model_name.lower().strip()
    if key in MODEL_MAP:
        return MODEL_MAP[key]
    try:
        return Model(key)
    except ValueError:
        raise ValueError(
            f"未知模型: '{model_name}'。可用模型: {AVAILABLE_MODELS}"
        )


def parse_characters(characters: list[dict] | None) -> list[Character] | None:
    """将用户传入的角色列表转换为 SDK 的 Character 对象"""
    if not characters:
        return None
    result = []
    for char in characters:
        center = CharacterPosition(
            x=char.get("center_x", 0.5),
            y=char.get("center_y", 0.5),
        )
        result.append(
            Character(
                prompt=char["prompt"],
                uc=char.get("negative_prompt", ""),
                center=center,
            )
        )
    return result


def save_image(image_data: bytes, prefix: str = "nai") -> str:
    """保存图片到 SAVE_DIR 并返回路径"""
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = save_dir / filename
    filepath.write_bytes(image_data)
    return str(filepath)


def load_reference_images(paths: list[str] | None) -> list[bytes] | None:
    """从文件路径列表读取参考图片，返回 bytes 列表"""
    if not paths:
        return None
    result = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"参考图片文件不存在: {p}")
        data = path.read_bytes()
        print(f"📂 已读取参考图: {p} ({len(data)} bytes)", file=sys.stderr)
        result.append(data)
    return result


# ---------------------------------------------------------------------------
# Precise Reference 辅助函数
# ---------------------------------------------------------------------------

ACCEPTED_CR_SIZES = [(1024, 1536), (1536, 1024), (1472, 1472)]

REFERENCE_MODES = ["character", "style", "character&style"]


def letterbox_for_reference(image_path: str) -> str:
    """
    读取图片 → 选择最接近宽高比的画布 → letterbox 居中 → 返回 base64 编码。
    Precise Reference 要求参考图为 1024x1536 / 1536x1024 / 1472x1472 之一。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Precise Reference 图片不存在: {image_path}")

    pil_img = PILImage.open(path).convert("RGB")
    w, h = pil_img.size

    # 选择宽高比最接近的画布
    aspect = w / h
    canvas_w, canvas_h = min(ACCEPTED_CR_SIZES, key=lambda s: abs(s[0] / s[1] - aspect))

    # 等比缩放 + 居中放到黑色画布上
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)

    canvas = PILImage.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)
    canvas.paste(resized, offset)

    # 编码为 base64 PNG
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print(f"📐 Precise Reference: {w}x{h} → letterbox {canvas_w}x{canvas_h}", file=sys.stderr)
    return b64


def build_precise_reference_params(
    image_b64: str,
    mode: str = "character&style",
    fidelity: float = 1.0,
    strength: float = 1.0,
) -> dict:
    """
    构建 director_reference_* 参数字典。

    mode: "character" / "style" / "character&style"
    fidelity: 0.0~1.0，映射为 secondary_strength = 1.0 - fidelity
    """
    return reference_fields(image_b64, mode, fidelity, strength)


def build_v4_prompt(prompt: str, negative_prompt: str, characters: list[dict] | None = None) -> tuple[dict, dict]:
    """构建 v4_prompt 和 v4_negative_prompt 结构"""
    char_captions = []
    use_coords = False
    if characters:
        use_coords = True
        for char in characters:
            char_captions.append({
                "char_caption": char["prompt"],
                "centers": [{
                    "x": char.get("center_x", 0.5),
                    "y": char.get("center_y", 0.5),
                }],
            })

    v4_prompt = {
        "use_coords": use_coords,
        "use_order": False,
        "caption": {
            "base_caption": prompt,
            "char_captions": char_captions,
        },
    }
    v4_negative_prompt = {
        "use_coords": use_coords,
        "use_order": False,
        "caption": {
            "base_caption": negative_prompt or "lowres",
            "char_captions": [],
        },
    }
    return v4_prompt, v4_negative_prompt


def build_vibe_transfer_params(ref_images: list[bytes] | None, strength: float, info_extracted: float) -> dict:
    """构建 Vibe Transfer 参数（reference_*_multiple 系列），图片 resize 到 448x448 + base64"""
    if not ref_images:
        return {
            "reference_image_multiple": [],
            "reference_strength_multiple": [],
            "reference_information_extracted_multiple": [],
        }
    b64_list = []
    for img_bytes in ref_images:
        pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")
        pil = pil.resize((448, 448), PILImage.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return {
        "reference_image_multiple": b64_list,
        "reference_strength_multiple": [strength] * len(b64_list),
        "reference_information_extracted_multiple": [info_extracted] * len(b64_list),
    }


async def generate_via_http(
    payload: dict,
) -> bytes:
    """
    绕过 SDK 直接调用 NovelAI HTTP API。
    返回生成图片的 bytes。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            "https://image.novelai.net/ai/generate-image",
            headers=headers,
            content=json.dumps(payload).encode("utf-8"),
        )
        if resp.status_code >= 400:
            error_text = resp.text[:500]
            raise RuntimeError(
                f"NovelAI API 错误 ({resp.status_code}): {error_text}"
            )

        # 响应是 zip 格式，解压提取图片
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            image_bytes = zf.read(zf.infolist()[0])
        return image_bytes


# ---------------------------------------------------------------------------
# MCP 工具
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate_image(
    prompt: Annotated[str, "主提示词。描述你想要生成的图片内容，支持自然语言和 NovelAI 标签。例如: 'masterpiece, best quality, 1girl, blue hair, school uniform'"],
    negative_prompt: Annotated[str, "负面提示词。描述你不想看到的内容。留空使用默认值。"] = "",
    model: Annotated[str, f"模型选择。可用: {AVAILABLE_MODELS}"] = "v4.5-full",
    width: Annotated[int, "图片宽度（像素），必须是 64 的倍数。常用: 832（竖图）、1216（横图）、1024（方图）"] = 832,
    height: Annotated[int, "图片高度（像素），必须是 64 的倍数。常用: 1216（竖图）、832（横图）、1024（方图）"] = 1216,
    characters: Annotated[
        list[dict] | None,
        "角色数组。每个角色是一个 dict，包含: prompt（角色描述）、negative_prompt（可选）、center_x（水平位置 0-0.9）、center_y（垂直位置 0-0.9）。单人场景传 1 个，多人传多个。"
    ] = None,
    seed: Annotated[int | None, "随机种子。不填则随机生成。"] = None,
    quality_toggle: Annotated[bool, "是否自动添加质量标签（推荐开启）"] = True,
    variety_boost: Annotated[bool, "多样性增强（Variety Boost），改善样本多样性"] = True,
    reference_image_paths: Annotated[
        list[str] | None,
        "【Vibe Transfer】参考图片的文件绝对路径列表（最多 16 张）。用于风格与视觉特征引导，费用以官方返回为准。不能与 Precise Reference 同时使用。"
    ] = None,
    reference_strength: Annotated[
        float,
        "【Vibe Transfer】参考图强度 0.0-1.0。推荐 0.4-0.6。"
    ] = 0.6,
    reference_information_extracted: Annotated[
        float,
        "【Vibe Transfer】信息提取量 0.0-1.0。推荐 0.8-1.0。"
    ] = 1.0,
    precise_reference_image_path: Annotated[
        str | None,
        "【Precise Reference】参考图片的文件绝对路径（单张）。消耗 Anlas。用于保持角色身份/画风；结果仍需目视确认，适合角色连续性。图片会自动 letterbox 到合适分辨率。"
    ] = None,
    precise_reference_mode: Annotated[
        str,
        "【Precise Reference】参考模式。'character'=仅复制角色身份（换装/换背景时用）；'style'=仅复制画风；'character&style'=同时复制角色和画风（推荐）。"
    ] = "character&style",
    precise_reference_fidelity: Annotated[
        float,
        "【Precise Reference】保真度 0.0-1.0。1.0=最严格匹配参考图，0.0=最自由。推荐 0.8-1.0。"
    ] = 1.0,
) -> Image:
    """
    使用 NovelAI 生成图片（文生图）。

    调用时机：
    - 用户要求「画」「生成」「创建」图片/图像时
    - 用户描述想要看到的视觉内容时
    - 示例：「画一只猫」「生成一个蓝发动漫女孩」「帮我做张图」

    支持单人和多人场景，通过 characters 参数定义角色位置。

    两种参考方式（不可同时使用，Precise Reference 仅 V4.5）：
    1. Vibe Transfer（reference_image_paths）：风格与特征引导，费用依官方接口
    2. Precise Reference（precise_reference_image_path）：消耗 Anlas，精确复制角色身份/画风
       - mode="character": 仅复制角色（适合换装、换背景）
       - mode="style": 仅复制画风（适合风格迁移）
       - mode="character&style": 同时复制角色和画风（推荐）
    """
    resolved_model = resolve_model(model)
    if precise_reference_image_path and reference_image_paths:
        raise ValueError("Precise Reference 与 Vibe Transfer 不兼容，请选择一种")
    if precise_reference_image_path and not resolved_model.value.startswith('nai-diffusion-4-5-'):
        raise ValueError("Precise Reference 仅支持 V4.5")
    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    # ===== 判断是否需要走 HTTP 直调路径 =====
    if precise_reference_image_path:
        # === Precise Reference 路径：绕过 SDK 直接发 HTTP ===
        print(f"🎯 Precise Reference 模式: {precise_reference_mode}, 保真度={precise_reference_fidelity}", file=sys.stderr)

        # 1. 准备参考图
        ref_b64 = letterbox_for_reference(precise_reference_image_path)
        pr_params = build_precise_reference_params(ref_b64, precise_reference_mode, precise_reference_fidelity)

        # 2. 准备 Vibe Transfer（可混合使用）
        ref_images = load_reference_images(reference_image_paths)
        vt_params = build_vibe_transfer_params(ref_images, reference_strength, reference_information_extracted)
        if ref_images:
            print(f"🔗 + Vibe Transfer: {len(ref_images)} 张参考图", file=sys.stderr)

        # 3. 构建 v4_prompt
        v4_prompt, v4_neg = build_v4_prompt(prompt, negative_prompt, characters)

        # 4. 构建完整 payload
        params = {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": 5.0,
            "sampler": "k_euler",
            "steps": 28,
            "seed": actual_seed,
            "n_samples": 1,
            "ucPreset": 3,
            "qualityToggle": quality_toggle,
            "sm": False,
            "sm_dyn": False,
            "dynamic_thresholding": False,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": 0.0,
            "noise_schedule": "native",
            "legacy_v3_extend": False,
            "uncond_scale": 1.0,
            "negative_prompt": negative_prompt or "lowres",
            "prompt": prompt,
            "extra_noise_seed": actual_seed,
            "v4_prompt": v4_prompt,
            "v4_negative_prompt": v4_neg,
            **vt_params,
            **pr_params,
        }

        if variety_boost:
            # skip_cfg_above_sigma 计算 (来自 ComfyUI 逆向)
            params["skip_cfg_above_sigma"] = round((width * height / 1011712) ** 0.5 * 19)

        payload = {
            "input": prompt,
            "model": resolved_model.value,
            "action": "generate",
            "parameters": params,
        }

        print(f"🎨 正在生成图片... (模型: {resolved_model.value}, Precise Reference)", file=sys.stderr)
        image_data = await generate_via_http(payload)
    else:
        # === 原有 SDK 路径 ===
        character_list = parse_characters(characters)

        ref_images = load_reference_images(reference_image_paths)
        ref_kwargs = {}
        if ref_images:
            ref_kwargs["reference_image_multiple"] = ref_images
            ref_kwargs["reference_strength_multiple"] = [reference_strength] * len(ref_images)
            ref_kwargs["reference_information_extracted_multiple"] = [reference_information_extracted] * len(ref_images)
            print(f"🔗 Vibe Transfer: {len(ref_images)} 张参考图, 强度={reference_strength}", file=sys.stderr)

        gen = GenerateImageInfer.build_generate(
            prompt=prompt,
            model=resolved_model,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            character_prompts=character_list,
            seed=actual_seed,
            qualityToggle=quality_toggle,
            variety_boost=variety_boost,
            ucPreset=UCPreset.TYPE0,
            **ref_kwargs,
        )

        try:
            cost = gen.calculate_cost(is_opus=True)
            if cost > 0:
                print(f"💰 预计消耗 {cost} Anlas", file=sys.stderr)
        except Exception:
            pass

        print(f"🎨 正在生成图片... (模型: {resolved_model.value})", file=sys.stderr)
        resp: ImageGenerateResp = await gen.request(session=credential)
        _, image_data = resp.files[0]

    saved_path = save_image(image_data, prefix="gen")
    print(f"💾 图片已保存: {saved_path}", file=sys.stderr)

    return [
        TextContent(type="text", text=f"图片已保存到: {saved_path}"),
        ImageContent(type="image", data=base64.b64encode(image_data).decode("utf-8"), mimeType="image/png"),
    ]


@mcp.tool()
async def img2img(
    prompt: Annotated[str, "提示词。描述你想要在图片基础上生成的内容。"],
    image_path: Annotated[str, "输入图片的文件绝对路径。例如: '/Users/xxx/Pictures/NovelAI/gen_20260217_100218.png'"],
    strength: Annotated[float, "变化强度，0.01-0.99。越大变化越多。"] = 0.7,
    noise: Annotated[float, "添加噪声量，0-0.99。"] = 0.0,
    negative_prompt: Annotated[str, "负面提示词。"] = "",
    model: Annotated[str, f"模型选择。可用: {AVAILABLE_MODELS}"] = "v4.5-full",
    width: Annotated[int, "输出宽度，必须是 64 的倍数。"] = 832,
    height: Annotated[int, "输出高度，必须是 64 的倍数。"] = 1216,
    seed: Annotated[int | None, "随机种子。"] = None,
    reference_image_paths: Annotated[
        list[str] | None,
        "【Vibe Transfer】参考图片的文件绝对路径列表（最多 16 张）。风格与特征引导，费用依官方接口。"
    ] = None,
    reference_strength: Annotated[
        float,
        "【Vibe Transfer】参考图强度 0.0-1.0。推荐 0.4-0.6。"
    ] = 0.6,
    reference_information_extracted: Annotated[
        float,
        "【Vibe Transfer】信息提取量 0.0-1.0。推荐 0.8-1.0。"
    ] = 1.0,
    precise_reference_image_path: Annotated[
        str | None,
        "【Precise Reference】参考图片的文件绝对路径（单张）。消耗 Anlas。精确复制角色身份/画风。"
    ] = None,
    precise_reference_mode: Annotated[
        str,
        "【Precise Reference】参考模式。'character' / 'style' / 'character&style'（推荐）。"
    ] = "character&style",
    precise_reference_fidelity: Annotated[
        float,
        "【Precise Reference】保真度 0.0-1.0。1.0=最严格匹配。推荐 0.8-1.0。"
    ] = 1.0,
) -> Image:
    """
    基于已有图片生成新图（图生图 / img2img）。

    调用时机：
    - 用户想修改或变换现有图片时
    - 用户提供了一张参考图并要求在此基础上修改

    支持 Vibe Transfer 或 Precise Reference，两者不可同时使用；Precise Reference 仅 V4.5。
    """
    resolved_model = resolve_model(model)
    if precise_reference_image_path and reference_image_paths:
        raise ValueError("Precise Reference 与 Vibe Transfer 不兼容，请选择一种")
    if precise_reference_image_path and not resolved_model.value.startswith('nai-diffusion-4-5-'):
        raise ValueError("Precise Reference 仅支持 V4.5")
    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    image_bytes = path.read_bytes()
    print(f"📂 已读取图片: {image_path} ({len(image_bytes)} bytes)", file=sys.stderr)

    if precise_reference_image_path:
        # === Precise Reference 路径：绕过 SDK ===
        print(f"🎯 Precise Reference 模式: {precise_reference_mode}, 保真度={precise_reference_fidelity}", file=sys.stderr)

        ref_b64 = letterbox_for_reference(precise_reference_image_path)
        pr_params = build_precise_reference_params(ref_b64, precise_reference_mode, precise_reference_fidelity)

        ref_images = load_reference_images(reference_image_paths)
        vt_params = build_vibe_transfer_params(ref_images, reference_strength, reference_information_extracted)
        if ref_images:
            print(f"🔗 + Vibe Transfer: {len(ref_images)} 张参考图", file=sys.stderr)

        # 将源图片 resize + base64
        src_pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        src_pil = src_pil.resize((width, height), PILImage.LANCZOS)
        src_buf = io.BytesIO()
        src_pil.save(src_buf, format="PNG")
        src_b64 = base64.b64encode(src_buf.getvalue()).decode("utf-8")

        v4_prompt, v4_neg = build_v4_prompt(prompt, negative_prompt)

        params = {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": 5.0,
            "sampler": "k_euler",
            "steps": 28,
            "seed": actual_seed,
            "n_samples": 1,
            "ucPreset": 3,
            "qualityToggle": True,
            "sm": False,
            "sm_dyn": False,
            "dynamic_thresholding": False,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": 0.0,
            "noise_schedule": "native",
            "legacy_v3_extend": False,
            "uncond_scale": 1.0,
            "negative_prompt": negative_prompt or "lowres",
            "prompt": prompt,
            "extra_noise_seed": actual_seed,
            "image": src_b64,
            "strength": strength,
            "noise": noise,
            "v4_prompt": v4_prompt,
            "v4_negative_prompt": v4_neg,
            **vt_params,
            **pr_params,
        }

        payload = {
            "input": prompt,
            "model": resolved_model.value,
            "action": "img2img",
            "parameters": params,
        }

        print(f"🖌️ 正在进行图生图... (强度: {strength}, Precise Reference)", file=sys.stderr)
        image_data = await generate_via_http(payload)
    else:
        # === 原有 SDK 路径 ===
        ref_images = load_reference_images(reference_image_paths)
        ref_kwargs = {}
        if ref_images:
            ref_kwargs["reference_image_multiple"] = ref_images
            ref_kwargs["reference_strength_multiple"] = [reference_strength] * len(ref_images)
            ref_kwargs["reference_information_extracted_multiple"] = [reference_information_extracted] * len(ref_images)
            print(f"🔗 Vibe Transfer: {len(ref_images)} 张参考图, 强度={reference_strength}", file=sys.stderr)

        gen = GenerateImageInfer.build_img2img(
            prompt=prompt,
            model=resolved_model,
            image=image_bytes,
            strength=strength,
            noise=noise,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            seed=actual_seed,
            ucPreset=UCPreset.TYPE0,
            **ref_kwargs,
        )

        print(f"🖌️ 正在进行图生图... (强度: {strength})", file=sys.stderr)
        resp: ImageGenerateResp = await gen.request(session=credential)
        _, image_data = resp.files[0]

    saved_path = save_image(image_data, prefix="i2i")
    print(f"💾 图片已保存: {saved_path}", file=sys.stderr)

    return [
        TextContent(type="text", text=f"图片已保存到: {saved_path}"),
        ImageContent(type="image", data=base64.b64encode(image_data).decode("utf-8"), mimeType="image/png"),
    ]


@mcp.tool()
async def suggest_tags(
    query: Annotated[str, "搜索关键词。可以是标签片段（如 'blue_h'）或概念关键词（如 'spread'）。支持下划线和空格。"],
    limit: Annotated[int, "返回结果数量上限，默认 20。"] = 20,
) -> list[dict]:
    """
    搜索 Danbooru 标签，用于构建 NovelAI 图片生成的 prompt。

    这是旧版通用 Danbooru 标签查询；V5 建议优先使用 suggest_tags_v5，并结合自然语言描述。
    用于查找标签名称，不应把计数当作模型训练频次或生成质量保证。

    使用建议：
    - 将用户描述拆解为多个独立概念，逐个查询
    - 例如「蓝发女孩M字开腿」→ 分别查 'blue_hair'、'girl'、'm_leg'
    - post_count 是 Danbooru 使用次数，不表示 NovelAI 的训练频次或生成效果
    - 按语义和实际生成效果选择标签

    返回字段：
    - tag: 标签名（直接用于 NovelAI prompt）
    - label: 人类可读名称
    - category: 标签分类（general=通用, character=角色, copyright=作品, artist=画师, meta=元数据）
    - post_count: Danbooru 使用次数
    """
    import httpx

    print(f"🔍 正在搜索 Danbooru 标签: '{query}'", file=sys.stderr)

    async with httpx.AsyncClient(headers={"User-Agent": "NovelAI-MCP/1.0"}) as client:
        resp = await client.get(
            "https://danbooru.donmai.us/autocomplete.json",
            params={
                "search[query]": query,
                "search[type]": "tag_query",
                "limit": min(limit, 20),
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data:
        tag_info = item.get("tag", {})
        category_id = item.get("category", tag_info.get("category", 0))
        results.append({
            "tag": item.get("value", tag_info.get("name", "")),
            "label": item.get("label", ""),
            "category": DANBOORU_CATEGORY_MAP.get(category_id, f"unknown({category_id})"),
            "post_count": item.get("post_count", tag_info.get("post_count", 0)),
        })

    print(f"✅ 找到 {len(results)} 个标签", file=sys.stderr)
    return results


@mcp.tool()
async def check_subscription() -> dict:
    """
    查询 NovelAI 账户的订阅状态。

    返回当前订阅等级、Anlas 余额等信息。
    在生成图片前调用此工具可以确认账户状态。
    """
    import httpx

    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with httpx.AsyncClient() as client:
        sub_resp = await client.get(
            "https://image.novelai.net/user/subscription",
            headers=headers,
        )
        sub_resp.raise_for_status()
        subscription = sub_resp.json()

        try:
            priority_resp = await client.get(
                "https://image.novelai.net/user/priority",
                headers=headers,
            )
            priority_resp.raise_for_status()
            priority = priority_resp.json()
        except Exception:
            priority = None

    tier_names = {0: "Free", 1: "Tablet", 2: "Scroll", 3: "Opus"}
    tier = subscription.get("tier", 0)

    result = {
        "tier": tier_names.get(tier, f"Unknown({tier})"),
        "active": subscription.get("active", False),
        "expiresAt": subscription.get("expiresAt"),
    }

    if priority:
        result["anlas"] = priority.get("maxPriorityActions", 0)

    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="NovelAI MCP Server - AI 图像生成工具集",
        epilog="示例: NOVELAI_API_KEY=pst-xxx python server.py --transport streamable-http",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="传输模式 (默认: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="HTTP 服务端口 (默认: 3000, 仅 HTTP 模式)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=f"图片保存目录 (默认: {DEFAULT_SAVE_DIR})",
    )
    args = parser.parse_args()

    # 加载环境变量并验证 API Key
    load_dotenv()

    global API_KEY, SAVE_DIR, credential
    API_KEY = os.getenv("NOVELAI_API_KEY")
    if not API_KEY:
        print("❌ 错误: 需要设置 NOVELAI_API_KEY 环境变量", file=sys.stderr)
        print("   示例: NOVELAI_API_KEY=pst-xxx python server.py", file=sys.stderr)
        sys.exit(1)

    # 优先级: CLI --save-dir > 环境变量 > 默认值
    if args.save_dir:
        SAVE_DIR = args.save_dir
    elif os.getenv("NOVELAI_SAVE_DIR"):
        SAVE_DIR = os.getenv("NOVELAI_SAVE_DIR")
    # else: 保持默认值 DEFAULT_SAVE_DIR

    credential = ApiCredential(api_token=SecretStr(API_KEY))

    # 动态设置 MCP instructions，包含保存路径信息
    mcp._mcp_server.instructions = V5_GUIDANCE + f"\n素材保存目录：{SAVE_DIR}。旧版工具的模型范围以各自参数为准。"

    print(f"🚀 NovelAI MCP Server 启动中... (传输: {args.transport})", file=sys.stderr)
    print(f"📁 图片保存目录: {SAVE_DIR}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    elif args.transport == "sse":
        mcp.settings.port = args.port
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()
