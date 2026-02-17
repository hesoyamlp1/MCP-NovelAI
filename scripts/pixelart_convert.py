#!/usr/bin/env python3
"""
将图片通过 PixelLab API 转换为像素画风格。
用法: python pixelart_convert.py <input_image> [--output <output_path>] [--width 64] [--height 64]
"""

import argparse
import base64
import sys
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="将图片转换为像素画")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output", "-o", help="输出路径 (默认: 输入文件名_pixel.png)")
    parser.add_argument("--width", type=int, default=64, help="输出宽度 (默认: 64)")
    parser.add_argument("--height", type=int, default=96, help="输出高度 (默认: 96)")
    parser.add_argument("--token", default="06c61347-2c1c-419d-b002-36f9cfafcdbf", help="PixelLab API token")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_pixel.png"

    # 读取图片并编码为 base64
    print(f"📖 读取图片: {input_path}", file=sys.stderr)
    image_bytes = input_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()

    # 获取输入图片尺寸 (通过 PIL)
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            input_w, input_h = img.size
        print(f"📐 输入尺寸: {input_w}×{input_h}", file=sys.stderr)
    except ImportError:
        print("⚠️  未安装 Pillow，使用默认输入尺寸 832×1216", file=sys.stderr)
        input_w, input_h = 832, 1216

    # 构造请求
    payload = {
        "image": {"type": "base64", "base64": image_b64, "format": "png"},
        "image_size": {"width": input_w, "height": input_h},
        "output_size": {"width": args.width, "height": args.height},
    }

    print(f"🎨 正在转换为像素画... ({input_w}×{input_h} → {args.width}×{args.height})", file=sys.stderr)

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                "https://api.pixellab.ai/v2/image-to-pixelart",
                headers={
                    "Authorization": f"Bearer {args.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            print(f"📡 API 状态码: {resp.status_code}", file=sys.stderr)
            if resp.status_code != 200:
                print(f"❌ API 错误 ({resp.status_code}): {resp.text[:500]}", file=sys.stderr)
                sys.exit(1)
            data = resp.json()
    except Exception as e:
        print(f"❌ 请求失败: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    # 提取并保存结果
    output_b64 = data.get("image", {}).get("base64", "")
    if not output_b64:
        print("❌ 未能从响应中提取图片", file=sys.stderr)
        sys.exit(1)

    output_bytes = base64.b64decode(output_b64)
    output_path.write_bytes(output_bytes)
    print(f"✅ 像素画已保存: {output_path}", file=sys.stderr)
    # 输出路径到 stdout 方便脚本串联
    print(str(output_path))


if __name__ == "__main__":
    main()
