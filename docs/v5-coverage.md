# V5 能力覆盖与真实验收

截至本轮实际接口验证；不是对后续版本永久可用性的承诺。

## 已通过真实 MCP 调用

| 能力 | V5 Full | V5 Curated | 证据 |
|---|---|---|---|
| 文生图 | 通过 | 通过 | basic，832×1216，PNG Source 标识 NovelAI Diffusion V5 |
| 图生图 | 通过 | 通过 | img2img / img2img_stronger，使用对应模型的原图 |
| 独立放大 | 通过 | 通过 | upscale，832×1216 → 1664×2432 |
| 多角色与坐标 | 通过 | 通过 | characters_alpha，左侧白外套女士、右侧蓝上衣男士 |
| 透明背景 | 通过 | 通过 | RGBA，alpha 范围 0–255，画面无背景 |
| 文字与 WebP | 通过 | 通过 | text，BOOKS 字样可辨，1024×1024 WebP |
| SSE | 通过 | 通过 | stream，接收最终图片，场景图可见；中间预览不会被当成最终成功 |
| 模型标签建议 | 通过 | 通过 | 官方 suggest-tags 接口，原样保留 count/confidence |

## 明确不可用的边界

- infill：两个 V5 模型均实际返回 HTTP 400，提示模型不支持 infill。响应后还附带图片内容，但这不证明局部重绘成立。工具已在提交前拒绝此动作；保留最初原始错误响应作为证据。
- Vibe Transfer、Precise Reference：当前官方 V5 页面声明尚未开放，V5 工具拒绝相关字段。不用 V4.5 实现冒充 V5。
- Director Tools 属于独立图像工具，尚未在本轮接入与逐项验证；不会声称它们属于已完成的 V5 原生覆盖。

## 实际画面结论

- 相同输入在两个模型中得到不同的角色构图，均符合基本场景和服装描述。
- strength=0.35 时角色和画面保存得很近，但没有可靠地把蓝书改成红书。
- strength=0.7 时两模型都产生红书，人物基本发色、服装和场景仍相近；Full 的姿态、鞋子等细节也发生变化。应明确写出必须保留的细节，不把 img2img 当成局部精确编辑或身份锁定。
- 已目视检查文字、透明双人图、放大图及 SSE 场景图。

## 使用与证据位置

新版 V5 工具当前需从源码运行，尚未发布新版 PyPI 包。

```sh
PYTHONPATH=src .venv/bin/python -m novelai_mcp.server
```

Mac 测试目录：`/Users/linsuki/passion/MCP-NovelAI/.runtime/v5-e2e/`。
每种 case 分别保存两个模型的 `*.tool-result.json`、`*.receipt.json`、原始请求、响应和图片。测试入口为 `scripts/accept_v5.py`，只按显式选择的 case 执行有限请求，不自动重试。
图集：https://show.toddout.work/a/mu04gdhy3f72

## 官方依据

- https://novelai.net/v5
- https://docs.novelai.net/en/image/models/
- https://docs.novelai.net/en/image/controltools/
- https://docs.novelai.net/en/image/multiplecharacters/
- https://image.novelai.net/docs/doc.json

目标 1 仍需接入游戏后台、补齐必要辅助能力和综合验收；目标 2、Feature 3 尚未完成，不能以本表替代整个 goal 的完成。
