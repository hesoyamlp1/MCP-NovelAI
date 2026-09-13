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
- Director Tools 属于独立图像工具，现已提供 director_image。线稿与去背景已对两个 V5 模型来源图完成真实 MCP 调用；其余操作已完成真实返回测试；效果限制见下，不计入 V5 原生能力。

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

游戏后台已接入 NovelAI、跨主机参考图导入及图生图，并完成实际画面检查；整个 Goal 的其余验收继续进行，本表不替代综合验收。

## 独立辅助工具实测

- director_lineart：Full 与 Curated 来源图片各返回 832×1216 RGB 线稿；已目视确认人物与书架线条，细线较淡。
- director_bg-removal：每张来源图返回三张 832×1216 RGBA PNG，全部 alpha 范围 0–255；已目视检查两个来源的第 0 张，背景去除且人物保留。接口 ZIP 只命名 image_0/1/2，未提供语义标签，工具不猜测哪张对应 Masked/Generated/Blend。全部文件保留。
- 证据：https://show.toddout.work/a/mu086wag7051 。
- 原始调用、响应与结果在 Mac .runtime/v5-e2e/*-director_*.receipt.json 及关联路径。
- sketch：两种来源均生成明显的铅笔草图，人物和书架结构保留，已目视确认。
- colorize：以两张线稿为输入得到彩色图，已目视确认；Curated 来源的书仍被画为红色而非提示中的蓝色，部分书架色彩较强。不是精确按标签上色的保证。
- emotion：早期侧脸/全身输入变化轻微；定向使用正面中性头像、happy;;open mouth, smiling, laughing、defry=0 后，两种 V5 来源都产生明显笑脸。气泡中文字也被改动，说明操作不局限于面部。
- declutter：定向图中浮动文字与气泡均去除；Full 来源还改动了背景和衣服，应保留原图审看。declutter-keep-bubbles：两个来源都去除文字并保留空气泡轮廓。原始无文字输入的早期测试只证明接口返回，本次定向测试才提供效果证据。
- 补充图集：https://show.toddout.work/a/mu08zhsw5eb2 。

## 定向语义验收补齐

- 使用各 V5 模型生成的正面中性人物、浮动文字与对白气泡作为输入。emotion_closeup、declutter_text、declutter-keep-bubbles_text 共六次真实 MCP 调用完成，原图和结果均已目视检查。
- 图集：https://show.toddout.work/a/mu0a4pf1f4c9 。
- 所有 Director 操作现在都有真实返回及针对其用途的可见效果记录；依然是独立工具，不宣称 V5 原生精确编辑或只改目标区域。
