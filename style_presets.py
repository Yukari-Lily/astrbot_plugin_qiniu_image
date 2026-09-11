"""内置绘图风格及其注入规则。"""

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple


STYLE_MODES = ("auto", "explicit_only", "disabled")
STYLE_STRENGTHS = ("subtle", "normal", "strong")


CLEAN_LINE_GUIDANCE = """
动画画面清洁度约束：以 large simple color masses、low visual frequency、restrained edge density、clear edge hierarchy、minimal internal contour lines、broad shadow shapes 组织画面。外轮廓明确，内部结构线少而准确；头发先分大束，衣物只保留解释体积所需的主要褶皱，每种材质用少量连续的大块明暗表现。抑制微纹理和微对比（suppress micro-texture and micro-contrast），避免碎阴影、密集发丝、细碎反光、无必要的镜面高光（no unnecessary specular highlights）、排线和脏灰过渡。背景云层、建筑和水面概括为少量大形，细节与对比低于主体；拼贴使用少量大窗口及充分留白，避免重复人脸、密集小窗和装饰碎片抢夺焦点。干净来自形体概括与细节取舍，不靠锐化或堆叠 highly detailed。明确要求雕刻、半调或故障时仅在局部保留必要特征；局部改图只约束修改区域，保持其他区域原貌。
""".strip()

# Also sent directly to the image model so compression by the optimizer cannot
# remove the execution constraint. Explicit styles and unchanged edit areas win.
LINE_EXECUTION_GUIDANCE = (
    "线条执行约束（不画成文字）：在不违背用户明确画风及保留项的前提下，"
    "轮廓使用单次、准确、连续的线条；消除草稿复线、杂乱排线、密集碎发线、放射速度线与无意义装饰线。"
    "头发按大束、衣褶按少量主要转折、阴影按大块组织；减少高频背景和碎片，脸部与人物交界保持清晰。"
    "不删除辨识人物所需的发饰、服装结构和标志物；明确要求雕刻或线性特效时仅在必要区域有序使用，"
    "局部编辑仅约束本次修改部分，其余保持原貌。"
)


QUALITY_GUIDANCE = """
全局审美与质量底线（无论是否选用内置风格都必须遵守）：
1. 画面保持高级、清爽、精心设计：单一视觉中心、明确层级、受控的细节密度、清楚的主体轮廓和有意安排的留白。
2. 绘制动漫角色或已有二次元 IP 人物时，默认采用干净的日系二次元插画语言：准确利落的线稿或边缘、稳定的赛璐璐/平涂关系、通透肤色、统一配色。只有输入方案明确要求写实摄影、油画或厚涂时才更换媒介。
3. 故障、喷溅、半调、颗粒、磨损、拼贴等效果必须克制、局部、边界清楚，服务于构图；即使所选风格包含这些元素，也不能演变成覆盖全画面的污渍、糊块和随机噪声。
4. 严格抑制脏乱廉价感：浑浊棕黑覆盖、油腻厚重笔触、灰脏肤色、无目的颗粒磨损、过量锐化、杂乱背景、无层次高频细节、互相打架的光源、堆叠过量的故障与特效。
5. 日系动漫、漫画或游戏角色使用官方日文作品名和角色名；画内需要标题或装饰文字时只使用准确的官方日文原文，或输入方案明确指定的文字，不使用中文译名或中文装饰字。不要擅自添加无意义文字。
6. 若题材不是二次元人物（例如商品、美食、建筑或输入方案明确要求摄影），选择适合题材的专业视觉语言，但仍保持干净、克制和清晰层级。
7. “masterpiece、best quality、highly detailed、anime style、cinematic、8k、插画、高清”等只是泛化质量词，不属于输入方案指定的完整画风，不能用它们作为跳过内置风格的理由。
""".strip() + "\n" + CLEAN_LINE_GUIDANCE + "\n" + LINE_EXECUTION_GUIDANCE


SAFE_REFRAME_GUIDANCE = """
安全转译规则：目标是在不放弃出图的前提下，优先通过措辞层面让提示词通过审核：先用含蓄、同义或隐晦的词语重述，保持原画面（服装、场景、姿势、构图）不变；只有措辞调整仍被拒绝时，才按递进级别逐步收敛画面。优先以二次元插画语境表达（动画角色插画、角色立绘、轻小说/漫画封面、动画剧照、角色设计图、时尚杂志内页插画），不要将私密场景、暴露服装与挑逗姿势叠加组合，至少弱化其中一项。遇到色情、裸露、透明衣物、脱衣或明显性暗示时，可将其转译为非露骨、非色情、衣着完整的审美替代方案，并保留仍然安全的主体身份、配色、场景、情绪和构图意图；可改为端庄睡衣、完整家居服、时尚写真、浪漫氛围或含蓄而不性化的姿态。不要保留裸体、可见私密部位、性行为、恋物特写或对未成年/年龄不明角色的性化表达。对年轻外貌或年龄不明的角色尤其使用端庄服装和非挑逗姿态。只输出可直接送入图片模型的安全提示词，不要道歉、解释政策或拒绝。
""".strip()


@dataclass(frozen=True)
class StylePreset:
    id: str
    name: str
    aliases: Tuple[str, ...]
    suitable_for: str
    avoid_when: str
    prompt: str
    auto_preference: int = 0


STYLE_PRESETS: Tuple[StylePreset, ...] = (
    StylePreset(
        id="clean_anime_wallpaper",
        name="净色动画壁纸",
        aliases=("净色动画电脑壁纸", "清爽动画壁纸", "极简净色壁纸"),
        suitable_for="动画角色电脑壁纸、干净构图与配色、低细节密度、大色块和充足留白",
        avoid_when="输入方案明确要求繁复纹理、密集拼贴、写实摄影或高信息密度海报",
        prompt=(
            "生成一张构图和色彩非常干净的电脑壁纸，电脑壁纸的主要角色是动画角色。"
            "large simple color masses, low visual frequency, restrained edge density, "
            "clear edge hierarchy, minimal internal contour lines, broad shadow shapes, "
            "suppress micro-texture and micro-contrast, no unnecessary specular highlights. "
            "以少量协调的纯净色块和宽阔留白突出角色，外轮廓清楚、内部线条精简，"
            "适合桌面图标的安静背景；电脑壁纸默认横向构图，输入方案指定比例或用途时以其要求为准。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="literary_commerce",
        name="文艺电商联动物料",
        aliases=("电商联动宣传物料", "文艺商业联动", "文艺电商海报"),
        suitable_for="品牌联动、新品发布、角色与商品共同出现的竖版商业宣传图；用户希望文艺、清新或编辑设计感",
        avoid_when="纯角色立绘、没有商品或品牌信息的场景、用户指定强烈工业或故障风格",
        prompt=(
            "文艺清新的商业编辑设计与品牌联名海报质感，主体和商品形成明确视觉层级，保留充足而整洁的文案区域，"
            "使用克制的品牌色、柔和自然光、精致留白与杂志式网格排版；用户给出的标题、宣传语、价格及成分说明层级清楚、"
            "文字易读，整体兼具角色魅力、商品吸引力和可直接发布的广告完成度。"
        ),
    ),
    StylePreset(
        id="russian_constructivism",
        name="俄国构成主义",
        aliases=("俄国构成主义", "俄罗斯构成主义", "俄国解构主义", "复古构成主义"),
        suitable_for="科技产品、政治宣传画式海报、需要工业力量感和强烈几何张力的主题",
        avoid_when="柔美写实摄影、空灵治愈、需要大量柔和渐变的主题",
        prompt=(
            "俄国构成主义平面设计插画与复古宣传海报，大块锐利三角形、圆形及粗重对角线切割构成非对称平衡；"
            "限定高饱和宝蓝、深黑与做旧米白三色，扁平高对比、锐利边缘，带细腻丝网印刷颗粒和纸张磨损纹理，"
            "呈现强烈工业力量感。"
        ),
    ),
    StylePreset(
        id="glitch_rectangles",
        name="错位矩形·强故障版",
        aliases=("错位矩形风格", "错位矩形风格第一版", "错位矩形第一版", "强故障错位矩形"),
        suitable_for="赛博朋克人物海报、数字损坏、超现实忧郁、高饱和天空与现代平面设计",
        avoid_when="传统绘画、自然写实摄影、安静简约且不希望出现数字噪点的画面",
        prompt=(
            "故障艺术与赛博朋克动漫美学，多个错位矩形窗口和几何切片形成数字碎片化构图；加入像素排序、RGB色偏、"
            "局部横向色带和少量边缘色偏，故障限于背景与窗口边缘，不覆盖人脸或变成密集电流乱线；"
            "以极简米白背景对比高饱和湛蓝天空和厚重积雨云，氛围超现实、忧郁而深邃。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="window_overlay_poetic",
        name="错位矩形·诗意窗口版",
        aliases=("错位矩形风格", "错位矩形风格第二版", "错位矩形第二版", "窗口重叠", "诗意窗口拼贴"),
        suitable_for="二次元全身动态人物、宁静诗意、蓝天积雨云、现代平面艺术",
        avoid_when="写实商业摄影、厚重暗黑、追求三维体积和复杂写实光照的画面",
        prompt=(
            "二次元平面艺术插画，采用窗口重叠与数字拼贴构图；以多个错位矩形框重构角色轮廓，局部透明视窗显露清朗"
            "蓝天和积雨云，仅用少量黑色长条与克制边缘色偏，不铺满电子扫描线；克莱因蓝和纯白主色，动态姿态与干净单线轮廓，"
            "画面简洁明快、宁静且富有诗意。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="photo_lineart_mixed_media",
        name="混合媒介·照片与素描",
        aliases=("混合媒介-照片+素描", "照片加素描", "照片与白色线稿", "摄影线稿混合媒介"),
        suitable_for="人物肖像、黄昏海岸、城市街景、空灵怀旧和梦幻忧郁的叙事画面",
        avoid_when="纯矢量海报、纯赛璐璐立绘、要求所有元素都清晰锐利的画面",
        prompt=(
            "混合媒介艺术：主体以极简纤细的半透明白色线稿呈现，仅保留少量局部发光色；背景为写实大光圈虚化摄影，"
            "采用电影感暮蓝与金黄渐变光影和明显倾斜的荷兰角构图。锐利白色线条与柔软模糊实景形成强烈反差，"
            "兼具怀旧 Lo-fi、空灵、梦幻和忧郁气质。"
        ),
    ),
    StylePreset(
        id="engraving_halftone_duotone",
        name="半调双色雕刻",
        aliases=("半调双色雕刻风格", "半调雕刻线稿", "Engraving Halftone Style"),
        suitable_for="单一主体、侧脸或轮廓鲜明的人像、标志性物件、现代主义极简海报",
        avoid_when="复杂多人场景、写实全彩环境、必须表现丰富材质颜色的画面",
        prompt=(
            "极简主义平面海报与半调雕刻线稿风格，以密集的同心圆或平行弧线构成画面，通过线条粗细和疏密变化勾勒"
            "主体轮廓、结构与阴影；严格使用背景色和线条色组成的高对比双色方案，构图简洁有力，兼具矢量质感、"
            "雕刻立体感和前卫现代主义设计感。"
        ),
    ),
    StylePreset(
        id="risograph_magazine",
        name="半调杂志",
        aliases=("半调杂志风格", "Risograph半调杂志", "复古半调杂志"),
        suitable_for="居中单体产品、器物、音乐或咖啡主题、复古杂志封面和波普海报",
        avoid_when="需要写实空间纵深、复杂叙事场景、柔和低对比摄影",
        prompt=(
            "现代复古平面海报与 Risograph 半调网点印刷风格，主体居中，以深蓝和米白半调纹理表现；粗糙颗粒米色纸张"
            "背景配明黄色几何实心拱门，周围仅点缀少量有序轨道弧线与品红四芒星，不叠加交错线网；上下使用复古粗体"
            "无衬线排版及黄色高光色块（仅排版用户实际要求呈现的文字），构图极简、色彩强烈，具有波普杂志封面的冲击力。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="pop_ink_splash",
        name="波普与水墨喷溅",
        aliases=("波普+水墨喷溅", "波普水墨喷溅", "日系波普水墨"),
        suitable_for="时尚人物、都市动态、需要高信息密度和瞬时爆发力的日系插画",
        avoid_when="安静低饱和、古典工笔、写实摄影或严格极简留白",
        prompt=(
            "现代日系混合媒介插画，采用倒置动态构图与扁平波普逻辑；以高饱和明黄为主，克莱因蓝和大红强烈对冲，"
            "以赛璐璐平涂和少量大块拼贴为主体，仅在背景局部使用半调波点或水墨喷溅，避免细碎喷溅和多层乱线；光影利落，"
            "兼具都市轻盈感、瞬时爆发力和符号化视觉冲击。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="klein_order",
        name="克莱因秩序",
        aliases=("克莱因秩序", "克莱因蓝秩序", "Klein Blue极简"),
        suitable_for="单个动漫角色、夏日、孤独清冷、强透视和动画分镜感",
        avoid_when="复古颗粒、繁复拼贴、柔光写实或暖色主导的画面",
        prompt=(
            "现代极简主义二次元赛璐璐插画，以几何切割和大面积负空间组织构图；克莱因蓝与高亮纯白构成双色核心，"
            "正午直射光形成锐利硬边阴影和极高明暗对比，采用强烈仰拍透视强调线条延伸；色块平整、线条利落、"
            "无杂色颗粒，呈现夏日清冷、孤独超现实和大师级动画分镜感。"
        ),
    ),
    StylePreset(
        id="cream_red_circuit",
        name="米白红色电路图",
        aliases=("米白-红色电路图", "米白红色电路图", "红色电路图风格"),
        suitable_for="悬浮或倒立的动漫人物、赛博流行、Y2K 极客文化和二维海报",
        avoid_when="自然写实姿态、古典场景、强调柔和体积光或复杂三维环境",
        prompt=(
            "极简平面化日系赛博流行插画，主体以失重倒立或悬浮姿势形成对角线动势；温暖奶油米色大面积留白中设置"
            "红色或与主体主要衣物同色的巨大不规则有机色块，使服装边缘与色块无缝交融；色块内部安排少量白色电路节点与"
            "清晰分支，保持节点间距，避免交叉线网和线条穿过人物面部；仅在用户要求时添加代码或符号，"
            "使用干净赛璐璐平涂和清晰黑色线稿，呈现 Y2K 极客海报张力。"
        ),
        auto_preference=2,
    ),
    StylePreset(
        id="y2k_pop_pixel",
        name="Y2K 与波普艺术",
        aliases=("Y2K+波普艺术", "Y2K波普艺术", "Y2K复古像素波普"),
        suitable_for="潮流人物、复古街景、千禧网络文化、躁动且高对比的平面视觉",
        avoid_when="含蓄文艺、传统绘画、自然写实摄影或纯净无纹理画面",
        prompt=(
            "Y2K 复古像素艺术与波普平面设计混合美学，扁平化二维构图，以高饱和电光蓝铺底并用荧光橘点缀；只保留"
            "纯色色块和粗犷黑色轮廓线，加入显眼像素抖动噪点、半调网点、线稿式复古街景，以及黑色星芒、橘色同心"
            "圆弧、螺旋线和 UI 排版色块，营造躁动、潮流的千禧年网络复古氛围。"
        ),
    ),
    StylePreset(
        id="cel_thought_burst",
        name="赛璐璐平涂与思维爆发",
        aliases=("赛璐璐平涂+思维爆发", "赛璐璐思维爆发", "思维爆发风格"),
        suitable_for="沉思人物、意识流、梦境、极简与高密度细节并存的日系独立插画",
        avoid_when="写实摄影、厚涂三维、中心放射式爆炸或要求完全无颗粒的画面",
        prompt=(
            "清透细腻的日系独立插画与超现实波普艺术，主体安静沉思，头部或发丝向上连接有层次的少量大块失重拼贴；"
            "避免中心放射，以轮廓明确的建筑切片、少量流动丝带和几何块表现思维扩展，辅以少量十字四芒星，"
            "不用密集碎片、细线建筑网或草稿复线。下方及四周保留大面积纯白负空间，以准确单线轮廓和无渐变赛璐璐平涂表现；"
            "冰蓝、钴蓝为主，搭配芥末黄和淡土金，边缘干净，仅在背景局部保留轻微 Risograph 纸张肌理。"
        ),
        auto_preference=2,
    ),
)


def _match_key(text: str) -> str:
    return re.sub(r"[\s+＋·・_—\-/]+", "", text).casefold()


def find_explicit_presets(user_prompt: str) -> Tuple[StylePreset, ...]:
    """按名称或别名找用户明确点名的风格，不负责语义猜测。"""
    prompt_key = _match_key(user_prompt)
    if not prompt_key:
        return ()
    return tuple(
        preset
        for preset in STYLE_PRESETS
        if any(_match_key(alias) in prompt_key for alias in (preset.name, *preset.aliases))
    )


def _catalog_lines(presets: Sequence[StylePreset]) -> List[str]:
    lines: List[str] = []
    for preset in presets:
        aliases = "、".join(preset.aliases)
        lines.extend(
            (
                f"- {preset.name}（id: {preset.id}；别名：{aliases}；自动偏好：{'高' if preset.auto_preference else '普通'}）",
                f"  适合：{preset.suitable_for}",
                f"  避免：{preset.avoid_when}",
                f"  视觉片段：{preset.prompt}",
            )
        )
    return lines


def style_catalog_text(*, concise: bool = False) -> str:
    """供聊天工具与文档复用的风格目录，保持与实际预设同源。"""
    if concise:
        return "\n".join(
            f"- {preset.name}" for preset in STYLE_PRESETS
        )
    return "\n".join(_catalog_lines(STYLE_PRESETS))


_STYLE_MARKER_RE = re.compile(
    r"\[\[\s*STYLE_PRESET\s*:\s*([a-z0-9_-]+)\s*\]\]",
    re.IGNORECASE,
)


def valid_style_choice(text: str) -> bool:
    """An enabled router must report one known choice, including explicit none."""
    choices = _STYLE_MARKER_RE.findall(text)
    return len(choices) == 1 and choices[0].casefold() in {"none", *(p.id for p in STYLE_PRESETS)}


def clean_style_metadata(text: str) -> Tuple[str, Tuple[StylePreset, ...]]:
    """移除内部风格标记及风格名称，避免图片模型把名称画成文字。"""
    preset_by_id = {preset.id: preset for preset in STYLE_PRESETS}
    selected: List[StylePreset] = []
    for preset_id in _STYLE_MARKER_RE.findall(text):
        preset = preset_by_id.get(preset_id.casefold())
        if preset and preset not in selected:
            selected.append(preset)

    cleaned = _STYLE_MARKER_RE.sub("", text)
    labels = sorted(
        {
            label
            for preset in STYLE_PRESETS
            for label in (preset.name, *preset.aliases)
            if label
        },
        key=len,
        reverse=True,
    )
    for label in labels:
        pattern = re.escape(label).replace(r"\ ", r"\s*")
        cleaned = re.sub(
            rf"{pattern}(?:\s*(?:风格|美学|视觉风格))?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(
        r"^[\s,，。:：;；、\-—·|/]*(?:(?:style|风格|美学)\s*[,，。:：;；、\-—·|/]*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^\S\r\n]{2,}", " ", cleaned).strip()
    return cleaned, tuple(selected)


def build_style_guidance(
    user_prompt: str,
    *,
    mode: str,
    strength: str,
    has_image: bool,
    drawing_task: dict | None = None,
) -> str:
    """构造给优化模型看的风格路由规则。"""
    if mode == "disabled":
        return ""

    # For structured calls, explicit_only must not mistake a creative choice
    # made by the chat model for a style explicitly requested by the user.
    explicit_text = user_prompt
    if drawing_task is not None and mode == "explicit_only":
        explicit_text = "\n".join(drawing_task.get("user_requirements", []) + drawing_task.get("changes", []))
    explicit = find_explicit_presets(explicit_text)
    if mode == "explicit_only" and not explicit:
        return ""

    presets: Sequence[StylePreset] = (
        sorted(STYLE_PRESETS, key=lambda preset: preset.auto_preference, reverse=True)
        if mode == "auto"
        else explicit
    )
    strength_rule = {
        "subtle": "只借用少量最有辨识度的视觉特征，不让风格压过主体和内容。",
        "normal": "完整使用核心视觉语言，但删除与输入方案无关或冲突的细节。",
        "strong": "在不改变输入方案硬性要求的前提下，充分使用所选风格的构图、色彩和材质语言。",
    }[strength]

    lines = [
        "",
        "以下是内置风格库及路由规则。风格库是可选参考，不是必须全部加入的关键词列表：",
        "1. 输入方案明确点名某个内置风格且不是在否定它时，优先采用该风格。",
        "2. 输入方案明确给出具体且有辨识度的其他画风时，忠实保留并停止自动叠加内置风格。",
        "3. 输入方案由主聊天模型基于人设、上下文、角色考据和创作委托形成，其中确定的画风、构图及视觉选择必须保留。anime style、highly detailed、masterpiece、cinematic、插画、高清等泛化词不算具体风格。",
        "4. 输入方案未确定具体画风时，主动选择一个相容的内置风格；已有明确视觉方案时不要另套模板。方案要求换风格时参照其中记录的上一版风格选择不同方案；仅在完整风格与主体或硬性要求冲突时借用部分特征。",
        "5. 自动选择时先从标记为‘自动偏好：高’的相容风格中选择一个；仅当高偏好项全部不相容才考虑普通项。只有用户明确要求其他具体画风、纯写实、忠实复刻原画风、不要风格化、或全部候选都明显冲突时才选择零个。图片本身不是画风来源。",
        "6. 最多选择一个风格，不混合多个内置风格。输入方案中的主体身份、品牌、商品、数量、动作、场景、构图和配色等明确要求永远优先；方案要求画进图片的文字必须逐字保留。",
        "7. 只吸收所选风格中适用的视觉属性，不复制示例主体，不添加输入方案未包含的角色、品牌、文字或物件。",
        "8. 把所选风格的构图、色彩、线条、材质等视觉属性自然展开写入提示词；正文中绝对不要写出风格的中文名称、别名或 id，避免图片模型把名称画进画面。",
        "9. 仅在输出开头添加插件内部标记 [[STYLE_PRESET:id]]，其中 id 替换为所选风格 id；未选择内置风格时使用 [[STYLE_PRESET:none]]。这是上一条的唯一例外，标记只供插件读取，不能代替视觉描述，插件会在出图前删除它。",
        "10. 严格遵守另外提供的全局审美与质量提示，内置风格不能覆盖其中的约束。",
        f"11. 当前风格强度：{strength}。{strength_rule}",
    ]
    if has_image:
        lines.append("12. 当前是改图：除非输入方案明确点名风格或要求整体重绘/风格化，否则不要改变原图画风。")
    if drawing_task is not None and mode == "auto":
        lines.append(
            "结构化任务的风格来源规则优先于上文的‘方案确定画风必须保留’："
            "user_requirements、changes 中用户明确指定的画风最高，随后是目标作品保留项；图片本身不提供画风依据。"
            "prompt 或 creative_choices 中模型自行加入的普通动画主视觉、cinematic、Pixar/3D 等并非用户指定；"
            "人物作品的原始媒介及形象版本也不自动锁定本图媒介。auto 模式下这些自选画风可让位于相容的内置偏好，"
            "保留主体、动作、场景及已确定构图，仅调整相容的色彩组织、线条和材质。"
            "只有上述更高优先级约束才可以阻止自动选风格；不要以自选普通画风为由输出 STYLE_PRESET:none。"
            "不要把整份英文 prompt 当成用户原话。"
        )
    if explicit:
        names = "、".join(preset.name for preset in explicit)
        lines.append(f"检测到输入方案可能明确提及：{names}。仍需识别否定语义，并以方案整体含义为准。")
    lines.append("\n可选风格：")
    lines.extend(_catalog_lines(presets))
    return "\n".join(lines)
