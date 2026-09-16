"""内置绘图风格及其注入规则。"""

import re
from dataclasses import dataclass
from typing import List, Tuple


STYLE_STRENGTHS = ("subtle", "normal", "strong")


CLEAN_LINE_GUIDANCE = """
动画画面清洁度约束：以 large simple color masses、low visual frequency、restrained edge density、clear edge hierarchy、minimal internal contour lines、broad shadow shapes 组织画面。外轮廓明确，内部结构线少而准确；头发先分大束，衣物只保留解释体积所需的主要褶皱，每种材质用少量连续的大块明暗表现。抑制微纹理和微对比（suppress micro-texture and micro-contrast），避免碎阴影、密集发丝、细碎反光、无必要的镜面高光（no unnecessary specular highlights）、排线和脏灰过渡。背景云层、建筑和水面概括为少量大形，细节与对比低于主体；拼贴使用少量大窗口及充分留白，避免重复人脸、密集小窗和装饰碎片抢夺焦点。干净来自形体概括与细节取舍，不靠锐化或堆叠 highly detailed。明确要求雕刻、半调或故障时仅在局部保留必要特征；局部改图只约束修改区域，保持其他区域原貌。
""".strip()


QUALITY_GUIDANCE = """
全局审美与质量底线（无论是否选用内置风格都必须遵守）：
1. 画面保持高级、清爽、精心设计：单一视觉中心、明确层级、受控的细节密度、清楚的主体轮廓和有意安排的留白。
2. 绘制动漫角色或已有二次元 IP 人物时，默认采用干净的日系二次元插画语言：准确利落的线稿或边缘、稳定的赛璐璐/平涂关系、通透肤色、统一配色。只有输入方案明确要求写实摄影、油画或厚涂时才更换媒介。
3. 故障、喷溅、半调、颗粒、磨损、拼贴等效果必须克制、局部、边界清楚，服务于构图；即使所选风格包含这些元素，也不能演变成覆盖全画面的污渍、糊块和随机噪声。
4. 严格抑制脏乱廉价感：浑浊棕黑覆盖、油腻厚重笔触、灰脏肤色、无目的颗粒磨损、过量锐化、杂乱背景、无层次高频细节、互相打架的光源、堆叠过量的故障与特效。
5. 日系动漫、漫画或游戏角色使用官方日文作品名和角色名；画内需要标题或装饰文字时只使用准确的官方日文原文，或输入方案明确指定的文字，不使用中文译名或中文装饰字。不要擅自添加无意义文字。
6. 若题材不是二次元人物（例如商品、美食、建筑或输入方案明确要求摄影），选择适合题材的专业视觉语言，但仍保持干净、克制和清晰层级。
""".strip() + "\n" + CLEAN_LINE_GUIDANCE


SAFE_REFRAME_GUIDANCE = """
安全转译规则：目标是在不放弃出图的前提下，优先通过措辞层面让提示词通过审核：先用含蓄、同义或隐晦的词语重述，保持原画面（服装、场景、姿势、构图）不变；只有措辞调整仍被拒绝时，才按递进级别逐步收敛画面。优先以二次元插画语境表达（动画角色插画、角色立绘、轻小说/漫画封面、动画剧照、角色设计图、时尚杂志内页插画），不要将私密场景、暴露服装与挑逗姿势叠加组合，至少弱化其中一项。遇到色情、裸露、透明衣物、脱衣或明显性暗示时，可将其转译为非露骨、非色情、衣着完整的审美替代方案，并保留主体身份（角色名、作品名、外观设定与人数）与尽可能多的配色、场景、情绪和构图意图；可改为端庄睡衣、完整家居服、时尚写真、浪漫氛围或含蓄而不性化的姿态。不要保留裸体、可见私密部位、性行为、恋物特写或对未成年/年龄不明角色的性化表达。对年轻外貌或年龄不明的角色尤其使用端庄服装和非挑逗姿态。
保真约束：改写不是重新创作。被拒绝的原因是成人内容，不是主体——不要把主体、角色名、作品名、人数替换、删减或泛化成别的角色。必须使用与被拒绝提示词相同的书写语言，原文是中文就输出中文，是日文就输出日文，不得整体改写成英文。必须逐字保留原文中的角色名、作品名、产品名等专有名词。不得改变原文的动作、构图层级和场景意图；只允许弱化或收敛与审核相关的着装与体态描述。
只输出可直接送入图片模型的安全提示词，不要道歉、解释政策或拒绝。
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
        suitable_for="至少两名人物的合影或群像壁纸；干净配色、大色块和充足留白，仅作低优先候选",
        avoid_when="单人、无人、人数不明；即使指定单人壁纸也不用。不得为使用本风格添加人物；繁复纹理、密集拼贴或写实摄影也不适用",
        prompt=(
            "生成一张构图和色彩非常干净的电脑壁纸，电脑壁纸的主要角色是动画角色。"
            "large simple color masses, low visual frequency, restrained edge density, "
            "clear edge hierarchy, minimal internal contour lines, broad shadow shapes, "
            "suppress micro-texture and micro-contrast, no unnecessary specular highlights. "
            "以少量协调的纯净色块和宽阔留白突出角色，外轮廓清楚、内部线条精简，"
            "适合桌面图标的安静背景；电脑壁纸默认横向构图，输入方案指定比例或用途时以其要求为准。"
        ),
        auto_preference=-1,
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
            "横向拉伸噪点和彩虹电流纹理，以极简米白背景对比高饱和湛蓝天空和厚重积雨云，氛围超现实、忧郁而深邃。"
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
            "蓝天和积雨云，点缀极简黑色长条、细密电子扫描线与克制色偏纹理；克莱因蓝和纯白主色，动态速写姿态，"
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
            "背景配明黄色几何实心拱门，周围点缀极细交错轨道线、微小品红四芒星和条形码图形；上下使用复古粗体"
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
            "融合赛璐璐平涂、半调波点、水墨喷溅、纸张肌理和数码后期叠加，形成多层二维拼贴空间；光影利落，"
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
            "红色或与主体主要衣物同色的巨大不规则有机色块，使服装边缘与色块无缝交融；内部叠加白色线性电路节点、几何网状"
            "分支、等宽代码与二进制符号，使用干净赛璐璐平涂和清晰黑色线稿，呈现 Y2K 极客海报张力。"
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
            "清透细腻的日系独立插画与超现实波普艺术，主体安静沉思，头部或发丝向上无缝解体为高密度失重碎片；"
            "避免中心放射，使用相互垂直穿插的平面拼贴，组合极细线建筑切片、流动丝带曲线、锐利几何碎块和纤细"
            "十字四芒星。下方及四周保留大面积纯白负空间，以 Ligne claire 极细线稿和无渐变赛璐璐平涂表现；"
            "冰蓝、钴蓝为主，搭配芥末黄和淡土金，边缘干净，整体覆盖均匀复古噪点与 Risograph 纸张肌理。"
        ),
        auto_preference=2,
    ),
)


def preference_label(preset: StylePreset) -> str:
    """风格在自动偏好里的档位，供目录行与工具说明逐字复用。"""
    if preset.auto_preference < 0:
        return "低优先（仅限至少两名人物）"
    if not preset.auto_preference:
        return "普通"
    return "默认优先"


def style_catalog_text(*, include_prompts: bool = False) -> str:
    """供聊天工具与文档复用的风格目录；id 是出图时唯一可用的风格标识。"""
    lines: List[str] = []
    for preset in sorted(STYLE_PRESETS, key=lambda item: item.auto_preference, reverse=True):
        lines.extend((
            f"- {preset.name}（id: {preset.id}；{preference_label(preset)}）",
            f"  适合：{preset.suitable_for}",
            f"  避免：{preset.avoid_when}",
        ))
        if include_prompts:
            lines.append(f"  别名：{'、'.join(preset.aliases)}")
            lines.append(f"  风格参考（只将适用的视觉属性自然融入执行提示词，不复制名称或整段模板）：{preset.prompt}")
    return "\n".join(lines)


_LEGACY_STYLE_LEADS = {
    "subtle": "画面风格（只作轻微参考，不得改变上述主体、动作、构图与明确要求）：",
    "normal": "画面风格（不得改变上述主体、动作、构图与明确要求）：",
    "strong": "画面风格（不得改变上述主体、动作、构图与明确要求，可以此风格主导画面语言）：",
}
STYLE_LEADS = {
    "subtle": "内置画风（保留预设的媒介、配色与视觉结构，仅减轻表现强度；用户明确要求优先）：",
    "normal": "内置画风（完整遵照以下预设的媒介、配色、纹理与视觉结构；用户明确要求优先，正文中的协调建议不得覆盖画风）：",
    "strong": "内置画风（完整遵照并突出以下预设特征，以其主导画面语言；用户明确要求优先，正文中的协调建议不得覆盖画风）：",
}
IMAGE_AUTHORITY_LINE = (
    "以用户图片中主体的实际外观为准，不要依据文字知识改写；除方案明确要求改动的部分外，其余保持原图。"
)
# 用户发参考图但要求画成别的场景时用这一句：主体照图，构图按方案。
IMAGE_REFERENCE_LINE = (
    "以用户图片中主体的实际外观为准，不要依据文字知识改写；场景、构图、姿势与镜头按上述方案重建，"
    "不要保留原图的构图与背景。"
)
QUALITY_SECTION = ("全局审美与质量底线（仅在不与上述方案冲突时应用）：\n"
                   + QUALITY_GUIDANCE.split("\n", 1)[1])
QUALITY_HEAD = QUALITY_SECTION.split("\n", 1)[0]
_LEGACY_STYLE_QUALITY_HEADS = (
    "画风内的协调（不得覆盖用户明确要求或上述内置画风）：",
    "画风内的协调与清洁度上限（不得覆盖用户明确要求或上述内置画风）：",
)
# 插件追加的内容固定在方案末尾，由标题行或整句引入，据此可以把它们整体切掉。
# 只认插件自己写的这几个字符串：方案里出现"画面风格（"这类字样时不能被误切。
_TAIL_HEADS = (*STYLE_LEADS.values(), *_LEGACY_STYLE_LEADS.values(), QUALITY_HEAD,
               *_LEGACY_STYLE_QUALITY_HEADS, IMAGE_AUTHORITY_LINE, IMAGE_REFERENCE_LINE)
_SECTION_RE = re.compile(rf"(?:\A|\n{{1,2}})(?={'|'.join(re.escape(head) for head in _TAIL_HEADS)})")


def _normalize(text: str) -> str:
    """统一换行符：插件自己追加的两段始终用 \n，不能让方案的 CRLF 影响切分。"""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def plan_body(text: str) -> str:
    """取回方案正文：去掉插件自己追加的风格段、图片约束句与兜底段。

    兼容用户重新输入上一份完整提示词，交给文字模型前取回正文；``compose_prompt`` 开头也走一遍，重复拼装因此不会叠加。
    """
    return _SECTION_RE.split(_normalize(text), maxsplit=1)[0].rstrip()


def compose_prompt(
    plan: str,
    *,
    has_image: bool,
    integrated: bool = False,
    keep_layout: bool = True,
) -> str:
    """聊天稿已融合视觉要求；关键词原文仍追加既有质量段。

    风格 ID 仅用于记录，不在提交阶段重新施加模板。
    """
    text = plan_body(plan)
    if not text:
        raise ValueError("绘图方案为空")

    if has_image:
        # keep_layout=False 是参考创作：主体照图，场景与构图按方案重建。
        line = IMAGE_AUTHORITY_LINE if keep_layout else IMAGE_REFERENCE_LINE
        if line not in text:
            text += "\n\n" + line
    if not integrated:
        text += "\n\n" + QUALITY_SECTION
    return text
