from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape


OUT = Path("paper_draft_erc.docx")


def p(text="", style=None):
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return (
        "<w:p><w:pPr>"
        f"{style_xml}"
        "</w:pPr><w:r><w:t xml:space=\"preserve\">"
        f"{escape(text)}"
        "</w:t></w:r></w:p>"
    )


def table(rows):
    xml = [
        "<w:tbl>",
        '<w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr>',
    ]
    for row in rows:
        xml.append("<w:tr>")
        for cell in row:
            xml.append(
                '<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>'
                f'{p(str(cell))}</w:tc>'
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def make_doc():
    body = []
    body.append(p("基于多域子锚点原型增强的对话情感识别方法研究", "Title"))
    body.append(p("摘要", "Heading1"))
    body.append(p("对话情感识别旨在根据对话历史、说话人交互关系以及当前话语内容判断目标话语的情感类别，是情感计算和自然语言处理中的重要任务。与普通句子级情感分类相比，对话情感识别面临上下文依赖强、情感转移频繁、相近情感类别边界模糊以及类别分布不均衡等问题。现有方法多通过循环网络、图神经网络或预训练语言模型建模话语之间的上下文关系，但在表示空间中仍容易出现相似情感类别重叠，从而影响模型的细粒度判别能力。"))
    body.append(p("针对上述问题，本文在情感锚点对比学习框架基础上，提出一种基于多域子锚点原型增强的对话情感识别方法。该方法首先利用 Sup-SimCSE-RoBERTa-large 编码历史上下文和目标话语，并通过提示模板获得上下文感知的话语表示；随后为每个情感类别构造多个语义子锚点，用以描述同一情感在不同语义域中的表达差异；最后结合原型增强监督对比学习、角度分离约束、锚点动量更新和 domain-gated 聚合机制，使话语表示能够向对应情感原型聚合，并在相近情感之间形成更清晰的类别间隔。"))
    body.append(p("本文在 IEMOCAP、MELD 和 EmoryNLP 三个公开数据集上进行实验。实验结果表明，本文方法在 IEMOCAP 和 MELD 数据集上分别取得 71.05 和 67.19 的加权 F1，均优于多个代表性基线模型；在 EmoryNLP 数据集上取得 39.98 的加权 F1，虽然略低于 EACL，但仍优于多数上下文建模和图结构方法。实验结果验证了多域子锚点原型增强机制在提升对话情感识别性能方面的有效性。"))
    body.append(p("关键词：对话情感识别；情感锚点；监督对比学习；预训练语言模型；原型学习"))

    body.append(p("1 绪论", "Heading1"))
    body.append(p("随着社交媒体、智能客服、在线教育和人机交互系统的快速发展，如何准确理解对话参与者在交流过程中的情感状态逐渐成为自然语言处理领域的重要研究问题。对话情感识别（Emotion Recognition in Conversation, ERC）要求模型根据一段多轮对话判断每个目标话语所表达的情感类别。该任务不仅能够为用户画像、舆情分析和情感陪伴等应用提供基础能力，也有助于提升智能系统在真实交互场景中的理解能力和响应质量。"))
    body.append(p("与传统文本情感分析任务相比，对话情感识别具有更复杂的语境依赖特征。首先，目标话语的情感往往不能仅由当前句子决定，而需要结合前文事件、说话人身份和交互关系进行推断。其次，对话中情感状态会随着话题推进和说话人反馈不断变化，模型需要捕捉这种动态转移过程。再次，不同情感类别之间存在明显语义相似性，例如 happy 与 excited、sad 与 frustrated 等类别在表达上容易混淆。最后，真实对话数据中不同情感类别分布并不均衡，低频类别样本较少，使模型容易偏向高频类别。"))
    body.append(p("已有研究主要从上下文建模和结构建模两个角度解决 ERC 问题。基于循环神经网络的方法通过建模历史状态刻画情感动态变化；基于图神经网络的方法将话语、说话人或时间关系构造成图结构，以捕捉远距离依赖；基于预训练语言模型的方法则借助大规模语义知识提升话语表示质量。近年来，对比学习和原型学习也被引入 ERC 任务，用于改善表示空间结构。然而，单一类别锚点难以充分刻画同一情感内部的表达多样性，当情感在不同语境下呈现不同强度和表达方式时，模型仍可能产生原型匹配不充分的问题。"))
    body.append(p("基于此，本文围绕相近情感类别区分困难和同类情感内部表达差异较大的问题展开研究，在现有情感锚点对比学习框架上进行改进。本文的主要工作包括：第一，构建面向 ERC 任务的提示式上下文编码方式，将历史话语、说话人信息和目标话语统一组织为预训练语言模型输入；第二，引入多域子锚点机制，为每个情感类别建立多个语义原型，以描述情感激活程度、交互方式、外显表达风格和上下文诱发转折等细粒度属性；第三，设计 domain-gated 聚合与锚点动量更新策略，使模型能够根据当前样本动态选择更相关的子锚点并更新原型表示；第四，在三个公开数据集上开展实验，验证本文方法的有效性和适用性。"))

    body.append(p("2 相关工作", "Heading1"))
    body.append(p("2.1 对话情感识别", "Heading2"))
    body.append(p("对话情感识别任务的核心在于从多轮交互中推断话语级情感标签。早期方法通常依赖人工特征、词典特征或浅层分类器，这类方法实现简单，但难以处理复杂上下文和隐式情感表达。随着深度学习的发展，DialogueRNN 等模型通过循环结构维护说话人状态和全局对话状态，从而捕捉情感随时间变化的过程。此类方法能够刻画连续对话中的情感动态，但在长距离依赖和多说话人关系建模方面仍存在不足。"))
    body.append(p("图结构方法进一步将对话建模为节点和边的组合，其中节点通常表示话语，边表示时间顺序、说话人关系或语义依赖。DialogueGCN、RGAT 和 DAG-ERC 等方法通过图神经网络传播上下文信息，在建模话语间交互关系方面取得了较好效果。然而，图结构构造往往依赖预设规则，不同数据集中的对话长度、说话人数量和标签体系差异较大，导致图关系设计的泛化能力受到一定限制。"))
    body.append(p("2.2 预训练语言模型与提示学习", "Heading2"))
    body.append(p("预训练语言模型在自然语言处理任务中表现出强大的语义建模能力。对于 ERC 任务，将 RoBERTa、DeBERTa 或 SimCSE 等模型作为基础编码器，可以显著提升话语表示质量。为了更好地适应情感分类任务，提示学习方法将任务转化为带有模板的语言理解问题，通过在输入中加入类似“speaker feels <mask>”的提示，使模型在预训练阶段形成的语言知识能够迁移到情感识别场景中。本文采用提示式输入组织方式，将历史上下文与目标话语共同输入 Sup-SimCSE-RoBERTa-large，并取 <mask> 位置的隐藏状态作为目标话语表示。"))
    body.append(p("2.3 对比学习与原型学习", "Heading2"))
    body.append(p("对比学习通过拉近同类样本表示、推远异类样本表示来改善表示空间结构，近年来被广泛应用于文本分类和情感分析任务。监督对比学习能够利用标签信息构造正负样本关系，使模型学习到更具判别性的类别边界。原型学习则通过类别中心或语义锚点表示类别先验，使样本表示能够围绕对应类别原型分布。EACL 将情感标签编码为锚点，并通过对比学习引导话语表示向对应情感锚点靠近，有效缓解了相近情感类别混淆问题。"))
    body.append(p("然而，真实对话中的同一情感类别并不总是对应单一表达模式。例如 angry 既可能表现为直接指责，也可能表现为克制的不满；sad 既可能来自明确负面事件，也可能由语境中的失落感隐含表达。单一情感锚点难以同时描述这些细粒度差异。因此，本文进一步引入多域子锚点，将每个情感类别划分为多个语义子空间，并通过动态聚合机制完成最终类别预测。"))

    body.append(p("3 方法", "Heading1"))
    body.append(p("本文提出一种基于多域子锚点原型增强的对话情感识别方法。整体框架由上下文编码、阶段一的原型增强表征学习、阶段二的域聚合分类以及联合训练目标四个部分组成。模型首先利用预训练语言模型获得目标话语的上下文感知表示，然后将话语表示映射到原型空间，并与情感子锚点进行匹配。通过监督对比学习和角度分离约束，模型能够同时增强类内聚合和类间分离；通过 domain-gated 聚合，模型能够根据样本语义动态融合不同子锚点响应。"))
    body.append(p("3.1 问题定义", "Heading2"))
    body.append(p("设一段对话表示为 D={u1,u2,...,uT}，其中 ut 表示第 t 个话语，st 表示对应说话人。给定情感标签集合 Y，对话情感识别任务的目标是为每个带有标注的目标话语 ut 预测其情感标签 yt∈Y。与句子级情感分类不同，本文将该任务建模为上下文条件下的话语级分类问题，即模型在预测 yt 时不仅考虑目标话语本身，还考虑其前若干轮历史话语及说话人信息。"))
    body.append(p("3.2 上下文编码", "Heading2"))
    body.append(p("对于目标话语 ut，本文首先选取其前 k 轮历史上下文，并将每个历史话语组织为“speaker says: utterance”的形式，以显式保留说话人身份信息。随后，在输入末尾加入面向目标话语的提示模板“For utterance: ut speaker feels <mask>”。模型将完整输入送入 Sup-SimCSE-RoBERTa-large，并取 <mask> 位置的隐藏状态作为目标话语的上下文感知表示 ht。该表示融合了历史语义、说话人身份和当前目标话语内容，为后续原型匹配提供基础。"))
    body.append(p("为了统一话语表示和情感原型所在空间，本文进一步引入映射网络 fmap(·)，将 ht 映射为原型空间中的表示 zt。映射网络由线性层、层归一化和 ReLU 激活组成，其作用是降低预训练表示中的任务无关噪声，并增强情感类别之间的可分性。"))
    body.append(p("3.3 多域子锚点原型建模", "Heading2"))
    body.append(p("传统情感锚点方法通常为每个情感类别设置一个类别中心，但同一情感在不同上下文中可能具有多种表达模式。为此，本文为每个情感类别 c 定义 K 个子锚点 Ac={ac,1,ac,2,...,ac,K}。在当前项目实现中，K 设置为 4，用于刻画情感激活程度、交互方式、外显表达风格和上下文诱发转折等语义域。各子锚点由情感标签文本编码初始化，并与话语表示共享同一原型空间。"))
    body.append(p("给定目标话语表示 zt，模型计算其与类别 c 下第 k 个子锚点 ac,k 的归一化余弦相似度，得到子锚点匹配分数 sc,k=(1+cos(zt,ac,k))/2。该分数表示当前话语与特定情感语义域的匹配程度。与单一锚点相比，多域子锚点能够更细致地刻画同一情感内部的表达差异，从而减少不同情感类别之间的重叠。"))
    body.append(p("3.4 原型增强监督对比学习", "Heading2"))
    body.append(p("阶段一的目标是构造具有良好类内聚合性和类间分离性的表示空间。本文在普通监督对比学习基础上，将情感子锚点也纳入对比集合。对于任一样本 zi，其正样本不仅包括同类话语表示，还包括同一情感类别下的子锚点；负样本则包括异类话语表示和异类子锚点。通过这种方式，模型能够显式学习样本表示与类别语义原型之间的对应关系。"))
    body.append(p("除对比损失外，本文同时使用交叉熵损失进行类别监督，并引入角度分离约束扩大不同类别原型之间的夹角。该约束鼓励类别中心在表示空间中保持较大角度间隔，从而缓解 happy 与 excited、sad 与 frustrated 等相似情感类别之间的混淆。训练过程中，模型还根据当前批次样本与子锚点的匹配关系对锚点进行动量更新，使子锚点能够逐步适应数据集中的真实表达分布。"))
    body.append(p("3.5 Domain-gated 聚合与分类", "Heading2"))
    body.append(p("阶段二的目标是在稳定的原型空间基础上学习最终分类边界。对于每个目标话语，模型分别通过多个 domain adapter 得到面向不同语义域的表示，并计算其与对应子锚点之间的匹配分布。随后，domain gate 根据原始上下文表示生成各语义域的动态权重，将多个子锚点响应融合为类别级得分。相比简单取最大值或平均值，domain-gated 聚合能够根据当前样本的语义特征自适应强调更相关的子锚点，降低无关语义域对预测结果的干扰。"))
    body.append(p("综合上述过程，本文模型的训练目标由交叉熵损失、原型增强监督对比损失和角度分离损失共同构成。整体方法既利用预训练语言模型获得上下文语义表示，又通过多域子锚点提供情感类别先验，使模型在复杂对话场景中获得更稳定的情感判别能力。"))

    body.append(p("4 实验", "Heading1"))
    body.append(p("4.1 数据集", "Heading2"))
    body.append(p("本文选取 IEMOCAP、MELD 和 EmoryNLP 三个对话情感识别领域常用公开数据集进行实验。三个数据集在对话来源、说话人数量、情感标签体系以及类别分布方面存在差异，能够从不同角度检验模型对复杂对话场景的适应能力。实验过程中，本文使用项目中已经预处理完成的训练集、验证集和测试集划分，并保持各数据集原有的对话或场景结构，以便在构造输入时保留历史上下文信息。"))
    body.append(p("IEMOCAP 是对话情感识别任务中使用较为广泛的双人对话数据集，包含即兴对话和脚本对话两类内容。本文采用常见设置，保留 neutral、excited、frustrated、sad、happy 和 angry 六类主要情感标签。MELD 是一个来源于电视剧 Friends 的多说话人对话数据集，本文仅使用其文本模态进行实验。EmoryNLP 同样基于 Friends 构建，但其场景划分方式和情感标签体系与 MELD 不完全相同，能够进一步检验模型在低频情感类别和复杂上下文中的泛化能力。"))
    body.append(p("表 4-1 数据集统计情况"))
    body.append(table([
        ["数据集", "训练集", "验证集", "测试集", "情感类别数"],
        ["IEMOCAP", "100 段对话 / 4778 条标注话语", "20 段对话 / 980 条标注话语", "31 段对话 / 1622 条标注话语", "6"],
        ["MELD", "1038 段对话 / 9989 条话语", "114 段对话 / 1109 条话语", "280 段对话 / 2610 条话语", "7"],
        ["EmoryNLP", "713 个场景 / 9934 条话语", "99 个场景 / 1344 条话语", "85 个场景 / 1328 条话语", "7"],
    ]))
    body.append(p("由表 4-1 可以看出，三个数据集在规模和结构上具有明显差异。IEMOCAP 的对话数量相对较少，但情感表达更集中；MELD 和 EmoryNLP 的话语数量较多，且对话场景更加贴近日常交流。由于 IEMOCAP 中部分原始话语没有情感标签，本文在训练和评测时仅对带有标签的话语计算监督目标与评价指标。"))
    body.append(p("4.2 实验设置", "Heading2"))
    body.append(p("本文模型以 Sup-SimCSE-RoBERTa-large 作为基础上下文编码器。最大输入长度设置为 256，历史窗口大小设置为 8，训练轮数设置为 8，dropout 设置为 0.1，最大梯度裁剪范数设置为 5.0。优化过程中，预训练语言模型部分学习率设置为 1e-5，任务层学习率设置为 4e-4；当启用第二阶段锚点自适应分类器时，其学习率设置为 1e-4。"))
    body.append(p("本文采用加权 F1 作为主要评价指标。对话情感识别数据集通常存在明显类别不平衡现象，若仅使用准确率进行评价，模型可能因偏向高频类别而获得较高结果，但不能充分反映其对低频类别的识别能力。加权 F1 根据各类别样本数量对类别 F1 进行加权平均，因此能够更稳定地反映模型在整体标签分布下的分类性能。"))
    body.append(p("表 4-2 主要实验参数设置"))
    body.append(table([
        ["参数", "设置"],
        ["基础编码器", "Sup-SimCSE-RoBERTa-large"],
        ["最大输入长度", "256"],
        ["历史窗口大小", "8"],
        ["训练轮数", "8"],
        ["预训练模型学习率", "1e-5"],
        ["任务层学习率", "4e-4"],
        ["第二阶段学习率", "1e-4"],
        ["Dropout", "0.1"],
        ["最大梯度裁剪范数", "5.0"],
        ["子锚点数量", "4"],
        ["评价指标", "Weighted F1"],
    ]))
    body.append(p("4.3 对比方法", "Heading2"))
    body.append(p("为全面评估本文方法的有效性，本文选取多种代表性对话情感识别模型作为对比方法，包括 DialogueRNN、DialogueGCN、DialogueCRN、RGAT、DAG-ERC、SACL、CoMPM、EmotionIC 和 EACL 等。其中，DialogueRNN 和 DialogueCRN 主要通过序列上下文建模捕捉对话中的情感动态变化；DialogueGCN、RGAT 和 DAG-ERC 通过图结构建模话语之间的依赖关系；SACL、EmotionIC 和 EACL 则侧重于通过表示学习、情感交互建模或情感语义先验提升话语表示质量。"))
    body.append(p("4.4 结果与分析", "Heading2"))
    body.append(p("表 4-3 三个数据集上的对比实验结果"))
    body.append(table([
        ["方法", "IEMOCAP", "MELD", "EmoryNLP"],
        ["DialogueRNN", "66.60", "64.09", "-"],
        ["DialogueGCN", "64.91", "63.02", "38.10"],
        ["DialogueCRN", "66.33", "63.42", "38.91"],
        ["RGAT", "66.36", "62.80", "37.89"],
        ["DAG-ERC", "68.03", "63.65", "39.02"],
        ["SACL", "69.22", "66.45", "39.65"],
        ["CoMPM", "69.46", "66.52", "38.93"],
        ["EmotionIC", "69.61", "66.40", "40.01"],
        ["EACL", "70.41", "67.12", "40.24"],
        ["本文方法", "71.05", "67.19", "39.98"],
    ]))
    body.append(p("由表 4-3 可以看出，本文方法在 IEMOCAP 数据集上取得 71.05 的加权 F1，优于所有对比方法。与表现最强的基线模型 EACL 相比，本文方法提升 0.64 个百分点；与 EmotionIC 相比，提升 1.44 个百分点；与 DAG-ERC 相比，提升 3.02 个百分点。该结果表明，仅依赖对话结构或上下文传播机制仍难以充分解决相近情感类别混淆问题，而将多域情感语义先验引入表示学习过程能够进一步提升模型的判别能力。"))
    body.append(p("在 MELD 数据集上，本文方法取得 67.19 的加权 F1，同样达到最优结果。与 EACL 相比，本文方法提升 0.07 个百分点；与 CoMPM 相比，提升 0.67 个百分点；与 SACL 相比，提升 0.74 个百分点。MELD 中的话语通常较短，情感线索不如 IEMOCAP 明显，同时多说话人交互使上下文关系更加复杂。本文方法通过提示式上下文编码保留历史对话信息，并利用情感子锚点提供类别语义参照，因此在日常多方对话场景下仍能保持稳定优势。"))
    body.append(p("在 EmoryNLP 数据集上，本文方法取得 39.98 的加权 F1，略低于 EACL 的 40.24 和 EmotionIC 的 40.01，但仍优于 DialogueGCN、DialogueCRN、RGAT、DAG-ERC、SACL 和 CoMPM 等多数方法。本文方法在该数据集上未取得最优结果，可能与其标签体系和类别分布有关。EmoryNLP 中 peaceful、powerful、neutral 等类别语义边界较细，且部分类别样本较少，使得多域子锚点在训练过程中可能受到样本不足影响。后续可考虑根据类别频次设置自适应数量的子锚点，或引入更细粒度的上下文筛选机制。"))
    body.append(p("总体来看，本文方法在 IEMOCAP 和 MELD 上均取得最优结果，在 EmoryNLP 上也达到接近最优的性能。该现象说明，多域子锚点原型增强方法对不同类型的对话数据具有较好的适应能力。与图结构方法相比，本文方法并未显式构建复杂对话图，而是通过预训练语言模型、提示式上下文编码和原型增强学习获得更具判别性的情感表示，说明表示空间结构优化对于 ERC 任务同样重要。"))

    body.append(p("5 结论", "Heading1"))
    body.append(p("本文针对对话情感识别任务中上下文依赖强、相近情感类别易混淆以及同类情感表达多样的问题，提出一种基于多域子锚点原型增强的对话情感识别方法。该方法利用 Sup-SimCSE-RoBERTa-large 获得上下文感知的话语表示，并通过多域子锚点描述同一情感类别内部的不同语义表达模式。结合原型增强监督对比学习、角度分离约束、锚点动量更新和 domain-gated 聚合机制，模型能够在表示空间中形成更清晰的类别结构。"))
    body.append(p("在 IEMOCAP、MELD 和 EmoryNLP 三个公开数据集上的实验结果表明，本文方法在 IEMOCAP 和 MELD 上均取得最优性能，在 EmoryNLP 上也保持较强竞争力。实验验证了多域子锚点机制在提升相似情感区分能力和增强情感表示结构方面的有效性。未来工作可以从两个方向继续改进：一是针对类别不平衡问题设计自适应子锚点分配策略；二是进一步结合说话人关系和情感转移路径，使模型能够更充分地利用对话结构信息。"))

    body.append(p("参考文献", "Heading1"))
    refs = [
        "[1] Busso C, Bulut M, Lee C C, et al. IEMOCAP: Interactive emotional dyadic motion capture database. Language Resources and Evaluation, 2008.",
        "[2] Poria S, Hazarika D, Majumder N, et al. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL, 2019.",
        "[3] Zahiri S M, Choi J D. Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks. AAAI Workshop, 2018.",
        "[4] Majumder N, Poria S, Hazarika D, et al. DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. AAAI, 2019.",
        "[5] Ghosal D, Majumder N, Poria S, et al. DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. EMNLP-IJCNLP, 2019.",
        "[6] Yu F, Guo J, Wu Z, Dai X. Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation. arXiv preprint arXiv:2403.20289, 2024.",
        "[7] Liu Y, Ott M, Goyal N, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692, 2019.",
        "[8] Gao T, Yao X, Chen D. SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP, 2021.",
    ]
    for ref in refs:
        body.append(p(ref))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="851" w:footer="992" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )
    return document


styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="24"/></w:rPr><w:pPr><w:spacing w:line="360" w:lineRule="auto"/><w:firstLineChars w:val="200"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="28"/></w:rPr></w:style>
</w:styles>'''


content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/document.xml", make_doc())
    print(OUT)


if __name__ == "__main__":
    main()
