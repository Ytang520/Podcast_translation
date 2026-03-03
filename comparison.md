### v2

闭源模型选择分析：

- 实验设置：~20 min 音频，计时和记成本不同模型效果
- 效果上：gpt-audio-mini WER 几乎为 0，同时段落划分更合适，方便修改
- 成本上：相差无几
    - gpt-audio-mini：
        - 使用 openrouter，0.12元，1min 转译
        - 人工校验 15 min
    - canary:
        - autodl 上开机 + 起服务 + 转译，单次时间 (> 4 min)，根据选取的机器租赁成本，合计成本 ~0.10 元
        - 质量过差，人工校验通常需 ~1h

结论：选择闭源 openai gpt_audio_mini 进行 transcript，并设置 temperature=0.8 生成多次文本，进行交叉验证

---

### v1

### 开源模型选择分析：

- 实验设置：
    - Dataset: ep002 of < History of Philosophy without any Gaps >
- 模型选择：
    - ~~granite-speech-3.3-8b: https://modelscope.cn/models/ibm-granite/granite-speech-3.3-8b (转译时完全复读机，无法正常运行，故放弃)~~
    - whisper
    - canary-1b-v2: https://huggingface.co/nvidia/canary-1b-v2
- 其他设置：
    - 保持初始设定：如 temperature 和 sliding window 等都无法轻易改变，此处保持初始设定
- 结果：(以 ep002 的结果作为标准，进行比较；都无法调节 temperature)
    
    
    |  | Whisper | Canary |
    | --- | --- | --- |
    | WER_mean | 0.1669 | 0.0999 |
    | WER_std | 0.0284 | 0.0000 |
    - 结果分析：
        - ~~模型本身优势是一方面，同时怀疑是否在数据上过拟合，因为对于其他音频并未明显好于 Whisper（whisper: 2022, canary: 2025)~~ 选用的音频是网上公开音频，发布时间为 Dec.23 2010，应只是 canary 架构 + 训练方式够好…
        - case study: 都有复读机现象，其中 Whisper 更严重 (Refer to the evaluation_test folder for the results)
- 结论：受限于资源，不考虑微调；后续使用 canary 来生成文本，whisper 作为辅助参考，生成后人工校验文本