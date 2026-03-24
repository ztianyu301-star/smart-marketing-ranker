# 智能营销多任务排序系统

一个面向“智能营销 / 搜索推荐 / 广告排序”岗位的完整项目示例，重点展示以下能力：

- 用 Python 独立完成数据建模、训练、评估与部署
- 同时覆盖传统机器学习基线与 PyTorch 深度学习模型
- 体现智能营销场景中的 CTR / CVR 多任务学习与业务排序
- 包含离线评估、可解释特征、推理 API 和简历可用表达

## 项目亮点

1. **业务贴合度高**：模拟广告与推荐融合场景，对曝光、点击、转化做联合建模。
2. **算法覆盖全面**：提供 Logistic Regression 基线，以及 PyTorch 多任务 Wide & Deep 排序模型。
3. **工程完整度高**：提供可复现实验数据、训练管线、指标评估、模型保存、在线排序接口。
4. **简历友好**：项目输出中直接生成示例排名结果与可写进简历的项目表述。

## 系统设计

- **数据层**：自动生成带用户、商品、查询、上下文和营销特征的排序样本
- **建模层**：构建 CTR 基线模型、CVR 条件转化模型、多任务深度排序模型
- **评估层**：输出 AUC、LogLoss、NDCG@10、平均 CTR/CVR 等指标
- **服务层**：通过 FastAPI 暴露 `/rank` 接口，支持在线候选集重排

## 目录结构

```text
ant_intelligent_ranking/
├─ main.py
├─ README.md
├─ requirements.txt
├─ resume_bullets.md
├─ src/
│  └─ smart_ranker/
│     ├─ baselines.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ data.py
│     ├─ evaluation.py
│     ├─ features.py
│     ├─ model.py
│     ├─ pipeline.py
│     ├─ ranking.py
│     └─ serving.py
└─ tests/
   └─ test_smoke.py
```

## 快速开始

```powershell
cd ant_intelligent_ranking
python -m pip install -r requirements.txt
python main.py pipeline --output-dir artifacts --requests 1200 --candidates-per-request 10 --epochs 6
python -m pytest tests/test_smoke.py -q
```

运行完成后会在 `artifacts/` 中生成：

- `metrics.json`：基线模型与深度模型对比结果
- `demo_ranking.json`：测试集中的真实请求重排示例
- `sample_dataset.csv`：部分训练样本
- `deep_model.pt`：PyTorch 模型与特征处理器
- `baseline.pkl`：Logistic Regression 基线模型

## 启动排序 API

```powershell
cd ant_intelligent_ranking
python main.py serve --artifact-dir artifacts --host 127.0.0.1 --port 8000
```

接口说明：

- `GET /health`：服务健康检查
- `POST /rank`：输入候选样本列表，输出重排结果和 CTR / CVR / business_score

## 和岗位要求的对应关系

- **智能营销 / 搜索推荐 / 广告项目经验**：项目核心就是营销排序与转化预估
- **传统机器学习 + 深度学习**：同时实现 LR 基线与 PyTorch 多任务模型
- **至少掌握一套主流训练框架**：使用 PyTorch 完成特征编码、训练和推理
- **独立编码建模能力**：从数据构造、特征工程、训练到服务化全部可运行
- **数据敏感、量化建模能力**：包含多维度用户/商品/上下文特征与离线评估

## 建议怎么写进简历

`resume_bullets.md` 中已经给出了一版可直接改写的项目描述，建议你根据自己真实参与度和掌握程度调整措辞，不要写超出项目实际内容的经历。
