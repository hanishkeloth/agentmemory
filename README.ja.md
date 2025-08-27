# AgentMemory 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/hanishkeloth/agentmemory)

**AgentMemory**は、AIエージェント向けの高度なメモリ管理フレームワークで、永続的で階層的、セマンティックなメモリ機能を提供します。現在のエージェンティックAI開発における重要なギャップである、エージェントセッション全体での堅牢なメモリとコンテキスト管理を解決します。

## 🚀 主要機能

### 🎯 複数のメモリタイプ
- **短期記憶**: 即座のコンテキストのためのTTL付きFIFOバッファ
- **長期記憶**: 重要な情報の重要度ベースの保持
- **エピソード記憶**: イベントシーケンスの時間的構成
- **セマンティック記憶**: 関係性を持つ概念ベースの知識
- **手続き記憶**: 実行追跡付きのスキルとハウツー知識

### 🔍 高度な検索
- **セマンティック検索**: 埋め込みを使用したベクトル類似性検索
- **フィルタークエリ**: タイプ、タグ、重要度、またはカスタムメタデータによる取得
- **関係ナビゲーション**: メモリの関連付けと関係の追跡
- **時間減衰スコアリング**: 減衰を含む自動関連性計算

### 💾 永続性とストレージ
- **複数のバックエンド**: インメモリ、ベクトルストア（NumPy/FAISS）
- **保存/読み込み**: メモリ永続性のためのJSONシリアル化
- **バッチ操作**: 効率的な一括追加/削除操作

### 🔄 メモリ管理
- **自動統合**: 短期から長期への自動昇格
- **重要度スコアリング**: 設定可能な重要度しきい値
- **メモリ関連付け**: メモリ間の関係作成
- **統計追跡**: メモリ使用量とパターンの監視

## 📦 インストール

```bash
pip install agentmemory
```

開発用:
```bash
git clone https://github.com/hanishkeloth/agentmemory.git
cd agentmemory
pip install -e ".[dev]"
```

## 🎓 クイックスタート

```python
from agentmemory import MemoryManager

# メモリマネージャーの初期化
memory = MemoryManager()

# メモリの追加
memory.add(
    "ユーザーはデータサイエンスにPythonを好む",
    memory_type="long_term",
    importance=0.9,
    tags=["preference", "python"]
)

# 関連するメモリの取得
memories = memory.retrieve(
    query="どのプログラミング言語を使うべきか？",
    limit=5
)

# 関連付けの作成
memory.create_association(memory1_id, memory2_id, "related_to")

# 永続化のための保存
memory.save("agent_memories.json")
```

## 💡 使用例

### 🤖 会話型AIエージェント
```python
class ChatAgent:
    def __init__(self):
        self.memory = MemoryManager()
    
    def process(self, user_input):
        # 会話の保存
        self.memory.add(
            user_input,
            memory_type="short_term",
            session_id=self.session_id
        )
        
        # コンテキストの取得
        context = self.memory.retrieve(user_input, limit=5)
        
        # コンテキストを使用した応答生成
        response = self.generate_response(user_input, context)
        return response
```

### 📚 知識管理
```python
# 概念と共に事実を保存
memory.add(
    "Pythonは1991年にGuido van Rossumによって作成された",
    memory_type="semantic",
    concepts=["python", "歴史", "プログラミング"],
    importance=0.8
)

# 概念によるクエリ
python_facts = memory.retrieve(concepts=["python"], limit=10)
```

### 🔧 スキル学習
```python
# 手順の保存
memory.add(
    {"procedure": "AWSへのデプロイ", "steps": [...]},
    memory_type="procedural",
    procedure_name="aws_deployment",
    skill_level=0.7
)

# 実行成功の追跡
memory.update_execution(procedure_id, success=True)
```

## 🏗️ アーキテクチャ

```
AgentMemory/
├── core/
│   ├── memory_manager.py    # 中央オーケストレーター
│   ├── memory_types.py      # メモリタイプ実装
│   └── memory_entry.py      # メモリデータ構造
├── stores/
│   ├── base.py             # 抽象ストアインターフェース
│   └── vector.py           # ベクトル類似性ストア
├── retrievers/
│   ├── base.py             # 抽象リトリーバーインターフェース
│   └── semantic.py         # セマンティック検索実装
└── utils/
    └── embeddings.py       # 埋め込みユーティリティ
```

## 🔬 高度な機能

### メモリ統合
```python
# 短期から長期への自動統合
manager = MemoryManager(consolidation_threshold=10)

# 手動統合
stats = manager.consolidate_memories()
print(f"長期記憶への昇格: {stats['promoted_to_long_term']}")
```

### カスタムメタデータ
```python
memory.add(
    "重要なイベント",
    memory_type="episodic",
    custom_metadata={
        "location": "サンフランシスコ",
        "participants": ["アリス", "ボブ"],
        "outcome": "成功"
    }
)
```

### 埋め込みを使用したベクトル検索
```python
from agentmemory.utils.embeddings import EmbeddingManager

embedder = EmbeddingManager()
embedding = embedder.encode_single("機械学習の概念")

memory.add(
    "ニューラルネットワークは生物学的ニューロンに触発されている",
    memory_type="semantic",
    embedding=embedding
)
```

## 🤝 人気のフレームワークとの統合

### LangChain
```python
from langchain.memory import ConversationBufferMemory
from agentmemory import MemoryManager

class AgentMemoryWrapper(ConversationBufferMemory):
    def __init__(self):
        super().__init__()
        self.agent_memory = MemoryManager()
    
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        self.agent_memory.add(
            {"input": inputs, "output": outputs},
            memory_type="episodic"
        )
```

### AutoGen
```python
from autogen import AssistantAgent
from agentmemory import MemoryManager

class MemoryAgent(AssistantAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.memory = MemoryManager()
    
    def receive(self, message, sender):
        # メッセージをメモリに保存
        self.memory.add(
            message,
            memory_type="short_term",
            agent_id=sender.name
        )
        return super().receive(message, sender)
```

## 📊 パフォーマンスの考慮事項

- **埋め込み次元**: デフォルト384（all-MiniLM-L6-v2）、パフォーマンスに応じて調整可能
- **容量制限**: リソース使用管理のためメモリタイプごとに設定可能
- **バッチ操作**: 大量操作にはバッチメソッドを使用
- **ベクトルストアの選択**: 小規模にはNumPy、本番環境にはFAISS

## 🛠️ 開発

テストの実行:
```bash
pytest tests/
```

コードフォーマット:
```bash
black agentmemory/
ruff check agentmemory/
```

型チェック:
```bash
mypy agentmemory/
```

## 📈 ベンチマーク

| 操作 | 1Kメモリ | 10Kメモリ | 100Kメモリ |
|------|----------|-----------|------------|
| 追加 | 0.8ms    | 0.9ms     | 1.1ms      |
| 取得 | 2.3ms    | 8.7ms     | 45ms       |
| 検索 | 3.1ms    | 12ms      | 89ms       |

*MacBook Pro M1、16GB RAMでテスト*

## 🗺️ ロードマップ

- [ ] 分散メモリストア（Redis、PostgreSQL）
- [ ] グラフベースのメモリ関係
- [ ] メモリ圧縮と要約
- [ ] マルチエージェントメモリ共有
- [ ] メモリバージョニングとロールバック
- [ ] 高度な統合戦略
- [ ] メモリアテンションメカニズム
- [ ] より多くのフレームワークとの統合

## 🤝 貢献

貢献を歓迎します！プルリクエストをお気軽に提出してください。

1. リポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを開く

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 👨‍💻 作者

**Hanish Keloth**
- GitHub: [@hanishkeloth](https://github.com/hanishkeloth)
- Email: hanishkeloth216@gmail.com

## 🙏 謝辞

- 認知アーキテクチャと人間の記憶システムに触発されて
- 現在のエージェンティックAIフレームワークで特定されたギャップに対処するために構築
- 継続的なイノベーションのためのオープンソースAIコミュニティに感謝

## 📚 引用

研究やプロジェクトでAgentMemoryを使用する場合は、次のように引用してください：

```bibtex
@software{agentmemory2025,
  author = {Keloth, Hanish},
  title = {AgentMemory: AIエージェントのための高度なメモリ管理},
  year = {2025},
  url = {https://github.com/hanishkeloth/agentmemory}
}
```

## 🔗 リンク

- [ドキュメント](https://github.com/hanishkeloth/agentmemory/wiki)
- [イシュー](https://github.com/hanishkeloth/agentmemory/issues)
- [ディスカッション](https://github.com/hanishkeloth/agentmemory/discussions)
- [PyPIパッケージ](https://pypi.org/project/agentmemory/)

---

**注**: このプロジェクトはベータ版です。APIは将来のバージョンで変更される可能性があります。問題や提案をご報告ください！