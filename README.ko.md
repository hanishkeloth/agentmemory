# AgentMemory 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/hanishkeloth/agentmemory)

**AgentMemory**는 AI 에이전트를 위한 고급 메모리 관리 프레임워크로, 지속적이고 계층적이며 의미론적인 메모리 기능을 제공합니다. 현재 에이전틱 AI 개발의 중요한 격차인 에이전트 세션 전반의 강력한 메모리 및 컨텍스트 관리를 해결합니다.

## 🚀 주요 기능

### 🎯 다양한 메모리 타입
- **단기 메모리**: 즉각적인 컨텍스트를 위한 TTL이 있는 FIFO 버퍼
- **장기 메모리**: 중요한 정보를 위한 중요도 기반 보존
- **에피소드 메모리**: 이벤트 시퀀스의 시간적 구성
- **의미 메모리**: 관계가 있는 개념 기반 지식
- **절차 메모리**: 실행 추적이 있는 기술 및 방법 지식

### 🔍 고급 검색
- **의미론적 검색**: 임베딩을 사용한 벡터 유사성 검색
- **필터링된 쿼리**: 타입, 태그, 중요도 또는 사용자 정의 메타데이터로 검색
- **관계 탐색**: 메모리 연관성 및 관계 추적
- **시간 감쇄 점수**: 감쇄를 포함한 자동 관련성 계산

### 💾 지속성 및 저장
- **다중 백엔드**: 인메모리, 벡터 스토어 (NumPy/FAISS)
- **저장/로드**: 메모리 지속성을 위한 JSON 직렬화
- **배치 작업**: 효율적인 대량 추가/삭제 작업

### 🔄 메모리 관리
- **자동 통합**: 단기에서 장기로 자동 승격
- **중요도 점수**: 구성 가능한 중요도 임계값
- **메모리 연관**: 메모리 간 관계 생성
- **통계 추적**: 메모리 사용 및 패턴 모니터링

## 📦 설치

```bash
pip install agentmemory
```

개발용:
```bash
git clone https://github.com/hanishkeloth/agentmemory.git
cd agentmemory
pip install -e ".[dev]"
```

## 🎓 빠른 시작

```python
from agentmemory import MemoryManager

# 메모리 매니저 초기화
memory = MemoryManager()

# 메모리 추가
memory.add(
    "사용자는 데이터 사이언스에 Python을 선호함",
    memory_type="long_term",
    importance=0.9,
    tags=["preference", "python"]
)

# 관련 메모리 검색
memories = memory.retrieve(
    query="어떤 프로그래밍 언어를 사용할까?",
    limit=5
)

# 연관 생성
memory.create_association(memory1_id, memory2_id, "related_to")

# 지속성을 위해 저장
memory.save("agent_memories.json")
```

## 💡 사용 사례

### 🤖 대화형 AI 에이전트
```python
class ChatAgent:
    def __init__(self):
        self.memory = MemoryManager()
    
    def process(self, user_input):
        # 대화 저장
        self.memory.add(
            user_input,
            memory_type="short_term",
            session_id=self.session_id
        )
        
        # 컨텍스트 검색
        context = self.memory.retrieve(user_input, limit=5)
        
        # 컨텍스트를 활용한 응답 생성
        response = self.generate_response(user_input, context)
        return response
```

### 📚 지식 관리
```python
# 개념과 함께 사실 저장
memory.add(
    "Python은 1991년 Guido van Rossum에 의해 만들어졌습니다",
    memory_type="semantic",
    concepts=["python", "history", "programming"],
    importance=0.8
)

# 개념으로 쿼리
python_facts = memory.retrieve(concepts=["python"], limit=10)
```

### 🔧 기술 학습
```python
# 절차 저장
memory.add(
    {"procedure": "AWS에 배포", "steps": [...]},
    memory_type="procedural",
    procedure_name="aws_deployment",
    skill_level=0.7
)

# 실행 성공 추적
memory.update_execution(procedure_id, success=True)
```

## 🏗️ 아키텍처

```
AgentMemory/
├── core/
│   ├── memory_manager.py    # 중앙 오케스트레이터
│   ├── memory_types.py      # 메모리 타입 구현
│   └── memory_entry.py      # 메모리 데이터 구조
├── stores/
│   ├── base.py             # 추상 스토어 인터페이스
│   └── vector.py           # 벡터 유사성 스토어
├── retrievers/
│   ├── base.py             # 추상 리트리버 인터페이스
│   └── semantic.py         # 의미론적 검색 구현
└── utils/
    └── embeddings.py       # 임베딩 유틸리티
```

## 🔬 고급 기능

### 메모리 통합
```python
# 단기에서 장기로 자동 통합
manager = MemoryManager(consolidation_threshold=10)

# 수동 통합
stats = manager.consolidate_memories()
print(f"장기로 승격됨: {stats['promoted_to_long_term']}")
```

### 사용자 정의 메타데이터
```python
memory.add(
    "중요한 이벤트",
    memory_type="episodic",
    custom_metadata={
        "location": "샌프란시스코",
        "participants": ["앨리스", "밥"],
        "outcome": "성공"
    }
)
```

### 임베딩을 사용한 벡터 검색
```python
from agentmemory.utils.embeddings import EmbeddingManager

embedder = EmbeddingManager()
embedding = embedder.encode_single("머신러닝 개념")

memory.add(
    "신경망은 생물학적 뉴런에서 영감을 받았습니다",
    memory_type="semantic",
    embedding=embedding
)
```

## 🤝 인기 프레임워크와의 통합

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
        # 메모리에 메시지 저장
        self.memory.add(
            message,
            memory_type="short_term",
            agent_id=sender.name
        )
        return super().receive(message, sender)
```

## 📊 성능 고려사항

- **임베딩 차원**: 기본 384 (all-MiniLM-L6-v2), 성능에 따라 조정 가능
- **용량 제한**: 리소스 사용 관리를 위해 메모리 타입별로 구성 가능
- **배치 작업**: 대량 작업에는 배치 메서드 사용
- **벡터 스토어 선택**: 소규모는 NumPy, 프로덕션은 FAISS

## 🛠️ 개발

테스트 실행:
```bash
pytest tests/
```

코드 포맷:
```bash
black agentmemory/
ruff check agentmemory/
```

타입 체크:
```bash
mypy agentmemory/
```

## 📈 벤치마크

| 작업 | 1K 메모리 | 10K 메모리 | 100K 메모리 |
|------|-----------|------------|-------------|
| 추가 | 0.8ms     | 0.9ms      | 1.1ms       |
| 검색 | 2.3ms     | 8.7ms      | 45ms        |
| 탐색 | 3.1ms     | 12ms       | 89ms        |

*MacBook Pro M1, 16GB RAM에서 테스트*

## 🗺️ 로드맵

- [ ] 분산 메모리 스토어 (Redis, PostgreSQL)
- [ ] 그래프 기반 메모리 관계
- [ ] 메모리 압축 및 요약
- [ ] 다중 에이전트 메모리 공유
- [ ] 메모리 버전 관리 및 롤백
- [ ] 고급 통합 전략
- [ ] 메모리 어텐션 메커니즘
- [ ] 더 많은 프레임워크와 통합

## 🤝 기여하기

기여를 환영합니다! Pull Request를 제출해 주세요.

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 열기

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👨‍💻 저자

**Hanish Keloth**
- GitHub: [@hanishkeloth](https://github.com/hanishkeloth)
- Email: hanishkeloth1256@gmail.com

## 🙏 감사의 말

- 인지 아키텍처와 인간 메모리 시스템에서 영감을 받음
- 현재 에이전틱 AI 프레임워크에서 확인된 격차를 해결하기 위해 구축
- 지속적인 혁신을 위한 오픈소스 AI 커뮤니티에 감사

## 📚 인용

AgentMemory를 연구나 프로젝트에서 사용하신다면, 다음과 같이 인용해 주세요:

```bibtex
@software{agentmemory2025,
  author = {Keloth, Hanish},
  title = {AgentMemory: AI 에이전트를 위한 고급 메모리 관리},
  year = {2025},
  url = {https://github.com/hanishkeloth/agentmemory}
}
```

## 🔗 링크

- [문서](https://github.com/hanishkeloth/agentmemory/wiki)
- [이슈](https://github.com/hanishkeloth/agentmemory/issues)
- [토론](https://github.com/hanishkeloth/agentmemory/discussions)
- [PyPI 패키지](https://pypi.org/project/agentmemory/)

---

**참고**: 이 프로젝트는 베타 버전입니다. API는 향후 버전에서 변경될 수 있습니다. 문제나 제안사항을 보고해 주세요!