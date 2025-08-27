# AgentMemory 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/hanishkeloth/agentmemory)

<div dir="rtl">

**AgentMemory** هو إطار عمل متقدم لإدارة الذاكرة لوكلاء الذكاء الاصطناعي، يوفر قدرات ذاكرة دائمة وهرمية ودلالية. يعالج الفجوة الحرجة في تطوير الذكاء الاصطناعي الوكيل الحالي: إدارة قوية للذاكرة والسياق عبر جلسات الوكيل.

## 🚀 الميزات الرئيسية

### 🎯 أنواع ذاكرة متعددة
- **الذاكرة قصيرة المدى**: مخزن FIFO مؤقت مع TTL للسياق الفوري
- **الذاكرة طويلة المدى**: الاحتفاظ بالمعلومات القيمة بناءً على الأهمية
- **الذاكرة العرضية**: التنظيم الزمني لتسلسل الأحداث
- **الذاكرة الدلالية**: المعرفة القائمة على المفاهيم مع العلاقات
- **الذاكرة الإجرائية**: المهارات والمعرفة التطبيقية مع تتبع التنفيذ

### 🔍 استرجاع متقدم
- **البحث الدلالي**: البحث عن التشابه المتجه باستخدام التضمينات
- **استعلامات مفلترة**: الاسترجاع حسب النوع، العلامات، الأهمية، أو البيانات الوصفية المخصصة
- **التنقل في العلاقات**: تتبع ارتباطات وعلاقات الذاكرة
- **تسجيل التحلل الزمني**: حساب الصلة التلقائي مع التحلل

### 💾 الثبات والتخزين
- **خلفيات متعددة**: في الذاكرة، مخازن المتجهات (NumPy/FAISS)
- **حفظ/تحميل**: تسلسل JSON لثبات الذاكرة
- **عمليات الدفعات**: عمليات إضافة/حذف جماعية فعالة

### 🔄 إدارة الذاكرة
- **التوحيد التلقائي**: الترقية التلقائية من قصيرة المدى إلى طويلة المدى
- **تسجيل الأهمية**: عتبات أهمية قابلة للتكوين
- **ارتباطات الذاكرة**: إنشاء علاقات بين الذكريات
- **تتبع الإحصائيات**: مراقبة استخدام الذاكرة والأنماط

## 📦 التثبيت

```bash
pip install agentmemory
```

للتطوير:
```bash
git clone https://github.com/hanishkeloth/agentmemory.git
cd agentmemory
pip install -e ".[dev]"
```

## 🎓 البدء السريع

```python
from agentmemory import MemoryManager

# تهيئة مدير الذاكرة
memory = MemoryManager()

# إضافة ذكريات
memory.add(
    "المستخدم يفضل Python لعلوم البيانات",
    memory_type="long_term",
    importance=0.9,
    tags=["preference", "python"]
)

# استرجاع الذكريات ذات الصلة
memories = memory.retrieve(
    query="ما لغة البرمجة التي يجب استخدامها؟",
    limit=5
)

# إنشاء ارتباطات
memory.create_association(memory1_id, memory2_id, "related_to")

# حفظ للثبات
memory.save("agent_memories.json")
```

## 💡 حالات الاستخدام

### 🤖 وكلاء الذكاء الاصطناعي للمحادثة
```python
class ChatAgent:
    def __init__(self):
        self.memory = MemoryManager()
    
    def process(self, user_input):
        # تخزين المحادثة
        self.memory.add(
            user_input,
            memory_type="short_term",
            session_id=self.session_id
        )
        
        # استرجاع السياق
        context = self.memory.retrieve(user_input, limit=5)
        
        # توليد رد مع السياق
        response = self.generate_response(user_input, context)
        return response
```

### 📚 إدارة المعرفة
```python
# تخزين الحقائق مع المفاهيم
memory.add(
    "تم إنشاء Python في عام 1991 بواسطة Guido van Rossum",
    memory_type="semantic",
    concepts=["python", "تاريخ", "برمجة"],
    importance=0.8
)

# الاستعلام حسب المفاهيم
python_facts = memory.retrieve(concepts=["python"], limit=10)
```

### 🔧 تعلم المهارات
```python
# تخزين الإجراءات
memory.add(
    {"procedure": "النشر على AWS", "steps": [...]},
    memory_type="procedural",
    procedure_name="aws_deployment",
    skill_level=0.7
)

# تتبع نجاح التنفيذ
memory.update_execution(procedure_id, success=True)
```

## 🏗️ البنية

```
AgentMemory/
├── core/
│   ├── memory_manager.py    # المنسق المركزي
│   ├── memory_types.py      # تنفيذ أنواع الذاكرة
│   └── memory_entry.py      # هياكل بيانات الذاكرة
├── stores/
│   ├── base.py             # واجهة التخزين المجردة
│   └── vector.py           # مخازن تشابه المتجهات
├── retrievers/
│   ├── base.py             # واجهة المسترجع المجردة
│   └── semantic.py         # تنفيذ البحث الدلالي
└── utils/
    └── embeddings.py       # أدوات التضمين
```

## 🔬 الميزات المتقدمة

### توحيد الذاكرة
```python
# التوحيد التلقائي من قصيرة المدى إلى طويلة المدى
manager = MemoryManager(consolidation_threshold=10)

# التوحيد اليدوي
stats = manager.consolidate_memories()
print(f"تمت الترقية إلى طويلة المدى: {stats['promoted_to_long_term']}")
```

### البيانات الوصفية المخصصة
```python
memory.add(
    "حدث مهم",
    memory_type="episodic",
    custom_metadata={
        "location": "سان فرانسيسكو",
        "participants": ["أليس", "بوب"],
        "outcome": "ناجح"
    }
)
```

### البحث المتجه مع التضمينات
```python
from agentmemory.utils.embeddings import EmbeddingManager

embedder = EmbeddingManager()
embedding = embedder.encode_single("مفهوم التعلم الآلي")

memory.add(
    "الشبكات العصبية مستوحاة من الخلايا العصبية البيولوجية",
    memory_type="semantic",
    embedding=embedding
)
```

## 🤝 التكامل مع أطر العمل الشائعة

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
        # تخزين الرسالة في الذاكرة
        self.memory.add(
            message,
            memory_type="short_term",
            agent_id=sender.name
        )
        return super().receive(message, sender)
```

## 📊 اعتبارات الأداء

- **بُعد التضمين**: الافتراضي 384 (all-MiniLM-L6-v2)، قابل للتعديل للأداء
- **حدود السعة**: قابلة للتكوين لكل نوع ذاكرة لإدارة استخدام الموارد
- **عمليات الدفعات**: استخدم طرق الدفعات للعمليات الجماعية
- **اختيار مخزن المتجهات**: NumPy للنطاق الصغير، FAISS للإنتاج

## 🛠️ التطوير

تشغيل الاختبارات:
```bash
pytest tests/
```

تنسيق الكود:
```bash
black agentmemory/
ruff check agentmemory/
```

فحص النوع:
```bash
mypy agentmemory/
```

## 📈 المعايير

| العملية | 1K ذاكرة | 10K ذاكرة | 100K ذاكرة |
|---------|----------|-----------|------------|
| إضافة   | 0.8ms    | 0.9ms     | 1.1ms      |
| استرجاع | 2.3ms    | 8.7ms     | 45ms       |
| بحث     | 3.1ms    | 12ms      | 89ms       |

*تم الاختبار على MacBook Pro M1، 16GB RAM*

## 🗺️ خارطة الطريق

- [ ] مخازن الذاكرة الموزعة (Redis، PostgreSQL)
- [ ] علاقات الذاكرة القائمة على الرسم البياني
- [ ] ضغط وتلخيص الذاكرة
- [ ] مشاركة الذاكرة متعددة الوكلاء
- [ ] إصدار الذاكرة والتراجع
- [ ] استراتيجيات التوحيد المتقدمة
- [ ] آليات انتباه الذاكرة
- [ ] التكامل مع المزيد من أطر العمل

## 🤝 المساهمة

نرحب بالمساهمات! لا تتردد في تقديم طلب سحب.

1. انسخ المستودع
2. أنشئ فرع الميزة (`git checkout -b feature/amazing-feature`)
3. قم بالتزام تغييراتك (`git commit -m 'Add amazing feature'`)
4. ادفع إلى الفرع (`git push origin feature/amazing-feature`)
5. افتح طلب سحب

## 📄 الترخيص

هذا المشروع مرخص بموجب ترخيص MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

## 👨‍💻 المؤلف

**Hanish Keloth**
- GitHub: [@hanishkeloth](https://github.com/hanishkeloth)
- Email: hanishkeloth@gmail.com

## 🙏 شكر وتقدير

- مستوحى من البنى المعرفية وأنظمة الذاكرة البشرية
- بُني لمعالجة الفجوات المحددة في أطر الذكاء الاصطناعي الوكيل الحالية
- شكراً لمجتمع الذكاء الاصطناعي مفتوح المصدر للابتكار المستمر

## 📚 الاستشهاد

إذا استخدمت AgentMemory في بحثك أو مشاريعك، يرجى الاستشهاد:

```bibtex
@software{agentmemory2025,
  author = {Keloth, Hanish},
  title = {AgentMemory: إدارة ذاكرة متقدمة لوكلاء الذكاء الاصطناعي},
  year = {2025},
  url = {https://github.com/hanishkeloth/agentmemory}
}
```

## 🔗 الروابط

- [التوثيق](https://github.com/hanishkeloth/agentmemory/wiki)
- [المشكلات](https://github.com/hanishkeloth/agentmemory/issues)
- [المناقشات](https://github.com/hanishkeloth/agentmemory/discussions)
- [حزمة PyPI](https://pypi.org/project/agentmemory/)

---

**ملاحظة**: هذا المشروع في مرحلة بيتا. قد تتغير واجهات برمجة التطبيقات في الإصدارات المستقبلية. يرجى الإبلاغ عن أي مشاكل أو اقتراحات!

</div>