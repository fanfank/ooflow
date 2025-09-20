# OoFlow

**A lightweight AI-Ready Python framework for building asynchronous data processing pipelines with stateful nodes.**

Chinese version description please see below : D

文件下半部分有中文说明哟 : D

## Quick Start

```python
import asyncio
import ooflow

@ooflow.Node
async def A(ctx: ooflow.Context):
    count = 0
    while True:
        count = count + 1
        msg = await ctx.fetch()
        await ctx.emit(f"{count} {msg} Hello")

@ooflow.Node
async def B(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg} World")

@ooflow.Node
async def C(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg}!")

async def main():
    # Create and run the flow
    flow = ooflow.create(
        A.to(B),
        B.to(C)
    )

    flow.run()
    count_down = 3
    while count_down > 0:
        count_down = count_down - 1

        await flow.emit("start")
        print(await flow.fetch())
        await asyncio.sleep(1)
    flow.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Installation

```bash
pip install ooflow
```

## Features

What's the biggest difference between OoFlow and other frameworks?
**Your processing functions/methods are stateful, making logic that depends on previous messages possible.**

Other Keypoints:
- **🎯 Stateful Nodes**: Nodes maintain state across messages, enabling complex business logic implementation
- **📝 Super Easy To Use APIs**: Intuitive APIs, no complex Python package dependencies
- **🤖 AI Ready**: Easy to process AI-related streaming messages, such as chat, pic, audio, video 
- **🔄 Asynchronous Processing**: Built on Python's asyncio for high-performance concurrent execution
- **📊 Flexible Topology**: Support for complex graph structures including branching and merging and cycling
- **⚡ Non-blocking Communication**: Efficient message passing between nodes
- **🛡️ Type Safety**: Full type hints support with runtime validation


## Core Concepts

### Nodes
Nodes are the basic processing units in OoFlow. They are defined using the `@ooflow.Node` decorator:

```python
@ooflow.Node
async def my_processor(ctx: ooflow.Context):
    # Fetch data from predecessor nodes
    while True:
        data = await ctx.fetch()
        # data_from_A = await ctx.fetch(A) / ctx.fetch_nowait(A)
        # data_from_B = await ctx.fetch(B) / ctx.fetch_nowait(B)
        # data_from_A_or_B = await ctx.fetch([A, B]) / ctx.fetch_nowait([A, B])
    
        ##############################
        # Your processing logic here #
        ##############################
    
        # Send result to successor nodes
        await ctx.emit(result)
        # await ctx.emit(result, C) / ctx.emit_nowait(result, C)
        # await ctx.emit(result, D) / ctx.emit_nowait(result, D)
        # await ctx.emit(result, [C, D]) / ctx.emit_nowait(result, [C, D])
```

### Context
Each node receives a `Context` object that provides methods for communication:

- `await ctx.fetch()` - Receive messages from all predecessor nodes
- `await ctx.emit(data)` - Send messages to all successor nodes
- `ctx.fetch_nowait()` - Non-blocking message retrieval
- `ctx.emit_nowait(data)` - Non-blocking message sending

If you want to specify the source / target nodes to fetch from / emit to, you can:
- `ctx.fetch_nowait(A)` or `await ctx.fetch(A)` - Receive messages only from node A
- `ctx.fetch_nowait([A, B])` or `await ctx.fetch([A, B])` - Receive messages only from nodes A and B
- `ctx.emit_nowait(data, C)` or `await ctx.emit(data, C)` - Send messages only to node C
- `ctx.emit_nowait(data, [C, D])` or `await ctx.emit(data, [C, D])` - Send messages only to nodes C and D

### Flow Creation
Connect nodes to create processing pipelines:

```python
"""
Flow topology diagram:
    A
    │
    ▼
    B
   ╱ ╲
  ▼   ▼
  C   D
   ╲ ╱
    ▼
    E
"""
flow = ooflow.create(
    A.to(B),           # A → B
    B.to(C, D),        # B → C, D (branching)
    C.to(E),           # C → E
    D.to(E)            # D → E (merging)
)
```

## Advanced Examples

### Branching and Merging

```python
@ooflow.Node
async def splitter(ctx: ooflow.Context):
    data = await ctx.fetch()
    # Send to multiple nodes
    await ctx.emit(data, [branch1, branch2])

@ooflow.Node
async def branch1(ctx: ooflow.Context):
    data = await ctx.fetch()
    result = await process_branch1(data)
    await ctx.emit(result)

@ooflow.Node
async def branch2(ctx: ooflow.Context):
    data = await ctx.fetch()
    result = await process_branch2(data)
    await ctx.emit(result)

@ooflow.Node
async def merger(ctx: ooflow.Context):
    # Collect from both branches
    result1 = await ctx.fetch(branch1)
    result2 = await ctx.fetch(branch2)
    combined = combine_results(result1, result2)
    await ctx.emit(combined)

# Create the flow
flow = ooflow.create(
    splitter.to(branch1, branch2),
    branch1.to(merger),
    branch2.to(merger)
)
```

### Method Decoration

```python
class DataProcessor:
    def __init__(self, multiplier=2):
        self.multiplier = multiplier
        self.processed_count = 0
    
    @ooflow.Node
    async def instance_method(self, ctx: ooflow.Context):
        """Instance method as a node - can access instance state"""
        while True:
            data = await ctx.fetch()
            result = data * self.multiplier
            self.processed_count += 1
            await ctx.emit({"result": result, "count": self.processed_count})
    
    @classmethod
    @ooflow.Node
    async def class_method(cls, ctx: ooflow.Context):
        """Class method as a node - can access class-level information"""
        while True:
            data = await ctx.fetch()
            result = {"processed_by": cls.__name__, "data": data}
            await ctx.emit(result)
    
    @staticmethod
    @ooflow.Node
    async def static_method(ctx: ooflow.Context):
        """Static method as a node - pure function behavior"""
        while True:
            data = await ctx.fetch()
            result = data.upper() if isinstance(data, str) else str(data).upper()
            await ctx.emit(result)

async def main():
    # Create processor instance
    processor = DataProcessor(multiplier=3)

    # Create flow using different method types
    flow = ooflow.create(
        processor.instance_method.to(processor.class_method),
        processor.class_method.to(processor.static_method)
    )

    flow.run()
    count_down = 3
    while count_down > 0:
        count_down = count_down - 1
        await flow.emit("Hello")
        print(await flow.fetch())
        await asyncio.sleep(1)
    flow.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## License

MIT License - see LICENSE file for details.
If you use OoFlow in your project, please cite this repo.

---

# OoFlow

**一个轻量级的 Python 框架，用于构建有状态节点的数据处理图。**

## 快速开始

```python
import asyncio
import ooflow

@ooflow.Node
async def A(ctx: ooflow.Context):
    count = 0
    while True:
        count = count + 1
        msg = await ctx.fetch()
        await ctx.emit(f"{count} {msg} Hello")

@ooflow.Node
async def B(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg} World")

@ooflow.Node
async def C(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg}!")

async def main():
    # 创建并运行流程
    flow = ooflow.create(
        A.to(B),
        B.to(C)
    )

    flow.run()
    count_down = 3
    while count_down > 0:
        count_down = count_down - 1

        await flow.emit("start")
        print(await flow.fetch())
        await asyncio.sleep(1)
    flow.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 安装

```bash
pip install ooflow
```

## 特性

OoFlow 与其它框架最大的区别是什么？
**你的函数/方法节点是有状态的，可以做跨消息逻辑的实现。**

其他要点：
- **🎯 状态节点**: 带状态的处理节点，支持你写更丰富的业务逻辑
- **📝 超级易用的 API**: 极其直观的使用方法，没有复杂 Python 包依赖
- **🤖 AI Ready**: 易于处理 AI 相关的流式消息，如聊天、图片、音频、视频
- **🔄 异步处理**: 基于 Python asyncio 构建，支持高性能并发执行
- **📊 灵活拓扑**: 支持复杂的图结构，包括分支、合并和循环
- **⚡ 非阻塞通信**: 节点间高效的消息传递
- **🛡️ 类型安全**: 完整的类型提示支持和运行时验证

## 核心概念

### 节点
节点是 OoFlow 中的基本处理单元，使用 `@ooflow.Node` 装饰器定义：

```python
@ooflow.Node
async def my_processor(ctx: ooflow.Context):
    # 从前驱节点获取数据
    while True:
        data = await ctx.fetch()
        # data_from_A = await ctx.fetch(A) / ctx.fetch_nowait(A)
        # data_from_B = await ctx.fetch(B) / ctx.fetch_nowait(B)
        # data_from_A_or_B = await ctx.fetch([A, B]) / ctx.fetch_nowait([A, B])
    
        ##############################
        #     你的处理逻辑写在这儿      #
        ##############################
    
        # 将结果发送到后继节点
        await ctx.emit(result)
        # await ctx.emit(result, C) / ctx.emit_nowait(result, C)
        # await ctx.emit(result, D) / ctx.emit_nowait(result, D)
        # await ctx.emit(result, [C, D]) / ctx.emit_nowait(result, [C, D])
```

### 上下文
每个节点接收一个 `Context` 对象，提供通信方法：

- `await ctx.fetch()` - 从所有前驱节点接收消息
- `await ctx.emit(data)` - 向所有后继节点发送消息
- `ctx.fetch_nowait()` - 非阻塞消息检索
- `ctx.emit_nowait(data)` - 非阻塞消息发送

如果你想指定从哪些源节点获取数据或向哪些目标节点发送数据，你可以：
- `ctx.fetch_nowait(A)` 或 `await ctx.fetch(A)` - 仅从节点 A 接收消息
- `ctx.fetch_nowait([A, B])` 或 `await ctx.fetch([A, B])` - 仅从节点 A 和 B 接收消息
- `ctx.emit_nowait(data, C)` 或 `await ctx.emit(data, C)` - 仅向节点 C 发送消息
- `ctx.emit_nowait(data, [C, D])` 或 `await ctx.emit(data, [C, D])` - 仅向节点 C 和 D 发送消息

### 流程创建
连接节点以创建处理管道：

```python
"""
流程拓扑图：
    A
    │
    ▼
    B
   ╱ ╲
  ▼   ▼
  C   D
   ╲ ╱
    ▼
    E
"""
flow = ooflow.create(
    A.to(B),           # A → B
    B.to(C, D),        # B → C, D (分支)
    C.to(E),           # C → E
    D.to(E)            # D → E (合并)
)
```

## 高级示例

### 分支和合并

```python
@ooflow.Node
async def splitter(ctx: ooflow.Context):
    data = await ctx.fetch()
    # 发送到多个节点
    await ctx.emit(data, [branch1, branch2])

@ooflow.Node
async def branch1(ctx: ooflow.Context):
    data = await ctx.fetch()
    result = await process_branch1(data)
    await ctx.emit(result)

@ooflow.Node
async def branch2(ctx: ooflow.Context):
    data = await ctx.fetch()
    result = await process_branch2(data)
    await ctx.emit(result)

@ooflow.Node
async def merger(ctx: ooflow.Context):
    # 从两个分支收集数据
    result1 = await ctx.fetch(branch1)
    result2 = await ctx.fetch(branch2)
    combined = combine_results(result1, result2)
    await ctx.emit(combined)

# 创建流程
flow = ooflow.create(
    splitter.to(branch1, branch2),
    branch1.to(merger),
    branch2.to(merger)
)
```

### 方法装饰

```python
class DataProcessor:
    def __init__(self, multiplier=2):
        self.multiplier = multiplier
        self.processed_count = 0
    
    @ooflow.Node
    async def instance_method(self, ctx: ooflow.Context):
        """实例方法作为节点 - 可以访问实例状态"""
        while True:
            data = await ctx.fetch()
            result = data * self.multiplier
            self.processed_count += 1
            await ctx.emit({"result": result, "count": self.processed_count})
    
    @classmethod
    @ooflow.Node
    async def class_method(cls, ctx: ooflow.Context):
        """类方法作为节点 - 可以访问类级别信息"""
        while True:
            data = await ctx.fetch()
            result = {"processed_by": cls.__name__, "data": data}
            await ctx.emit(result)
    
    @staticmethod
    @ooflow.Node
    async def static_method(ctx: ooflow.Context):
        """静态方法作为节点 - 纯函数行为"""
        while True:
            data = await ctx.fetch()
            result = data.upper() if isinstance(data, str) else str(data).upper()
            await ctx.emit(result)

async def main():
    # 创建处理器实例
    processor = DataProcessor(multiplier=3)

    # 使用不同方法类型创建流程
    flow = ooflow.create(
        processor.instance_method.to(processor.class_method),
        processor.class_method.to(processor.static_method)
    )

    flow.run()
    count_down = 3
    while count_down > 0:
        count_down = count_down - 1
        await flow.emit("Hello")
        print(await flow.fetch())
        await asyncio.sleep(1)
    flow.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 许可证

MIT 许可证 - 详见 LICENSE 文件。
如果你在项目中用了 OoFlow，麻烦也注明、引用下我的仓库，谢谢！