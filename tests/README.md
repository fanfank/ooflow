# OoFlow Test Suite

This is the complete test suite for OoFlow, including unit tests and integration tests.

## Quick Start

### 一键运行所有测试
```bash
# 在项目根目录下运行
python tests/run_tests.py
```

### 或者直接运行测试文件
```bash
python tests/test_ooflow.py
```

## 测试结构

### 测试模块

| 测试类 | 功能描述 | 测试数量 |
|--------|----------|----------|
| `TestLogger` | Logger setup and configuration功能 | 4 |
| `TestContext` | Context 类的消息传递方法 | 12 |
| `TestNode` | Node 装饰器和参数验证 | 10 |
| `TestEdge` | Edge 类和队列机制 | 3 |
| `TestOoFlow` | OoFlow 核心功能 | 12 |
| `TestOoFlowIntegration` | 完整工作流集成场景 | 3 |
| `TestCreateFunction` | create 函数 | 1 |

### Test Coverage范围

#### 🔧 核心功能测试
- ✅ Node 装饰器的参数验证
- ✅ Context 的消息发送和接收
- ✅ OoFlow 的图构建和拓扑验证
- ✅ 异步任务的并发执行

#### 🚀 高级功能测试
- ✅ 分支和合并流程
- ✅ 多图独立执行
- ✅ 错误处理和异常传播
- ✅ 边界条件处理

#### 🎯 集成测试
- ✅ 简单管道流程
- ✅ 复杂分支合并场景
- ✅ 实际工作流模拟

## 运行选项

### 基本运行
```bash
python tests/run_tests.py
```

### 详细输出
```bash
python tests/run_tests.py --verbose
```

### 显示运行时间
```bash
python tests/run_tests.py --time
```

### 查看测试模块信息
```bash
python tests/run_tests.py --help-tests
```

### 不显示横幅
```bash
python tests/run_tests.py --no-banner
```

## 测试输出示例

### 成功运行
```
╔══════════════════════════════════════════════════════════╗
║                OoFlow 测试套件                        ║
║            完整的单元测试和集成测试                      ║
╚══════════════════════════════════════════════════════════╝

开始运行 OoFlow 完整测试套件...
============================================================

>>> 运行同步单元测试...
test_setup_logger_default (test_ooflow.TestLogger) ... ok
test_setup_logger_custom_level (test_ooflow.TestLogger) ... ok
...

>>> 运行异步单元测试...
=== 异步测试结果 ===
✓ test_emit_to_all
✓ test_emit_to_specific_node
...

============================================================
测试套件运行完成!
同步测试: 35 通过, 0 失败, 0 错误
异步测试: 8 通过, 0 失败
总计: 43 通过, 0 失败

🎉 所有测试通过！OoFlow 工作正常。
```

## 测试详细说明

### Logger 测试 (`TestLogger`)
- 测试默认 logger 配置
- 测试自定义日志级别
- 测试防重复 handler 机制
- 测试日志格式

### Context 测试 (`TestContext`)
- 测试消息的同步/异步发送 (`emit`, `emit_nowait`)
- 测试消息的同步/异步接收 (`fetch`, `fetch_nowait`)
- 测试目标节点指定
- 测试队列满和空的异常处理
- 测试多节点轮询机制

### Node 测试 (`TestNode`)
- 测试函数签名验证
- 测试异步函数要求
- 测试参数类型检查
- 测试方法绑定机制
- 测试 `to` 方法功能

### OoFlow 测试 (`TestOoFlow`)
- 测试简单和复杂图构建
- 测试起始/结束节点识别
- 测试多图支持
- 测试 Yang/Yin 虚拟节点
- 测试运行时控制

### 集成测试 (`TestOoFlowIntegration`)
- 测试完整的数据管道
- 测试分支合并场景
- 测试错误处理流程

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'ooflow'
   ```
   确保在项目根目录运行测试，或正确设置 Python 路径。

2. **异步测试超时**
   ```
   asyncio.TimeoutError
   ```
   某些集成测试可能需要更多时间，这是正常的。

3. **队列相关错误**
   ```
   asyncio.QueueEmpty
   ```
   这通常是测试中的预期行为，用于验证边界条件。

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.getLogger("ooflow").setLevel(logging.DEBUG)
   ```

2. **单独运行特定测试**
   ```bash
   python -m unittest test_ooflow.TestContext.test_emit_nowait_to_all
   ```

3. **使用 Python 调试器**
   ```bash
   python -m pdb tests/test_ooflow.py
   ```

## 扩展测试

如果需要添加新的测试用例：

1. 在相应的测试类中添加测试方法
2. 方法名以 `test_` 开头
3. 异步测试使用 `async def`
4. 使用 `self.assert*` 方法进行断言

示例：
```python
def test_new_feature(self):
    """测试新功能"""
    # 测试代码
    self.assertEqual(expected, actual)

async def test_async_feature(self):
    """测试异步新功能"""
    # 异步测试代码
    result = await some_async_function()
    self.assertIsNotNone(result)
```
