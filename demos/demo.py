#!/usr/bin/env python3
"""
OoFlow test suite demo script
Demonstrate various test scenarios and usage examples

Author: fanfank@github
Date: 2025-09-19
"""

import asyncio
import sys
import os

# Add project root directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ooflow


def print_section(title: str):
    """Print section title"""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)


def print_subsection(title: str):
    """Print subsection title"""
    print(f"\n📋 {title}")
    print("-" * 40)


async def demo_basic_flow():
    """Demonstrate basic flow"""
    print_section("Basic Flow Demo")
    
    @ooflow.Node
    async def source(ctx: ooflow.Context):
        """data source node"""
        print("  🟢 Source: Generate data")
        for i in range(3):
            data = f"data_{i}"
            print(f"     📤 Send: {data}")
            await ctx.emit(data)
            await asyncio.sleep(0.1)
    
    @ooflow.Node
    async def processor(ctx: ooflow.Context):
        """Processor node"""
        print("  🟡 Processor: Start processing")
        while True:
            try:
                data = await asyncio.wait_for(ctx.fetch(), timeout=0.5)
                result = f"processed_{data}"
                print(f"     🔄 Process: {data} -> {result}")
                await ctx.emit(result)
            except asyncio.TimeoutError:
                print("     ⏰ Processor timeout, finished")
                break
    
    @ooflow.Node
    async def sink(ctx: ooflow.Context):
        """Sink node"""
        print("  🔴 Sink: Start receiving")
        results = []
        while True:
            try:
                data = await asyncio.wait_for(ctx.fetch(), timeout=0.5)
                results.append(data)
                print(f"     📥 Receive: {data}")
                await ctx.emit(f"Collected {len(results)} results")
            except asyncio.TimeoutError:
                print("     ⏰ Receiver timeout, finished")
                break
    
    # Create flow
    flow = ooflow.create(
        source.to(processor),
        processor.to(sink)
    )
    
    print(f"\n✅ Flow created successfully:")
    print(f"   - Start nodes: {len(flow.start_nodes)} nodes")
    print(f"   - End nodes: {len(flow.end_nodes)} nodes") 
    print(f"   - Graph count: {len(flow.graphs)} graphs")
    
    # Run flow
    print("\n🚀 Starting flow...")
    flow.run()
    
    # Trigger flow
    await flow.emit("start")
    
    # Wait for processing to complete
    await asyncio.sleep(1.0)
    
    # Get final result
    try:
        result = flow.fetch_nowait()
        print(f"\n🎯 Final result: {result}")
    except asyncio.QueueEmpty:
        print("\n⚠️ No final result (normal case)")
    
    # Stop flow
    flow.stop()
    print("🛑 Flow stopped")


async def demo_branching_flow():
    """Demonstrate branching and merging flow"""
    print_section("Branching and Merging Flow Demo")
    
    @ooflow.Node
    async def splitter(ctx: ooflow.Context):
        """Splitter"""
        data = await ctx.fetch()
        print(f"  🔀 Splitter: Received {data}")
        
        # Distribute to two branches
        branch1_data = f"{data}_branch1"
        branch2_data = f"{data}_branch2"
        
        print(f"     📤 Send to branch1: {branch1_data}")
        await ctx.emit(branch1_data, branch1)
        
        print(f"     📤 Send to branch2: {branch2_data}")
        await ctx.emit(branch2_data, branch2)
    
    @ooflow.Node
    async def branch1(ctx: ooflow.Context):
        """Branch 1 processor"""
        data = await ctx.fetch()
        result = f"b1_processed_{data}"
        print(f"  🟢 Branch1: {data} -> {result}")
        await ctx.emit(result)
    
    @ooflow.Node
    async def branch2(ctx: ooflow.Context):
        """Branch 2 processor"""
        data = await ctx.fetch()
        result = f"b2_processed_{data}"
        print(f"  🟡 Branch2: {data} -> {result}")
        await ctx.emit(result)
    
    @ooflow.Node
    async def merger(ctx: ooflow.Context):
        """Merger"""
        print("  🔄 Merger: Waiting for results from two branches...")
        
        # Wait for results from two branches
        result1 = await ctx.fetch(branch1)
        print(f"     📥 Received from branch1: {result1}")
        
        result2 = await ctx.fetch(branch2)
        print(f"     📥 Received from branch2: {result2}")
        
        # Merge results
        merged = f"merged({result1}, {result2})"
        print(f"     🎯 Merged result: {merged}")
        await ctx.emit(merged)
    
    # Create branching and merging flow
    flow = ooflow.create(
        splitter.to(branch1, branch2),
        branch1.to(merger),
        branch2.to(merger)
    )
    
    print(f"\n✅ Branching flow created successfully")
    
    # Run flow
    print("\n🚀 Starting branching flow...")
    flow.run()
    
    # Send test data
    await flow.emit("test_data")
    
    # Wait for processing to complete
    await asyncio.sleep(0.5)
    
    # Get result
    try:
        result = await asyncio.wait_for(flow.fetch(), timeout=1.0)
        print(f"\n🎉 Branching merge result: {result}")
    except asyncio.TimeoutError:
        print("\n⏰ Timeout waiting for result")
    
    # Stop flow
    flow.stop()
    print("🛑 Branching flow stopped")


def demo_error_scenarios():
    """Demonstrate error scenarios"""
    print_section("Error Scenarios Demo")
    
    print_subsection("1. Invalid Function Signature")
    
    try:
        @ooflow.Node
        async def invalid_no_params():
            return "invalid"
        print("❌ Should throw exception but didn't")
    except ValueError as e:
        print(f"✅ Correctly caught exception: {e}")
    
    try:
        @ooflow.Node
        async def invalid_wrong_type(x: int):
            return x
        print("❌ Should throw exception but didn't")
    except ValueError as e:
        print(f"✅ Correctly caught exception: {e}")
    
    print_subsection("2. Synchronous Function Error")
    
    try:
        @ooflow.Node
        def sync_function(ctx: ooflow.Context):
            return "sync"
        print("❌ Should throw exception but didn't")
    except ValueError as e:
        print(f"✅ Correctly caught exception: {e}")
    
    print_subsection("3. Graph Topology Error")
    
    @ooflow.Node
    async def A(ctx: ooflow.Context):
        pass
    
    @ooflow.Node
    async def B(ctx: ooflow.Context):
        pass
    
    try:
        # Create circular graph (no start/end nodes)
        flow = ooflow.OoFlow(
            A.to(B),
            B.to(A)
        )
        print("❌ Should throw exception but didn't")
    except ValueError as e:
        print(f"✅ Correctly caught topology error: {e}")


def demo_advanced_features():
    """Demonstrate advanced features"""
    print_section("Advanced Features Demo")
    
    print_subsection("1. Multi-Graph Support")
    
    @ooflow.Node
    async def graph1_a(ctx: ooflow.Context):
        pass
    
    @ooflow.Node
    async def graph1_b(ctx: ooflow.Context):
        pass
    
    @ooflow.Node
    async def graph2_x(ctx: ooflow.Context):
        pass
    
    @ooflow.Node
    async def graph2_y(ctx: ooflow.Context):
        pass
    
    # Create flow with two independent graphs
    flow = ooflow.OoFlow(
        graph1_a.to(graph1_b),  # Graph 1
        graph2_x.to(graph2_y)   # Graph 2
    )
    
    print(f"✅ Multi-graph flow created:")
    print(f"   - Graph count: {len(flow.graphs)}")
    print(f"   - Start nodes: {len(flow.start_nodes)} nodes")
    print(f"   - End nodes: {len(flow.end_nodes)} nodes")
    
    print_subsection("2. Class Method Support")
    
    class DataProcessor:
        def __init__(self, name: str):
            self.name = name
        
        @ooflow.Node
        async def process(self, ctx: ooflow.Context):
            data = await ctx.fetch()
            result = f"{self.name}_processed_{data}"
            await ctx.emit(result)
        
        @classmethod
        @ooflow.Node
        async def class_method(cls, ctx: ooflow.Context):
            data = await ctx.fetch()
            await ctx.emit(f"class_processed_{data}")
        
        @staticmethod
        @ooflow.Node
        async def static_method(ctx: ooflow.Context):
            data = await ctx.fetch()
            await ctx.emit(f"static_processed_{data}")
    
    processor = DataProcessor("MyProcessor")
    
    print("✅ Class method nodes created successfully:")
    print(f"   - Instance method: {type(processor.process)}")
    print(f"   - Class method: {type(DataProcessor.class_method)}")
    print(f"   - Static method: {type(DataProcessor.static_method)}")


def demo_test_coverage():
    """Show test coverage scope"""
    print_section("Test Coverage Scope")
    
    print("🎯 Test coverage areas:")
    coverage_areas = [
        "✅ Parameter validation and error handling",
        "✅ Asynchronous message passing mechanism", 
        "✅ Graph construction and topology validation",
        "✅ Concurrent execution and synchronization control",
        "✅ Boundary conditions and exception handling",
        "✅ Class method and instance method support",
        "✅ Complex workflow integration scenarios"
    ]
    
    for area in coverage_areas:
        print(f"   {area}")
    
    print(f"\n📝 For current test statistics, run:")
    print(f"   python tests/run_tests.py")
    print(f"   python tests/test_ooflow.py")


async def main():
    """Main demo function"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "OoFlow Test Suite Demo" + " " * 15 + "║")
    print("║" + " " * 12 + "Complete Feature Demo and Test Coverage" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Run various demos
    await demo_basic_flow()
    await demo_branching_flow()
    demo_error_scenarios()
    demo_advanced_features()
    demo_test_coverage()
    
    print_section("Demo Complete")
    print("🎉 OoFlow test suite demo finished!")
    print("\n📝 Run test suite:")
    print("   python tests/run_tests.py")
    print("   python tests/test_ooflow.py")
    print("\n📚 View test documentation:")
    print("   cat tests/README.md")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted")
        sys.exit(130)
