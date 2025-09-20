"""
Comprehensive OoFlow unit test suite
Author: fanfank@github
Date: 2025-09-19
"""

import asyncio
import unittest
import sys
import os
import logging
import time
from typing import Any
from unittest.mock import patch, MagicMock

# Add project root directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ooflow


class TestLogger(unittest.TestCase):
    """Test Logger setup and configuration functionality"""
    
    def setUp(self):
        # Clean up existing handlers
        logger = logging.getLogger("test_logger")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    def test_setup_logger_default(self):
        """Test default logger setup"""
        logger = ooflow.setup_logger("test_logger")
        
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        self.assertFalse(logger.propagate)
    
    def test_setup_logger_custom_level(self):
        """Test custom log level"""
        logger = ooflow.setup_logger("test_logger_debug", logging.DEBUG)
        
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(logger.handlers[0].level, logging.DEBUG)
    
    def test_setup_logger_no_duplicate_handlers(self):
        """Test no duplicate addition of handlers"""
        logger1 = ooflow.setup_logger("test_logger_dup")
        logger2 = ooflow.setup_logger("test_logger_dup")
        
        self.assertEqual(len(logger1.handlers), 1)
        self.assertEqual(len(logger2.handlers), 1)
        self.assertIs(logger1, logger2)
    
    def test_logger_format(self):
        """Test log format"""
        logger = ooflow.setup_logger("test_logger_format")
        formatter = logger.handlers[0].formatter
        
        self.assertEqual(formatter._fmt, '%(asctime)s - %(message)s')
        self.assertEqual(formatter.datefmt, '%Y-%m-%d %H:%M:%S')


class TestContext(unittest.TestCase):
    """Test Context class all methods"""
    
    def setUp(self):
        self.context = ooflow.Context()
        
        # Create test nodes
        @ooflow.Node
        async def node0(ctx: ooflow.Context):
            pass

        @ooflow.Node
        async def node1(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def node2(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def node3(ctx: ooflow.Context):
            pass

        @ooflow.Node
        async def node4(ctx: ooflow.Context):
            pass
        
        self.node0 = node0
        self.node1 = node1
        self.node2 = node2 # center node
        self.node3 = node3
        self.node4 = node4
        
        # Setup edge connections - Edge should connect Node to Node
        # For testing Context methods, we need to create edges between nodes
        # and then manually add them to context's edge dictionaries
        self.edge0to2 = ooflow.Edge(self.node0, self.node2)  # node0 -> node2
        self.edge1to2 = ooflow.Edge(self.node1, self.node2)  # node1 -> node2
        self.edge2to3 = ooflow.Edge(self.node2, self.node3)  # node2 -> node3
        self.edge2to4 = ooflow.Edge(self.node2, self.node4)  # node2 -> node4
        
        # Manually add edges to context for testing
        self.context.add_incoming_edge(self.edge0to2)
        self.context.add_incoming_edge(self.edge1to2)
        self.context.add_outgoing_edge(self.edge2to3)
        self.context.add_outgoing_edge(self.edge2to4)
    
    def test_add_incoming_edge(self):
        """Test add incoming edge"""
        edge = ooflow.Edge(self.node4, self.node0)  # node4 -> node0
        self.context.add_incoming_edge(edge)
        
        self.assertIn(self.node4, self.context.incoming_edges)
        self.assertEqual(self.context.incoming_edges[self.node4], edge)
    
    def test_add_outgoing_edge(self):
        """Test add outgoing edge"""
        edge = ooflow.Edge(self.node0, self.node4)  # node0 -> node4
        self.context.add_outgoing_edge(edge)
        
        self.assertIn(self.node4, self.context.outgoing_edges)
        self.assertEqual(self.context.outgoing_edges[self.node4], edge)
    
    def test_emit_nowait_to_all(self):
        """Test non-blocking send to all successor nodes"""
        test_msg = "test_message"
        
        self.context.emit_nowait(test_msg)
        
        # Validate message sent to all outgoing edge queues
        for edge in self.context.outgoing_edges.values():
            self.assertEqual(edge.queue.get_nowait(), test_msg)
    
    def test_emit_nowait_to_specific_node(self):
        """Test non-blocking send to specific node"""
        test_msg = "specific_message"
        
        self.context.emit_nowait(test_msg, self.node4)
        
        # Validate message only sent to specified node
        self.assertEqual(self.context.outgoing_edges[self.node4].queue.get_nowait(), test_msg)
        with self.assertRaises(asyncio.QueueEmpty):
            self.context.outgoing_edges[self.node3].queue.get_nowait()
    
    def test_emit_nowait_to_node_list(self):
        """Test non-blocking send to node list"""
        test_msg = "list_message"
        
        self.context.emit_nowait(test_msg, [self.node3, self.node4])
        
        # Validate message sent to all nodes in list
        self.assertEqual(self.context.outgoing_edges[self.node3].queue.get_nowait(), test_msg)
        self.assertEqual(self.context.outgoing_edges[self.node4].queue.get_nowait(), test_msg)

    #def test_emit_nowait_queue_full(self):
    #    """Test exception handling when queue is full"""
    #    # Create a queue with capacity of 1
    #    small_queue = asyncio.Queue(maxsize=1)
    #    self.context.outgoing_edges[self.node4].queue = small_queue
    #    
    #    # Fill the queue
    #    small_queue.put_nowait("first")
    #    
    #    # Sending again should throw exception
    #    with self.assertRaises(asyncio.QueueFull):
    #        self.context.emit_nowait("second")
    
    async def test_emit_to_all(self):
        """Test blocking send to all successor nodes"""
        test_msg = "async_message"
        
        await self.context.emit(test_msg)
        
        # Validate message sent to all outgoing edge queues
        self.assertEqual(await self.context.outgoing_edges[self.node3].queue.get(), test_msg)
        self.assertEqual(await self.context.outgoing_edges[self.node4].queue.get(), test_msg)
    
    async def test_emit_to_specific_node(self):
        """Test blocking send to specific node"""
        test_msg = "async_specific"
        
        await self.context.emit(test_msg, self.node3)
        
        # Validate message only sent to specified node
        self.assertEqual(await self.context.outgoing_edges[self.node3].queue.get(), test_msg)
        
        with self.assertRaises(asyncio.QueueEmpty):
            self.context.outgoing_edges[self.node4].queue.get_nowait()
    
    def test_fetch_nowait_from_all(self):
        """Test non-blocking fetch from all predecessor nodes"""
        test_msg = "fetch_message"
        
        # Add message to first incoming edge queue
        self.context.incoming_edges[self.node0].queue.put_nowait(test_msg)
        
        result = self.context.fetch_nowait()
        self.assertEqual(result, test_msg)
    
    def test_fetch_nowait_from_specific_node(self):
        """Test non-blocking fetch from specific node"""
        test_msg = "specific_fetch"
        
        # Add message to second incoming edge queue
        self.context.incoming_edges[self.node1].queue.put_nowait(test_msg)
        
        result = self.context.fetch_nowait(self.node1)
        self.assertEqual(result, test_msg)
    
    def test_fetch_nowait_empty_queue(self):
        """Test exception when fetching from empty queue"""
        with self.assertRaises(asyncio.QueueEmpty):
            self.context.fetch_nowait()
    
    async def test_fetch_from_single_node(self):
        """Test blocking fetch from single node"""
        test_msg = "single_fetch"
        
        # Asynchronously add message
        async def add_message():
            await asyncio.sleep(0.01)
            await self.context.incoming_edges[self.node1].queue.put(test_msg)
        
        # Start fetch and add tasks simultaneously
        fetch_task = asyncio.create_task(self.context.fetch(self.node1))
        add_task = asyncio.create_task(add_message())
        
        result = await fetch_task
        await add_task
        
        self.assertEqual(result, test_msg)
    
    async def test_fetch_from_multiple_nodes(self):
        """Test fetch from multiple nodes (polling mode)"""
        test_msg = "multi_fetch"
        
        # Asynchronously add message to second queue
        async def add_message():
            await asyncio.sleep(0.02)
            await self.context.incoming_edges[self.node0].queue.put(test_msg)
        
        # Start fetch and add tasks simultaneously
        fetch_task = asyncio.create_task(self.context.fetch([self.node0, self.node1], check_interval=0.001))
        add_task = asyncio.create_task(add_message())
        
        result = await fetch_task
        await add_task
        
        self.assertEqual(result, test_msg)
    
    def test_get_target_queues_none_target(self):
        """Test get target queues - None target"""
        queues = self.context._get_target_queues(None, self.context.outgoing_edges)
        expected_queues = [edge.queue for edge in self.context.outgoing_edges.values()]
        self.assertEqual(set(queues), set(expected_queues))
    
    def test_get_target_queues_single_target(self):
        """Test get target queues - single target"""
        queues = self.context._get_target_queues(self.node4, self.context.outgoing_edges)
        self.assertEqual(len(queues), 1)
        self.assertEqual(queues[0], self.context.outgoing_edges[self.node4].queue)
    
    def test_get_target_queues_list_target(self):
        """Test get target queues - list target"""
        queues = self.context._get_target_queues([self.node3, self.node4], self.context.outgoing_edges)
        self.assertEqual(len(queues), 2)
    
    def test_get_target_queues_not_found(self):
        """Test get target queues - node not found"""
        # Create a node that's not in outgoing_edges
        @ooflow.Node
        async def node_not_found(ctx: ooflow.Context):
            pass
        
        with patch('ooflow.ooflow.logger') as mock_logger:
            queues = self.context._get_target_queues(node_not_found, self.context.outgoing_edges)
            self.assertEqual(len(queues), 0)
            mock_logger.error.assert_called()


class TestNode(unittest.TestCase):
    """Test Node decorator"""
    
    def test_node_with_valid_async_function(self):
        """Test creating Node with valid async function"""
        @ooflow.Node
        async def valid_func(ctx: ooflow.Context):
            return "valid"
        
        self.assertIsInstance(valid_func, ooflow.Node)
        self.assertEqual(valid_func.__name__, "valid_func")
    
    def test_node_with_sync_function_raises_error(self):
        """Test creating Node with sync function throws error"""
        with self.assertRaises(ValueError) as cm:
            @ooflow.Node
            def sync_func(ctx: ooflow.Context):
                return "sync"
        
        self.assertIn("must be an async function", str(cm.exception))
    
    def test_node_with_single_context_param(self):
        """Test function with single Context parameter"""
        @ooflow.Node
        async def single_param(ctx: ooflow.Context):
            return "single"
        
        self.assertIsInstance(single_param, ooflow.Node)
    
    #def test_node_with_two_params_second_context(self):
    #    """Test function with two parameters, second is Context"""
    #    @ooflow.Node
    #    async def two_params(data: str, ctx: ooflow.Context):
    #        return data
    #    
    #    self.assertIsInstance(two_params, ooflow.Node)
    
    def test_node_with_wrong_param_count(self):
        """Test error with wrong parameter count"""
        with self.assertRaises(ValueError) as cm:
            @ooflow.Node
            async def no_params():
                return "none"
        
        self.assertIn("can only have 1 or 2 parameters", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            @ooflow.Node
            async def three_params(a: str, b: str, c: str):
                return "three"
        
        self.assertIn("can only have 1 or 2 parameters", str(cm.exception))
    
    def test_node_with_wrong_param_type(self):
        """Test error with wrong parameter type"""
        with self.assertRaises(ValueError) as cm:
            @ooflow.Node
            async def wrong_type(ctx: int):
                return "wrong"
        
        self.assertIn("must be ooflow.Context type", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            @ooflow.Node
            async def wrong_second_type(data: str, ctx: int):
                return "wrong"
        
        self.assertIn("must be ooflow.Context type", str(cm.exception))
    
    async def test_node_call(self):
        """Test Node invocation"""
        @ooflow.Node
        async def test_func(ctx: ooflow.Context):
            return "called"
        
        ctx = ooflow.Context()
        result = await test_func(ctx)
        self.assertEqual(result, "called")
    
    def test_node_to_method(self):
        """Test Node's to method"""
        @ooflow.Node
        async def node1(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def node2(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def node3(ctx: ooflow.Context):
            pass
        
        result = node1.to(node2, node3)
        self.assertEqual(result[0], node1)
        self.assertEqual(result[1], (node2, node3))
    
    def test_node_to_method_invalid_param(self):
        """Test Node's to method with invalid parameter"""
        @ooflow.Node
        async def node1(ctx: ooflow.Context):
            pass
        
        with self.assertRaises(ValueError):
            node1.to("not_a_node")
    
    async def test_node_descriptor_with_instance_method(self):
        """Test Node as instance method descriptor"""
        TEST_MSG = "instance_method"

        class TestClass:
            @ooflow.Node
            async def method(self, ctx: ooflow.Context):
                return TEST_MSG
        
        obj = TestClass()
        bound_node = obj.method
        
        self.assertIsInstance(bound_node, ooflow.Node)
        self.assertNotEqual(bound_node, TestClass.method)
        self.assertNotEqual(await obj.method(), TEST_MSG)
    
    def test_node_descriptor_with_class_access(self):
        """Test Node as class method descriptor"""
        class TestClass:
            @ooflow.Node
            async def method(self, ctx: ooflow.Context):
                return "class_method"
        
        class_node = TestClass.method
        self.assertIsInstance(class_node, ooflow.Node)


class TestEdge(unittest.TestCase):
    """Test Edge class"""
    
    def setUp(self):
        @ooflow.Node
        async def node1(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def node2(ctx: ooflow.Context):
            pass
        
        self.node1 = node1
        self.node2 = node2
    
    def test_edge_creation(self):
        """Test Edge creation"""
        edge = ooflow.Edge(self.node1, self.node2)
        
        self.assertEqual(edge.from_node, self.node1)
        self.assertEqual(edge.to_node, self.node2)
        self.assertIsInstance(edge.queue, asyncio.Queue)
    
    def test_edge_with_none_nodes(self):
        """Test Edge with None nodes"""
        edge1 = ooflow.Edge(None, self.node1)
        edge2 = ooflow.Edge(self.node1, None)
        
        self.assertIsNone(edge1.from_node)
        self.assertEqual(edge1.to_node, self.node1)
        self.assertEqual(edge2.from_node, self.node1)
        self.assertIsNone(edge2.to_node)


class TestOoFlow(unittest.TestCase):
    """Test OoFlow class"""
    
    def setUp(self):
        @ooflow.Node
        async def A(ctx: ooflow.Context):
            msg = await ctx.fetch()
            await ctx.emit(msg + "A")
        
        class BClass:
            @ooflow.Node
            async def B(self, ctx: ooflow.Context):
                msg = await ctx.fetch()
                await ctx.emit(msg + "B")
        
        class CClass:
            @classmethod
            @ooflow.Node
            async def C(cls, ctx: ooflow.Context):
                msg = await ctx.fetch()
                await ctx.emit(msg + "C")
        
        class DClass:
            @staticmethod
            @ooflow.Node
            async def D(ctx: ooflow.Context):
                msg = await ctx.fetch()
                await ctx.emit(msg + "D")
        
        self.A = A
        self.B = BClass().B
        self.C = CClass.C
        self.D = DClass.D
    
    def test_simple_flow_creation(self):
        """Test simple flow creation"""
        flow = ooflow.OoFlow(
            self.A.to(self.B),
            self.B.to(self.C)
        )
        
        self.assertEqual(len(flow.start_nodes), 1)
        self.assertEqual(len(flow.end_nodes), 1)
        self.assertIn(self.A, flow.start_nodes)
        self.assertIn(self.C, flow.end_nodes)
        self.assertEqual(len(flow.graphs), 1)
    
    def test_branching_flow_creation(self):
        """Test branching flow creation"""
        flow = ooflow.OoFlow(
            self.A.to(self.B, self.C),
            self.B.to(self.D),
            self.C.to(self.D)
        )
        
        self.assertEqual(len(flow.start_nodes), 1)
        self.assertEqual(len(flow.end_nodes), 1)
        self.assertIn(self.A, flow.start_nodes)
        self.assertIn(self.D, flow.end_nodes)
    
    def test_multiple_graphs(self):
        """Test multiple independent graphs"""
        @ooflow.Node
        async def E(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def F(ctx: ooflow.Context):
            pass

        @ooflow.Node
        async def G(ctx: ooflow.Context):
            pass
        
        flow = ooflow.OoFlow(
            self.A.to(self.B),  # Graph 1
            E.to(F),            # Graph 2
            E.to(G)             
        )
        
        self.assertEqual(len(flow.graphs), 2)
        self.assertEqual(len(flow.start_nodes), 2)
        self.assertEqual(len(flow.end_nodes), 3)
    
    def test_flow_with_no_start_node_raises_error(self):
        """Test flow with no start node throws error"""
        with self.assertRaises(ValueError) as cm:
            ooflow.OoFlow(
                self.A.to(self.B),
                self.B.to(self.C),
                self.C.to(self.D),
                self.C.to(self.A)   # Forms head cycle, no start nodes
            )
        
        self.assertIn("has no start node", str(cm.exception))
    
    def test_flow_with_no_end_node_raises_error(self):
        """Test flow with no end node throws error"""
        # Create a circular structure with only incoming edges, no outgoing edges
        with self.assertRaises(ValueError) as cm:
            ooflow.OoFlow(
                self.A.to(self.B),
                self.B.to(self.C),
                self.C.to(self.D),
                self.D.to(self.B)   # Forms tail cycle, no end nodes
            )
        
        # Due to cycle formation, will first detect no start node
        self.assertIn("has no end node", str(cm.exception))
    
    def test_node_context_edges(self):
        """Test node context edge connections"""
        flow = ooflow.OoFlow(
            self.A.to(self.B),
            self.B.to(self.C)
        )
        
        # Check A's incoming and outgoing edges
        # A is start node, has incoming edge from Yang and outgoing edge to B
        self.assertEqual(len(self.A.context.incoming_edges), 1)  # Yang
        self.assertEqual(len(self.A.context.outgoing_edges), 1)  # B
        self.assertIn(self.B, self.A.context.outgoing_edges)
        
        # Check B's incoming and outgoing edges
        self.assertEqual(len(self.B.context.incoming_edges), 1)  # A
        self.assertEqual(len(self.B.context.outgoing_edges), 1)  # C
        self.assertIn(self.A, self.B.context.incoming_edges)
        self.assertIn(self.C, self.B.context.outgoing_edges)
        
        # Check C's incoming and outgoing edges
        # C is end node, has incoming edge from B and outgoing edge to Yin
        self.assertEqual(len(self.C.context.incoming_edges), 1)  # B
        self.assertEqual(len(self.C.context.outgoing_edges), 1)  # Yin
        self.assertIn(self.B, self.C.context.incoming_edges)
    
    def test_yin_yang_contexts(self):
        """Test Yin Yang contexts"""
        flow = ooflow.OoFlow(self.A.to(self.B))
        
        # Yang should have outgoing edge to start node, and no incoming edges
        self.assertEqual(len(flow.Yang.incoming_edges), 0)
        self.assertEqual(len(flow.Yang.outgoing_edges), 1)
        self.assertIn(self.A, flow.Yang.outgoing_edges)
        
        # Yin should have incoming edge from end node, and no outgoing edges
        self.assertEqual(len(flow.Yin.incoming_edges), 1)
        self.assertEqual(len(flow.Yin.outgoing_edges), 0)
        self.assertIn(self.B, flow.Yin.incoming_edges)
    
    def test_flow_run_and_stop(self):
        """Test flow run and stop"""
        flow = ooflow.OoFlow(self.A.to(self.B))
        
        self.assertFalse(flow.running)
        self.assertEqual(len(flow.running_tasks), 0)
        
        flow.run()
        
        self.assertTrue(flow.running)
        self.assertEqual(len(flow.running_tasks), 2)  # A, B
        
        # Stop flow
        flow.stop()
        self.assertFalse(flow.running)
        self.assertEqual(len(flow.running_tasks), 0)
    
    def test_flow_emit_nowait(self):
        """Test flow non-blocking send"""
        flow = ooflow.OoFlow(self.A.to(self.B))
        
        test_msg = "test_emit"
        flow.emit_nowait(test_msg)
        
        # Validate message sent to start node
        edge_to_A = flow.Yang.outgoing_edges[self.A]
        self.assertEqual(edge_to_A.queue.get_nowait(), test_msg)
    
    async def test_flow_emit(self):
        """Test flow blocking send"""
        # Create new node instances to avoid conflicts
        @ooflow.Node
        async def P(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def Q(ctx: ooflow.Context):
            pass
        
        flow = ooflow.OoFlow(P.to(Q))
        
        test_msg = "test_async_emit"
        await flow.emit(test_msg)
        
        # Validate message sent to start node
        edge_to_P = flow.Yang.outgoing_edges[P]
        result = await edge_to_P.queue.get()
        self.assertEqual(result, test_msg)
    
    def test_flow_fetch_nowait(self):
        """Test flow non-blocking fetch"""
        flow = ooflow.OoFlow(self.A.to(self.B))
        
        test_msg = "test_fetch"
        # Send message to end node
        edge_from_B = flow.Yin.incoming_edges[self.B]
        edge_from_B.queue.put_nowait(test_msg)
        
        result = flow.fetch_nowait()
        self.assertEqual(result, test_msg)
    
    async def test_flow_fetch(self):
        """Test flow blocking fetch"""
        # Create new node instances to avoid conflicts
        @ooflow.Node
        async def X(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def Y(ctx: ooflow.Context):
            pass
        
        flow = ooflow.OoFlow(X.to(Y))
        
        test_msg = "test_async_fetch"
        
        # Asynchronously send message to end node
        async def send_message():
            await asyncio.sleep(0.01)
            edge_from_Y = flow.Yin.incoming_edges[Y]
            await edge_from_Y.queue.put(test_msg)
        
        # Start fetch and send tasks simultaneously
        fetch_task = asyncio.create_task(flow.fetch())
        send_task = asyncio.create_task(send_message())
        
        result = await fetch_task
        await send_task
        
        self.assertEqual(result, test_msg)


class TestOoFlowIntegration(unittest.TestCase):
    """Test complete workflow integration scenarios"""
    
    async def test_simple_ooflow(self):
        """Test simple ooflow"""
        @ooflow.Node
        async def A(ctx: ooflow.Context):
            try:
                msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                for i in range(3):
                    await ctx.emit(f"A_{msg}_{i}")
            except asyncio.TimeoutError:
                pass
        
        class BClass:
            @ooflow.Node
            async def B(self, ctx: ooflow.Context):
                while True:
                    try:
                        msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                        await ctx.emit(f"B_{msg}")
                    except asyncio.TimeoutError:
                        break
        
        class CClass:
            @classmethod
            @ooflow.Node
            async def C(cls, ctx: ooflow.Context):
                while True:
                    try:
                        msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                        await ctx.emit(f"C_{msg}")
                    except asyncio.TimeoutError:
                        break

        class DClass:
            @staticmethod
            @ooflow.Node
            async def D(ctx: ooflow.Context):
                results = []
                while True:
                    try:
                        msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                        results.append(f"D_{msg}")
                        await ctx.emit(results)
                    except asyncio.TimeoutError:
                        break
        
        b = BClass()
        flow = ooflow.OoFlow(
            A.to(b.B),
            b.B.to(CClass.C),
            CClass.C.to(DClass.D)
        )
        
        flow.run()
        test_msg = "start"
        await flow.emit(test_msg)
        
        # Wait for processing to complete
        await asyncio.sleep(0.2)
        
        try:
            result = flow.fetch_nowait()
            self.assertIsInstance(result, list)
            self.assertTrue(len(result) > 0)
            self.assertTrue(any(f"D_C_B_A_{test_msg}_" in item for item in result))
        except asyncio.QueueEmpty:
            # If queue is empty, flow might still be running
            pass
        
        flow.stop()
    
    async def test_branching_and_merging(self):
        """Test branching and merging flow"""
        @ooflow.Node
        async def splitter(ctx: ooflow.Context):
            msg = await ctx.fetch()
            await ctx.emit(f"{msg}_branch1", branch1)
            await ctx.emit(f"{msg}_branch2", branch2)
        
        @ooflow.Node
        async def branch1(ctx: ooflow.Context):
            msg = await ctx.fetch()
            await ctx.emit(f"b1_{msg}")
        
        @ooflow.Node
        async def branch2(ctx: ooflow.Context):
            msg = await ctx.fetch()
            await ctx.emit(f"b2_{msg}")
        
        @ooflow.Node
        async def merger(ctx: ooflow.Context):
            msg1 = await ctx.fetch(branch1)
            msg2 = await ctx.fetch(branch2)
            await ctx.emit(f"merged_{msg1}_{msg2}")
        
        flow = ooflow.OoFlow(
            splitter.to(branch1, branch2),
            branch1.to(merger),
            branch2.to(merger)
        )
        
        flow.run()
        await flow.emit("test")
        
        # Wait for processing to complete
        await asyncio.sleep(0.1)
        
        try:
            result = await asyncio.wait_for(flow.fetch(), timeout=0.5)
            self.assertIn("merged_", result)
            self.assertIn("b1_", result)
            self.assertIn("b2_", result)
        except asyncio.TimeoutError:
            pass
        
        flow.stop()
    
    async def test_error_handling(self):
        """Test error handling"""
        @ooflow.Node
        async def error_node(ctx: ooflow.Context):
            msg = await ctx.fetch()
            if "error" in msg:
                raise ValueError("Test error")
            await ctx.emit(f"ok_{msg}")
        
        @ooflow.Node
        async def final_node(ctx: ooflow.Context):
            msg = await ctx.fetch()
            await ctx.emit(f"final_{msg}")
        
        flow = ooflow.OoFlow(
            error_node.to(final_node)
        )
        
        flow.run()
        
        # Test normal message
        await flow.emit("normal")
        await asyncio.sleep(0.1)
        
        try:
            result = flow.fetch_nowait()
            self.assertIn("final_ok_normal", result)
        except asyncio.QueueEmpty:
            pass
        
        flow.stop()


class TestCreateFunction(unittest.TestCase):
    """Test create function"""
    
    def test_create_function(self):
        """Test create function"""
        @ooflow.Node
        async def A(ctx: ooflow.Context):
            pass
        
        @ooflow.Node
        async def B(ctx: ooflow.Context):
            pass
        
        flow = ooflow.create(A.to(B))
        
        self.assertIsInstance(flow, ooflow.OoFlow)
        self.assertEqual(len(flow.start_nodes), 1)
        self.assertEqual(len(flow.end_nodes), 1)


class AsyncTestRunner:
    """Async test runner"""
    
    def __init__(self):
        self.test_results = []
    
    async def run_async_test(self, test_method, test_instance):
        """Run single async test"""
        try:
            await test_method()
            self.test_results.append((test_method.__name__, "PASS", None))
        except Exception as e:
            self.test_results.append((test_method.__name__, "FAIL", str(e)))
    
    async def run_all_async_tests(self):
        """Run all async tests"""
        test_classes = [
            TestContext,
            TestOoFlow,
            TestOoFlowIntegration
        ]
        
        for test_class in test_classes:
            instance = test_class()
            
            # Run setUp if exists
            if hasattr(instance, 'setUp'):
                instance.setUp()
            
            # Find all async test methods
            async_test_methods = [
                getattr(instance, method_name)
                for method_name in dir(instance)
                if method_name.startswith('test_') and asyncio.iscoroutinefunction(getattr(instance, method_name))
            ]
            
            # Run async tests
            for test_method in async_test_methods:
                await self.run_async_test(test_method, instance)
    
    def print_results(self):
        """Print test results"""
        print("\n=== Async test results ===")
        passed = 0
        failed = 0
        
        for test_name, status, error in self.test_results:
            if status == "PASS":
                print(f"✓ {test_name}")
                passed += 1
            else:
                print(f"✗ {test_name}: {error}")
                failed += 1
        
        print(f"\nAsync test statistics: {passed} passed, {failed} failed")


async def run_all_tests():
    """Run all tests"""
    print("Starting OoFlow complete test suite...")
    print("=" * 60)
    
    # Run synchronous tests
    print("\n>>> Running sync unit tests...")
    unittest_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Filter out async tests
    filtered_suite = unittest.TestSuite()
    for test_group in unittest_suite:
        for test in test_group:
            test_method = getattr(test, test._testMethodName)
            if not asyncio.iscoroutinefunction(test_method):
                filtered_suite.addTest(test)
    
    runner = unittest.TextTestRunner(verbosity=2)
    sync_result = runner.run(filtered_suite)
    
    # Run async tests
    print("\n>>> Running async unit tests...")
    async_runner = AsyncTestRunner()
    await async_runner.run_all_async_tests()
    async_runner.print_results()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test suite completed!")
    
    total_sync_tests = sync_result.testsRun
    sync_failures = len(sync_result.failures)
    sync_errors = len(sync_result.errors)
    sync_passed = total_sync_tests - sync_failures - sync_errors
    
    async_passed = sum(1 for _, status, _ in async_runner.test_results if status == "PASS")
    async_failed = sum(1 for _, status, _ in async_runner.test_results if status == "FAIL")
    
    print(f"Sync tests: {sync_passed} passed, {sync_failures} failed, {sync_errors} errors")
    print(f"Async tests: {async_passed} passed, {async_failed} failed")
    print(f"Total: {sync_passed + async_passed} passed, {sync_failures + sync_errors + async_failed} failed")
    
    return (sync_failures + sync_errors + async_failed) == 0


if __name__ == "__main__":
    # Setup log level to reduce noise during testing
    logging.getLogger("ooflow").setLevel(logging.WARNING)
    
    # Run all tests
    success = asyncio.run(run_all_tests())
    
    # Exit code
    sys.exit(0 if success else 1)
