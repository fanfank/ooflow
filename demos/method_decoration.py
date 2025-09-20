import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ooflow

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
