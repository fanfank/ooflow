import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ooflow

@ooflow.Node
async def A(context: ooflow.Context):
    while True:
        msg = await context.fetch()
        await context.emit(f"{msg} A | ")

@ooflow.Node
async def B(context: ooflow.Context):
    while True:
        msg = await context.fetch()
        await context.emit(f"{msg} B | ", C)
        await context.emit(f"{msg} B | ", D)

        # # you can also emit to C, D all at once
        # await context.emit(f"{msg} B | ")

@ooflow.Node
async def C(context: ooflow.Context):
    while True:
        msg = await context.fetch()
        await context.emit(f"{msg} C | ")

@ooflow.Node
async def D(context: ooflow.Context):
    while True:
        msg = await context.fetch()
        await context.emit(f"{msg} D | ")

@ooflow.Node
async def E(context: ooflow.Context):
    while True:
        msg_from_C = await context.fetch(C)
        msg_from_D = await context.fetch(D)
        await context.emit(f"{msg_from_C} E")
        await context.emit(f"{msg_from_D} E")

        # # you can also fetch from C, D in one line
        # msg = await context.fetch()
        # await context.emit(f"{msg} E")

async def main():
    flow = ooflow.create(
        A.to(B),
        B.to(C, D),
        C.to(E),
        D.to(E)
    )
    flow.run()

    async def producer():
        count = 0
        while True:
            count = count + 1
            await flow.emit(f"{count}")
            await asyncio.sleep(1)

    asyncio.create_task(producer()),
    while True:
        print(await flow.fetch())

if __name__ == "__main__":
    asyncio.run(main())
