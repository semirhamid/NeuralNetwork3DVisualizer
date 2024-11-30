import asyncio

from app.models.abc import start_server


async def main():
    print("Starting WebSocket server on ws://localhost:8000")
    async with start_server():
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.run_forever()
