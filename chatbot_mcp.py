import sys
import contextlib
from mcp.server.mcpserver import MCPServer

import tools as t
import graph as g


def ask_hoe(question: str) -> str:
    """Heroes of Might and Magic Olden Era chatbot. It retrieves information about the game rules or units, prints details about them and also gives advices or opinions on the game rules"""
    with contextlib.redirect_stdout(sys.stderr):   # any print inside the graph goes to stderr
        out = g.app.invoke({"prompt": question})
    return out["answer"]


mcp = MCPServer("heroes-olden-era-chatbot")
mcp.tool()(ask_hoe)

if __name__ == "__main__":
    t.get_reranker()
    t.get_model()
    t.get_collection()
    print("heroes-olden-era chatbot MCP server is ready!")
    mcp.run()
