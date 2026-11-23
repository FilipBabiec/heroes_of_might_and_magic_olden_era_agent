import sys

from mcp.server.mcpserver import MCPServer

import tools as t

# The game-data tools. The whole pipeline (ask_hoe) is served by chatbot_mcp.py.
mcp = MCPServer("heroes-olden-era")

mcp.tool()(t.query_units)
mcp.tool()(t.search_rules)

if __name__ == "__main__":
    t.get_reranker()
    t.get_model()
    t.get_collection()
    print("heroes-olden-era MCP server is ready!", file=sys.stderr)  # stdout is the MCP channel
    mcp.run()
