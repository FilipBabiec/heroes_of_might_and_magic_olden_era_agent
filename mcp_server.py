import sys

from mcp.server.mcpserver import MCPServer

import tools as t

mcp = MCPServer("heroes-olden-era")

mcp.tool()(t.query_units)
mcp.tool()(t.search_rules)

if __name__ == "__main__":
    t.get_reranker()
    t.get_model()
    t.get_collection()
    print("heroes-olden-era MCP server is ready!")
    mcp.run()