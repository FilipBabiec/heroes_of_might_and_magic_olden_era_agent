import sys

from mcp.server.mcpserver import MCPServer

import tools as t
import graph as g

mcp = MCPServer("heroes-olden-era")
mcp_agent = MCPServer("heroes-olden-era-chatbot")

mcp.tool()(t.query_units)
mcp.tool()(t.search_rules)

mcp_agent.tool()(g.ask)

if __name__ == "__main__":
    t.get_reranker()
    t.get_model()
    t.get_collection()
    print("heroes-olden-era MCP server is ready!")
    mcp.run()
    mcp_agent.run()