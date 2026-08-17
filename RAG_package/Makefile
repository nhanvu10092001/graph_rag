# Makefile for LLM Utils MCP Package
# ===================================

.PHONY: setup-mcp setup-rag setup-graph-rag setup-all run-fastmcp simple-fastmcp cvt-fastmcp-2-openapi mcp-io-demo test-rag run-graph-rag-demo rag-mcp-demo rag-indexing-api langchain-mcp-demo help
.DEFAULT_GOAL := help

# Variables
UV := uv
PACKAGE_DIR := packages/llm-utils-mcp
EXAMPLE_DIR := examples

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
RESET := \033[0m

help: ## Show available commands
	@echo "$(CYAN)LLM Utils MCP - Commands$(RESET)"
	@echo "$(CYAN)========================$(RESET)"
	@echo ""
	@echo "$(GREEN)setup-mcp$(RESET)              - Setup MCP package with all dependencies"
	@echo "$(GREEN)setup-rag$(RESET)              - Setup RAG package with all dependencies"
	@echo "$(GREEN)setup-graph-rag$(RESET)        - Setup Graph RAG (Neo4j) package with all dependencies"
	@echo "$(GREEN)setup-all$(RESET)              - Setup all packages (MCP, RAG, Graph RAG) (complete install)"
	@echo "$(GREEN)run-fastmcp$(RESET)            - Create and run FastMCP server (port 8000)"
	@echo "$(GREEN)simple-fastmcp$(RESET)         - Start simple FastMCP server with basic tools (port 8000)"
	@echo "$(GREEN)cvt-fastmcp-2-openapi$(RESET)  - Convert MCP to OpenAPI using MCPO (port 8001)"
	@echo "$(GREEN)mcp-io-demo$(RESET)            - Start FastMCP server with MCP-IO tools (port 8002)"
	@echo "$(GREEN)rag-mcp-demo$(RESET)           - Start FastMCP server with RAG tools (port 8003)"
	@echo "$(GREEN)rag-indexing-api$(RESET)       - Start RAG document indexing API (port 9999)"
	@echo "$(GREEN)langchain-mcp-demo$(RESET)     - Run LangChain agent with MCP client integration"
	@echo "$(GREEN)test-rag$(RESET)               - Interactive RAG feature demo (sync/async + index/query)"
	@echo "$(GREEN)run-graph-rag-demo$(RESET)     - Interactive Neo4j Graph RAG demo (index + graph traversal)"
	@echo ""
	@echo "$(BLUE)Quick Start:$(RESET)"
	@echo "  make setup-mcp              # MCP-only setup"
	@echo "  make simple-fastmcp         # Run simple FastMCP server"
	@echo "  make setup-all              # Complete setup (MCP + RAG + Graph RAG)"
	@echo "  make run-graph-rag-demo     # Run Graph RAG demo"
	@echo "  make test-rag               # Run RAG demo"
	@echo ""

setup-mcp: ## Setup MCP package with all dependencies using UV
	@echo "$(BLUE)🔧 Setting up MCP package...$(RESET)"
	@echo "$(BLUE)Installing workspace with MCP full features...$(RESET)"
	$(UV) sync --group mcp-full
	@echo "$(GREEN)✅ MCP setup complete!$(RESET)"
	@echo ""
	@echo "$(CYAN)Next step: make run-fastmcp$(RESET)"

setup-rag: ## Setup RAG package with all dependencies using UV
	@echo "$(BLUE)🧠 Setting up RAG package...$(RESET)"
	@echo "$(BLUE)Installing workspace with RAG full features...$(RESET)"
	$(UV) sync --group rag
	@echo "$(GREEN)✅ RAG dependencies installed!$(RESET)"
	@echo ""
	@echo "$(MAGENTA)📋 Additional Setup Required:$(RESET)"
	@echo "$(MAGENTA)1. Choose and setup your LLM provider(s):$(RESET)"
	@echo ""
	@echo "$(CYAN)   📡 Ollama (Local):$(RESET)"
	@echo "   - Install: https://ollama.com/download"
	@echo "   - Start service: ollama serve"
	@echo "   - Pull models: ollama pull qwen3:4b && ollama pull nomic-embed-text"
	@echo ""
	@echo "$(CYAN)   🤖 OpenAI (API):$(RESET)"
	@echo "   - Get API key from: https://platform.openai.com/api-keys"
	@echo "   - Add to .env: OPENAI_API_KEY=your_key_here"
	@echo ""
	@echo "$(MAGENTA)2. Setup PostgreSQL with pgvector:$(RESET)"
	@echo "   - Install PostgreSQL 12+"
	@echo "   - Install pgvector extension"
	@echo "   - Create database and enable pgvector"
	@echo ""
	@echo "$(MAGENTA)3. Configure environment (.env file):$(RESET)"
	@echo "   - Database connection settings"
	@echo "   - LLM provider settings (API keys, endpoints)"
	@echo "   - Example: OPENAI_API_KEY=sk-... or OLLAMA_BASE_URL=http://localhost:11434"
	@echo ""
	@echo "$(GREEN)✅ RAG setup complete!$(RESET)"
	@echo "$(CYAN)Next step: make test-rag$(RESET)"

setup-graph-rag: ## Setup Graph RAG package with all dependencies using UV
	@echo "$(BLUE)🕸️ Setting up Graph RAG package...$(RESET)"
	@echo "$(BLUE)Installing workspace with Graph RAG features...$(RESET)"
	$(UV) sync --group rag
	@echo "$(GREEN)✅ Graph RAG dependencies installed!$(RESET)"
	@echo ""
	@echo "$(MAGENTA)📋 Neo4j Setup Required:$(RESET)"
	@echo "1. Run local Neo4j container:"
	@echo "   docker run -d --name neo4j-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest"
	@echo "2. Configure environment (.env file):"
	@echo "   NEO4J_URI=bolt://localhost:7687"
	@echo "   NEO4J_USERNAME=neo4j"
	@echo "   NEO4J_PASSWORD=password"
	@echo ""
	@echo "$(GREEN)✅ Graph RAG setup complete!$(RESET)"
	@echo "$(CYAN)Next step: make run-graph-rag-demo$(RESET)"

setup-all: ## Setup MCP, RAG and Graph RAG packages with all dependencies
	@echo "$(BLUE)🚀 Setting up complete LLM Utils workspace...$(RESET)"
	@echo "$(BLUE)============================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Installing all dependencies (MCP + RAG + Graph RAG)...$(RESET)"
	$(UV) sync --group mcp-full --group rag
	@echo ""
	@echo "$(GREEN)✅ All dependencies installed successfully!$(RESET)"
	@echo ""
	@echo "$(MAGENTA)📋 Additional Setup Required:$(RESET)"
	@echo "$(MAGENTA)1. Choose and setup your LLM provider(s):$(RESET)"
	@echo ""
	@echo "$(CYAN)   📡 Ollama (Local - Recommended):$(RESET)"
	@echo "   - Install: https://ollama.com/download"
	@echo "   - Start service: ollama serve"
	@echo "   - Pull models: ollama pull qwen3:4b && ollama pull nomic-embed-text"
	@echo ""
	@echo "$(CYAN)   🤖 OpenAI (API):$(RESET)"
	@echo "   - Get API key from: https://platform.openai.com/api-keys"
	@echo "   - Add to .env: OPENAI_API_KEY=your_key_here"
	@echo ""
	@echo "$(MAGENTA)2. Setup Vector Database (choose one):$(RESET)"
	@echo ""
	@echo "$(CYAN)   🐘 PostgreSQL with pgvector (Default):$(RESET)"
	@echo "   - Install PostgreSQL 12+"
	@echo "   - Install pgvector extension"
	@echo "   - Create database and enable pgvector"
	@echo ""
	@echo "$(CYAN)   🗃️  Qdrant (Alternative):$(RESET)"
	@echo "   - Install: docker run -p 6333:6333 qdrant/qdrant"
	@echo "   - Or download from: https://qdrant.tech/documentation/guides/installation/"
	@echo ""
	@echo "$(MAGENTA)3. Available demos:$(RESET)"
	@echo "   - make run-fastmcp        # Basic MCP calculator server"
	@echo "   - make mcp-io-demo        # MCP server with file operations"
	@echo "   - make rag-mcp-demo       # MCP server with RAG tools"
	@echo "   - make test-rag           # Interactive RAG testing"
	@echo ""
	@echo "$(GREEN)🎉 Setup complete! Choose a demo to get started.$(RESET)"

run-fastmcp: ## Create and run FastMCP server
	@echo "$(BLUE)🚀 Starting FastMCP Calculator Server$(RESET)"
	@echo "$(BLUE)=====================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Starting server on http://localhost:8000$(RESET)"
	@echo "$(CYAN)Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/run_fastmcp.py

simple-fastmcp: ## Start simple FastMCP server with basic tools
	@echo "$(BLUE)🚀 Starting Simple FastMCP Server$(RESET)"
	@echo "$(BLUE)=================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Starting simple server on http://localhost:8000$(RESET)"
	@echo "$(CYAN)Available tools: add_numbers, multiply_numbers, get_weather, reverse_text, count_words$(RESET)"
	@echo "$(CYAN)Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/simple_fastmcp_server.py

cvt-fastmcp-2-openapi: ## Convert MCP server to OpenAPI using MCPO
	@echo "$(MAGENTA)🔄 Converting MCP to OpenAPI with MCPO$(RESET)"
	@echo "$(MAGENTA)=====================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Enter the MCP server URL (e.g., http://localhost:8002/mcp):$(RESET)"
	@read -p "MCP Server URL: " MCP_URL; \
	if [ -z "$$MCP_URL" ]; then \
		echo "$(MAGENTA)No URL provided. Using default: http://localhost:8002/mcp$(RESET)"; \
		MCP_URL="http://localhost:8002/mcp"; \
	fi; \
	echo "$(CYAN)Starting OpenAPI server on http://localhost:8001$(RESET)"; \
	echo "$(CYAN)API Documentation: http://localhost:8001/docs$(RESET)"; \
	echo "$(CYAN)API Key: demo-key$(RESET)"; \
	echo "$(CYAN)Connecting to MCP server: $$MCP_URL$(RESET)"; \
	echo "$(CYAN)Server will run for 15 seconds then auto-stop$(RESET)"; \
	echo ""; \
	uvx mcpo --port 8001 \
	--server-type "streamable-http" -- "$$MCP_URL"
#   --api-key "demo-key" \#
	SERVER_PID=$$!; \
	sleep 3; \
	echo "$(GREEN)✓ OpenAPI server started successfully$(RESET)"; \
	echo "$(GREEN)Testing API endpoints...$(RESET)"; \
	curl -s "http://localhost:8001/docs" > /dev/null && echo "$(GREEN)✓ Swagger UI available at http://localhost:8001/docs$(RESET)" || echo "$(YELLOW)⚠ Swagger UI not yet ready$(RESET)"; \
	curl -s -H "Authorization: Bearer demo-key" "http://localhost:8001/" > /dev/null && echo "$(GREEN)✓ API root endpoint responding$(RESET)" || echo "$(YELLOW)⚠ API not yet ready$(RESET)"; \
	echo "$(CYAN)Waiting for server to auto-stop...$(RESET)"; \
	wait $$SERVER_PID; \
	echo "$(GREEN)🎉 OpenAPI conversion test completed!$(RESET)"


mcp-io-demo: ## Start FastMCP server with MCP-IO tools
	@echo "$(BLUE)🚀 Starting FastMCP Server with MCP-IO Tools$(RESET)"
	@echo "$(BLUE)=============================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Starting FastMCP server on http://localhost:8002$(RESET)"
	@echo "$(CYAN)Available tools: All 20 MCP-IO tools (file, directory, path, temporary operations)$(RESET)"
	@echo "$(CYAN)Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/demo_mcp_io_server.py

rag-mcp-demo: ## Start FastMCP server with RAG tools for document indexing and querying
	@echo "$(BLUE)🧠 Starting FastMCP Server with RAG Tools$(RESET)"
	@echo "$(BLUE)=======================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Starting FastMCP server on http://localhost:8003$(RESET)"
	@echo "$(CYAN)Available tools: RAG document indexing and querying$(RESET)"
	@echo "$(CYAN)Prerequisites: Ollama (localhost:11434) and PostgreSQL with pgvector$(RESET)"
	@echo "$(CYAN)Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/demo_rag_mcp_server.py

rag-indexing-api: ## Start RAG document indexing API for HTTP-based document indexing
	@echo "$(BLUE)📚 Starting RAG Document Indexing API$(RESET)"
	@echo "$(BLUE)=====================================$(RESET)"
	@echo ""
	@echo "$(CYAN)Starting FastAPI server on http://localhost:9999$(RESET)"
	@echo "$(CYAN)Available endpoints: POST /index-document, GET /health$(RESET)"
	@echo "$(CYAN)API Documentation: http://localhost:9999/docs$(RESET)"
	@echo "$(CYAN)Prerequisites: Ollama (localhost:11434) and vector store (Qdrant/PostgreSQL)$(RESET)"
	@echo "$(CYAN)Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/rag_indexing_api.py

test-rag: ## Interactive RAG feature demo with sync/async execution and index/query operations
	@echo "$(BLUE)🧠 Starting RAG Feature Interactive Demo$(RESET)"
	@echo "$(BLUE)=====================================$(RESET)"
	@echo ""
	@echo "$(CYAN)This demo showcases:$(RESET)"
	@echo "$(CYAN)✅ Both sync and async execution modes$(RESET)"
	@echo "$(CYAN)✅ Document indexing capabilities$(RESET)"
	@echo "$(CYAN)✅ Intelligent querying with LangChain$(RESET)"
	@echo "$(CYAN)✅ Factory-based component creation$(RESET)"
	@echo ""
	@echo "$(MAGENTA)Requirements:$(RESET)"
	@echo "$(MAGENTA)- Ollama running (ollama serve)$(RESET)"
	@echo "$(MAGENTA)- Models: ollama pull llama3:8b && ollama pull nomic-embed-text$(RESET)"
	@echo "$(MAGENTA)- Qdrant vector store accessible$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/demo_rag_interactive.py

run-graph-rag-demo: ## Interactive Neo4j Graph RAG demo
	@echo "$(BLUE)🕸️ Starting Graph RAG Feature Interactive Demo$(RESET)"
	@echo "$(BLUE)=============================================$(RESET)"
	@echo ""
	@echo "$(CYAN)This demo showcases:$(RESET)"
	@echo "$(CYAN)✅ Extraction of entity-relationships into Neo4j graph$(RESET)"
	@echo "$(CYAN)✅ Self-correcting Cypher generation with LangGraph$(RESET)"
	@echo "$(CYAN)✅ Semantic entity search and neighborhood traversal$(RESET)"
	@echo ""
	@echo "$(MAGENTA)Requirements:$(RESET)"
	@echo "$(MAGENTA)- Running Neo4j instance (make setup-graph-rag for docker command)$(RESET)"
	@echo "$(MAGENTA)- Ollama running (llama3 and nomic-embed-text)$(RESET)"
	@echo "$(MAGENTA)- Environmental vars: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/demo_graph_rag.py

langchain-mcp-demo: ## Run LangChain agent with MCP client integration
	@echo "$(BLUE)🦜 Starting LangChain Agent with MCP Client$(RESET)"
	@echo "$(BLUE)===========================================$(RESET)"
	@echo ""
	@echo "$(CYAN)This demo showcases:$(RESET)"
	@echo "$(CYAN)✅ LangChain integration with MCP servers$(RESET)"
	@echo "$(CYAN)✅ Official langchain-mcp-adapters package$(RESET)"
	@echo "$(CYAN)✅ OpenAI-powered conversational agent$(RESET)"
	@echo "$(CYAN)✅ Interactive chat with MCP tools$(RESET)"
	@echo ""
	@echo "$(MAGENTA)Prerequisites:$(RESET)"
	@echo "$(MAGENTA)- MCP RAG server running: make rag-mcp-demo$(RESET)"
	@echo "$(MAGENTA)- OpenAI API key in .env file$(RESET)"
	@echo "$(MAGENTA)- langchain packages installed$(RESET)"
	@echo ""
	@$(UV) run python $(EXAMPLE_DIR)/langchain_mcp_client.py