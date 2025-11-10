from agent_framework import ExecutorCompletedEvent, ExecutorInvokedEvent, ChatAgent, WorkflowBuilder

from domain.agents.chunk_agregator import ChunksAgregatorExec
from domain.agents.chunk_exporter import ChunksExporterExec
from domain.agents.secrets_detector import SecretsDetectorExec
from domain.settings import settings
from domain.utils import CustomResponseEvent
chat_client = settings.chat_client
GITHUB_TOKEN = settings.github_token
GITHUB_REPO = settings.github_repo
GITHUB_OWNER = settings.github_owner
TARGET_PR_NUMBER = settings.target_pr_number
toolsets = settings.github_toolsets


async def init_github_mcp():
    github_mcp = settings.github_mcp_server
    if github_mcp is None:
        raise RuntimeError("GitHub MCP server tool was not initialized in settings.")
    await github_mcp.connect()
    print("GitHub MCP connected")
    return github_mcp


async def run():
    github_mcp = await init_github_mcp()
    total_agents=3
    secrets_detector_1 = SecretsDetectorExec(chat_client, id="SecretsDetector1", my_shard=0, total_agents=total_agents)
    secrets_detector_2 = SecretsDetectorExec(chat_client, id="SecretsDetector2", my_shard=1, total_agents=total_agents)
    secrets_detector_3 = SecretsDetectorExec(chat_client, id="SecretsDetector3", my_shard=2, total_agents=total_agents)
    exporter = ChunksExporterExec(id="ChunkExporterAgent", github_mcp_server=settings.github_mcp_server, chat_client=chat_client)
    aggregator = ChunksAgregatorExec(id="ChunksAgregatorAgent", github_mcp_server=settings.github_mcp_server)
    builder = WorkflowBuilder()
    builder.set_start_executor(exporter)
    builder.add_fan_out_edges(exporter, [secrets_detector_1, secrets_detector_2, secrets_detector_3])
    builder.add_fan_in_edges([secrets_detector_1,
                            secrets_detector_2,
                            secrets_detector_3],
                            aggregator)

    workflow = builder.build()
    async for event in workflow.run_stream(""):
        match event:
            case CustomResponseEvent() as output:
                print(f"Workflow finished")
            case ExecutorInvokedEvent() as invoke:
                print(f"Starting {invoke.executor_id}")
            case ExecutorCompletedEvent() as complete:
                print(f"Completed {complete.executor_id}: {complete.data}")


if __name__ == "__main__":
    run()
    
