from google.adk.evaluation.agent_evaluator import AgentEvaluator
import pytest

@pytest.mark.asyncio
async def test_with_single_test_file():
    """Test the agent's ability to create a complete brand audit."""
    await AgentEvaluator.evaluate(
        agent_module="audit_brand_agent.agent",
        eval_dataset_file_path_or_dir="audit_brand_agent/test_config.test.json",
    )