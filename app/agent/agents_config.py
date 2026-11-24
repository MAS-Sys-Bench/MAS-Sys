# Agent metadata configuration

AGENT_METADATA = {
    'manus': {
        'class_name': 'Manus',
        'module': 'app.agent.manus',
        'description': 'General-purpose agent with local tool support for various tasks',
        'available_tools': ['python_execute', 'browser_use', 'str_replace_editor'],
        'llm_type': 'vision',
    },
    'browser': {
        'class_name': 'BrowserAgent',
        'module': 'app.agent.browser',
        'description': 'Browser automation specialist for web interaction and scraping',
        'available_tools': ['browser_use'],
        'llm_type': 'vision',
    },
    'image_generator': {
        'class_name': 'ImageGeneratorAgent',
        'module': 'app.agent.image_generator',
        'description': (
            'General-purpose text-to-image generator. Takes a visual request in text '
            'and produces a final image using an image-capable model.'
        ),
        'available_tools': [],
        'llm_type': 'image',
    },
    # SWE Agents
    'swe': {
        'class_name': 'SWEAgent',
        'module': 'app.agent.swe.general',
        'description': 'Software engineering specialist for code editing and debugging',
        'available_tools': ['bash', 'str_replace_editor'],
        'llm_type': 'lang',
    },
    'swe_solver': {
        'class_name': 'SWESolverAgent',
        'module': 'app.agent.swe.solver',
        'description': 'SWE-specific repair agent for iterative repository patching',
        'available_tools': ['bash', 'str_replace_editor'],
        'llm_type': 'lang',
    },
    'swe_verifier': {
        'class_name': 'SWEVerifierAgent',
        'module': 'app.agent.swe.verifier',
        'description': (
            'SWE-specific verification agent that runs validation commands '
            'and returns structured repair feedback. It may use bash for inspection '
            'and tests; it must not edit files.'
        ),
        'available_tools': ['bash'],
        'llm_type': 'lang',
    },
    # Research Agents
    'research_proposal_writer': {
        'class_name': 'ProposalWriterAgent',
        'module': 'app.agent.proposal.proposal_writer',
        'description': (
            'Aligned proposal writer. Each instance should correspond to one author. '
            'PlanAgent should provide per-node agent_params.author_name, '
            'agent_params.affiliation, and agent_params.publications. '
            'The agent then generates a persona from those publications and writes one proposal candidate '
            'from that persona plus shared related-paper context.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'research_proposal_aggregator': {
        'class_name': 'ProposalAggregatorAgent',
        'module': 'app.agent.proposal.proposal_aggregator',
        'description': (
            'Aligned final proposal aggregator. Consumes all proposal candidates and '
            'summarizes them into one final five-question proposal.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'research_review_writer': {
        'class_name': 'ReviewWriterAgent',
        'module': 'app.agent.review.review_writer',
        'description': (
            'Aligned review writer. Each instance should correspond to one reviewer. '
            'PlanAgent should provide per-node agent_params.reviewer_name, '
            'agent_params.affiliation, and agent_params.publications. '
            'It first generates a persona from those publications, '
            'then writes strength, weakness, and score for the shared submission.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'research_metareview_chair': {
        'class_name': 'MetaReviewChairAgent',
        'module': 'app.agent.review.metareview_chair',
        'description': (
            'ResearchTown-style metareview chair. Consumes reviewer outputs '
            'and writes a metareview summary with a simple accept/reject '
            'decision.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    # Paper Visualization Agents
    'paperviz_planner': {
        'class_name': 'PaperVizPlannerAgent',
        'module': 'app.agent.paperviz.planner',
        'description': (
            'Academic illustration planner. Generates a detailed textual description '
            'of the target figure using runtime RAG over the local PaperBanana '
            'reference pool, then injects the retrieved multimodal few-shot '
            'examples directly into the planner prompt before execution.'
        ),
        'available_tools': [],
        'llm_type': 'vision',
    },
    'paperviz_stylist': {
        'class_name': 'PaperVizStylistAgent',
        'module': 'app.agent.paperviz.stylist',
        'description': (
            'Academic illustration stylist. Refines a figure description with '
            'publication-quality aesthetics following NeurIPS 2025 style guidelines. '
            'Use after the planner in an illustration pipeline.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'paperviz_plot_visualizer': {
        'class_name': 'PaperVizPlotVisualizerAgent',
        'module': 'app.agent.paperviz.plot_visualizer',
        'description': (
            'Academic plot renderer. Generates plot code or rendering instructions '
            'from a detailed plot description. Use after the stylist in a plot pipeline.'
        ),
        'available_tools': ['paperviz_execute_plot_code'],
        'llm_type': 'lang',
    },
    'paperviz_diagram_visualizer': {
        'class_name': 'PaperVizDiagramVisualizerAgent',
        'module': 'app.agent.paperviz.diagram_visualizer',
        'description': (
            'Academic diagram renderer. Uses an image-generation-capable model to '
            'produce a final scientific diagram image from a detailed diagram description. Use after the stylist in a diagram pipeline. '
            'And it is suggested to be used together with the paperviz_critic agent in a closed loop for iterative refinement until the critic signals finish.'
        ),
        'available_tools': [],
        'llm_type': 'image',
    },
    'paperviz_critic': {
        'class_name': 'PaperVizCriticAgent',
        'module': 'app.agent.paperviz.critic',
        'description': (
            'Academic illustration critic. Evaluates a generated figure against its '
            'source content and provides structured revision suggestions. Forms a '
            'closed loop with the visualizer for iterative refinement. When used as '
            'an FSM decision node, its output decision contract is: "continue" to '
            'request another revision round by routing back to the visualizer, or '
            '"finish" to terminate.'
        ),
        'available_tools': [],
        'llm_type': 'vision',
    },
    # Fund Agents
    'fund_technical': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'technical',
        'description': (
            'Technical analyst. Evaluates stock price trends, momentum, RSI, '
            'volatility, volume patterns, and support/resistance levels from '
            'historical OHLCV data to generate a Bullish/Bearish/Neutral signal. '
            'The agent should not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_fundamental': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'fundamental',
        'description': (
            'Fundamental analyst. Evaluates company financial health using PE ratio, '
            'EPS, ROE, profit margins, debt levels, and growth metrics from quarterly '
            'reports to generate a Bullish/Bearish/Neutral signal. The agent should '
            'not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_insider': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'insider',
        'description': (
            'Insider trading analyst. Analyzes recent insider buy/sell transactions '
            'by executives and directors to detect institutional sentiment and '
            'generate a Bullish/Bearish/Neutral signal. The agent should not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_news': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'news',
        'description': (
            'Company news analyst. Evaluates recent news articles about a company '
            'to assess sentiment and likely impact on stock price, generating a '
            'Bullish/Bearish/Neutral signal. The agent should not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_macro': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'macro',
        'description': (
            'Macroeconomic analyst. Evaluates GDP, CPI, unemployment, interest '
            'rates, and treasury yields to assess broad economic conditions and '
            'generate a Bullish/Bearish/Neutral signal. The agent should not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_policy': {
        'class_name': 'FinancialAnalystAgent',
        'module': 'app.agent.fund.analyst',
        'role': 'policy',
        'description': (
            'Policy analyst. Evaluates fiscal and monetary policy news to assess '
            'their short-term economic impact and generate a Bullish/Bearish/Neutral '
            'signal. The agent should not call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
    'fund_manager': {
        'class_name': 'FundManagerAgent',
        'module': 'app.agent.fund.manager',
        'description': (
            'Portfolio manager. Performs internal risk analysis first, then makes final '
            'Buy/Sell/Hold trading decisions with share quantities. '
            'Manages portfolio positions across multiple tickers. The agent should not '
            'call tools.'
        ),
        'available_tools': [],
        'llm_type': 'lang',
    },
}


def build_agents_info_for_prompt() -> str:
    """Format AGENT_METADATA into a human-readable block for LLM system prompts.

    Returns one line per agent type showing its key, description, available tools,
    and llm_type (lang/vision/image).
    This is the single source-of-truth formatter consumed by PlanAgent.
    """
    lines = []
    for key, meta in AGENT_METADATA.items():
        lines.append(
            f"- {key}: {meta['description']}\n"
            f"  tools: {', '.join(meta.get('available_tools', []))}\n"
            f"  llm_type: {meta.get('llm_type', 'lang')}"
        )
    return "\n".join(lines)
