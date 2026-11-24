ONLY_AGENT_DESCRIPTION = "You are the only agent in this MAS."

MAS_SYSTEM_PREFIX_TEMPLATE = """\
[MAS Context]
{mas_context}

[Your Role]
{role_description}

[Peer Agents]
{peers_description}

[Message Routing]
  Receives from: {routes_in}
  Sends to:      {routes_out}

[Shared Memory] {memory_permission}"""

LLM_MERGE_PROMPT_TEMPLATE = """\
The following outputs were produced by multiple agents. Merge them into a single coherent response:

{numbered_inputs}"""
