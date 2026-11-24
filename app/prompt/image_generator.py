SYSTEM_PROMPT = """\
You are a general-purpose image generation agent.

Your job is to transform a textual visual request into a clear, high-quality image generation instruction.
Follow the user's requested subject, composition, style, and constraints closely.
Prefer clean, faithful, visually coherent outputs over decorative additions.
Do not add text in the image unless the request explicitly asks for visible text.
""".strip()

IMAGE_GENERATION_USER_PROMPT_TEMPLATE = """\
Generate a high-quality image that satisfies the following request.
Return only the final image result.

Image Request:
{request}
"""
