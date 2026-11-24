import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(..., description="Azure, Openai, or Ollama")
    api_version: str = Field(..., description="Azure Openai version if AzureOpenai")
    api_style: str = Field("", description="Image API style: 'openai' for OpenAI/vLLM-compatible images.generate(), 'dashscope' for DashScope MultiModalConversation-style HTTP API")
    # Candidate metadata (used by PlanAgent to choose LLMs for agents)
    size: str = Field("", description="Model parameter size (e.g. '14B', '27B')")
    capabilities: str = Field("", description="Model capabilities description")
    cost: str = Field("", description="Deployment cost hint (e.g. 'Low — single GPU')")
    # Serving / deployment fields (only entries with non-empty backend get deployed)
    backend: str = Field(
        "",
        description="Inference backend: 'vllm', 'continuum', 'sglang', or 'vllm-omni'",
    )
    model_path: str = Field("", description="HuggingFace model path for serving")
    port: int = Field(0, description="Port number for the inference server")
    gpus: str = Field("", description="Comma-separated GPU IDs (e.g. '0,1')")
    lmcache_cpu_gb: float = Field(
        0.0,
        description="LMCache CPU offload size in GB; currently used by backend='continuum'",
    )


class OrlaCostModelSettings(BaseModel):
    input_cost_per_mtoken: float = Field(
        0.0, description="Estimated input token cost in USD per million tokens"
    )
    output_cost_per_mtoken: float = Field(
        0.0, description="Estimated output token cost in USD per million tokens"
    )


class OrlaBackendSettings(BaseModel):
    endpoint: str = Field(..., description="Backend endpoint registered into Orla")
    type: str = Field(
        "openai",
        description="Backend provider type (e.g. 'openai' or 'sglang')",
    )
    model_id: str = Field(..., description="Backend model identifier")
    api_key_env_var: str = Field(
        "", description="Environment variable containing the backend API key"
    )
    max_concurrency: int = Field(
        1, description="Maximum concurrent requests Orla should send"
    )
    queue_capacity: int = Field(
        0, description="Optional queue capacity override for the backend"
    )
    quality: Optional[float] = Field(
        None, description="Optional relative quality score for the backend"
    )
    cost_model: Optional[OrlaCostModelSettings] = Field(
        None, description="Optional cost model for Orla routing metadata"
    )


class OrlaSettings(BaseModel):
    url: str = Field("http://localhost:8081", description="Orla daemon base URL")
    timeout: float = Field(300.0, description="HTTP timeout for Orla requests")
    backends: Dict[str, OrlaBackendSettings] = Field(
        default_factory=dict,
        description="Registered backend definitions keyed by llm_config_name",
    )


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(
        default="en",
        description="Language code for search results (e.g., en, zh, fr)",
    )
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., us, cn, uk)",
    )


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )


class SandboxSettings(BaseModel):
    """Configuration for the execution sandbox"""

    use_sandbox: bool = Field(False, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image")
    work_dir: str = Field("/workspace", description="Container working directory")
    memory_limit: str = Field("512m", description="Memory limit")
    cpu_limit: float = Field(1.0, description="CPU limit")
    timeout: int = Field(300, description="Default command timeout (seconds)")
    network_enabled: bool = Field(
        False, description="Whether network access is allowed"
    )


class MemorySettings(BaseModel):
    """Configuration for LongTermMemory embedding model."""

    model: str = Field("text-embedding-3-small", description="Embedding model name")
    base_url: str = Field("", description="API endpoint URL")
    api_key: str = Field("", description="API key")
    dimensions: Optional[int] = Field(None, description="Embedding dimensions")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    orla_config: Optional[OrlaSettings] = Field(
        None, description="Orla runtime configuration"
    )
    sandbox: Optional[SandboxSettings] = Field(
        None, description="Sandbox configuration"
    )
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )
    search_config: Optional[SearchSettings] = Field(
        None, description="Search configuration"
    )
    memory_config: Optional[MemorySettings] = Field(
        None, description="Long-term memory embedding configuration"
    )

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._workspace_root = WORKSPACE_ROOT
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # Flatten nested candidate groups: [llm.lang.xxx] / [llm.image.xxx]
        # become "lang.xxx" / "image.xxx" keys in the final llm dict.
        # Direct group-level settings like [llm.vision] are preserved as "vision".
        _CANDIDATE_GROUPS = ("lang", "image", "vision")
        flattened_overrides: Dict[str, dict] = {}
        for name, value in llm_overrides.items():
            if name in _CANDIDATE_GROUPS:
                direct_group_settings = {
                    sub_name: sub_config
                    for sub_name, sub_config in value.items()
                    if not isinstance(sub_config, dict)
                }
                if direct_group_settings:
                    flattened_overrides[name] = direct_group_settings

                # value may also contain {sub_name: {settings_dict}} — flatten those
                for sub_name, sub_config in value.items():
                    if isinstance(sub_config, dict):
                        flattened_overrides[f"{name}.{sub_name}"] = sub_config
            else:
                flattened_overrides[name] = value
        llm_overrides = flattened_overrides

        # Validate: at least one lang candidate must exist
        lang_keys = [k for k in llm_overrides if k.startswith("lang.")]
        if not lang_keys:
            raise ValueError(
                "No language model candidates found in config. "
                "Define at least one [llm.lang.<name>] section in config.toml."
            )

        # handle browser config.
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # handle proxy settings.
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # filter valid browser config parameters.
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # if there is proxy settings, add it to the parameters.
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # only create BrowserSettings when there are valid parameters.
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()
        memory_config = raw_config.get("memory", {})
        memory_settings = MemorySettings(**memory_config) if memory_config else None

        def _flatten_named_sections(raw: dict, prefix: str = "") -> Dict[str, dict]:
            flattened: Dict[str, dict] = {}
            for key, value in raw.items():
                full_key = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict) and value and not any(
                    k in value
                    for k in (
                        "endpoint",
                        "type",
                        "model_id",
                        "api_key_env_var",
                        "max_concurrency",
                        "queue_capacity",
                        "quality",
                        "cost_model",
                    )
                ):
                    flattened.update(_flatten_named_sections(value, full_key))
                elif isinstance(value, dict):
                    flattened[full_key] = value
            return flattened

        orla_config = raw_config.get("orla", {})
        orla_settings = None
        if orla_config:
            raw_backends = orla_config.get("backends", {})
            flattened_backends = _flatten_named_sections(raw_backends)
            orla_settings = OrlaSettings(
                **{
                    **{k: v for k, v in orla_config.items() if k != "backends"},
                    "backends": flattened_backends,
                }
            )

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "orla_config": orla_settings,
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "memory_config": memory_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    def require_llm_config(self, config_name: str) -> LLMSettings:
        """Return a configured LLM entry or raise a clear error."""
        llm_cfg = self._config.llm.get(config_name)
        if llm_cfg is None:
            available = ", ".join(sorted(self._config.llm.keys()))
            raise KeyError(
                f"LLM config '{config_name}' is not defined in config.toml. "
                f"Available keys: {available}"
            )
        return llm_cfg

    @property
    def orla(self) -> Optional[OrlaSettings]:
        return self._config.orla_config

    @property
    def sandbox(self) -> SandboxSettings:
        return self._config.sandbox

    @property
    def memory(self) -> Optional[MemorySettings]:
        """Get the long-term memory embedding configuration"""
        return self._config.memory_config

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._config.search_config

    @property
    def workspace_root(self) -> Path:
        """Get the workspace root directory"""
        return self._workspace_root

    def set_workspace_root(self, workspace_root: Path, create: bool = True) -> None:
        """Set workspace root directory for current process."""
        self._workspace_root = Path(workspace_root)
        if create:
            self._workspace_root.mkdir(parents=True, exist_ok=True)

    def set_workspace_by_run_id(self, run_id: str) -> Path:
        """Set workspace to an isolated run directory under workspace root."""
        run_workspace = WORKSPACE_ROOT / run_id
        self.set_workspace_root(run_workspace)
        return run_workspace

    @property
    def root_path(self) -> Path:
        """Get the root path of the application"""
        return PROJECT_ROOT


config = Config()
