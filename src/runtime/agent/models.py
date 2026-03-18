from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from src.llm_models.payload_content.tool_option import ToolParamType


class AgentMode(str, Enum):
    SINGLE_PASS = "single_pass"
    TOOL_LOOP = "tool_loop"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULT = "no_result"
    ERROR = "error"
    TIMEOUT = "timeout"


class PromptBlockRole(str, Enum):
    SYSTEM = "system"
    USER = "user"


class OutputMode(str, Enum):
    REPLY_TEXT = "reply_text"
    ACTION_SELECTION = "action_selection"
    QA_ANSWER = "qa_answer"
    TOOL_ONLY = "tool_only"


class MemoryPolicy(str, Enum):
    OFF = "off"
    LIGHT = "light"
    FULL = "full"


class McpPolicy(str, Enum):
    OFF = "off"
    ALLOW_LIST = "allow_list"
    AUTO = "auto"


@dataclass(slots=True)
class PromptBlock:
    name: str
    content: str
    role: PromptBlockRole = PromptBlockRole.USER
    order: int = 0


@dataclass(slots=True)
class ToolParamSpec:
    name: str
    param_type: ToolParamType
    description: str
    required: bool
    enum_values: list[str] | None = None


@dataclass(slots=True)
class ToolExecutionContext:
    chat_id: str
    chat_stream: Any | None = None
    session: Any | None = None
    plugin_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    content: Any = ""
    success: bool = True
    call_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallRecord:
    call_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    result: ToolExecutionResult | None = None


@dataclass(slots=True)
class SkillProfile:
    name: str
    description: str = ""
    agent_mode: AgentMode = AgentMode.SINGLE_PASS
    model_task: str = "replyer"
    enabled_context_blocks: list[str] = field(default_factory=list)
    allowed_tool_tags: set[str] = field(default_factory=set)
    blocked_tools: set[str] = field(default_factory=set)
    max_iterations: int = 1
    timeout_seconds: float = 30.0
    output_mode: OutputMode = OutputMode.REPLY_TEXT
    memory_policy: MemoryPolicy = MemoryPolicy.OFF
    mcp_policy: McpPolicy = McpPolicy.OFF
    allow_plain_text_fallback: bool = True
    terminal_tools: set[str] = field(default_factory=set)


@dataclass(slots=True)
class AgentProfile:
    name: str
    skill_name: str
    agent_mode: AgentMode
    model_task: str
    enabled_context_blocks: list[str] = field(default_factory=list)
    allowed_tool_tags: set[str] = field(default_factory=set)
    blocked_tools: set[str] = field(default_factory=set)
    max_iterations: int = 1
    timeout_seconds: float = 30.0
    output_mode: OutputMode = OutputMode.REPLY_TEXT
    memory_policy: MemoryPolicy = MemoryPolicy.OFF
    mcp_policy: McpPolicy = McpPolicy.OFF
    allow_plain_text_fallback: bool = True
    terminal_tools: set[str] = field(default_factory=set)

    @classmethod
    def from_skill(
        cls,
        skill: SkillProfile,
        *,
        name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "AgentProfile":
        data = {
            "name": name or skill.name,
            "skill_name": skill.name,
            "agent_mode": skill.agent_mode,
            "model_task": skill.model_task,
            "enabled_context_blocks": list(skill.enabled_context_blocks),
            "allowed_tool_tags": set(skill.allowed_tool_tags),
            "blocked_tools": set(skill.blocked_tools),
            "max_iterations": skill.max_iterations,
            "timeout_seconds": skill.timeout_seconds,
            "output_mode": skill.output_mode,
            "memory_policy": skill.memory_policy,
            "mcp_policy": skill.mcp_policy,
            "allow_plain_text_fallback": skill.allow_plain_text_fallback,
            "terminal_tools": set(skill.terminal_tools),
        }
        if overrides:
            data.update(overrides)
        return cls(**data)


@dataclass(slots=True)
class ComposedPrompt:
    system_prompt: str
    user_prompt: str
    ordered_blocks: list[PromptBlock] = field(default_factory=list)


@dataclass(slots=True)
class AgentRequest:
    request_id: str
    chat_id: str
    user_id: str | None
    message: str
    chat_history: str = ""
    skill_name: str = ""
    prompt_blocks: list[PromptBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


TraceEvent = dict[str, Any]


@dataclass(slots=True)
class AgentResult:
    status: AgentStatus
    output_text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    used_skill: str = ""
    used_tools: list[str] = field(default_factory=list)
    source_summary: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    reasoning: str = ""
    model_name: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    terminal_tool_name: str | None = None
    error_message: str | None = None


ToolSource = Literal["builtin", "plugin", "memory", "mcp"]
ToolFailMode = Literal["raise", "soft_error"]
