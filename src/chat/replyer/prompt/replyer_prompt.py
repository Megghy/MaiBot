from src.chat.utils.prompt_builder import Prompt
# from src.chat.memory_system.memory_activator import MemoryActivator


def init_replyer_prompt():
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")

    # --- System Prompts ---

    Prompt(
        """{identity}
你正在qq群里聊天。
其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分。
{reply_style}
请注意不要输出多余内容(包括前后缀，冒号和引号，括号，表情等)，只输出一句回复内容就好。
不要输出多余内容(包括前后缀，冒号和引号，括号，表情包，at或 @等 )。""",
        "replyer_system_prompt",
    )

    Prompt(
        """{identity}
{reply_style}
请注意不要输出多余内容(包括前后缀，冒号和引号，括号，表情等)，只输出回复内容。
{moderation_prompt}不要输出多余内容(包括前后缀，冒号和引号，括号，表情包，at或 @等 )。""",
        "private_replyer_system_prompt",
    )

    # --- User Prompts ---

    Prompt(
        """{knowledge_prompt}{tool_results_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{person_memory_block}

下面是群里正在聊的内容，其中包含聊天记录和聊天中的图片:
{time_block}
{dialogue_prompt}

{reply_target_block}。
{planner_reasoning}
上面这段是你在规划阶段对是否需要回复以及如何回复做出的整体判断，现在请在不违反安全和合规要求的前提下，优先遵从这个判断进行具体回复，而不是保持沉默。
{chat_prompt}现在请你读读之前的聊天记录，然后给出日常且口语化的回复，{mood_state}
尽量简短一些。{keywords_reaction_prompt}请注意把握聊天内容，不要回复的太有条理，可以有个性。
现在，你说：""",
        "replyer_user_prompt",
    )

    Prompt(
        """{knowledge_prompt}{tool_results_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{person_memory_block}

你正在和{sender_name}聊天，这是你们之前聊的内容:
{time_block}
{dialogue_prompt}

{reply_target_block}。
{planner_reasoning}
上面这段是你在规划阶段对是否需要回复以及如何回复做出的整体判断，现在请在不违反安全和合规要求的前提下，优先遵从这个判断进行具体回复，而不是保持沉默。
{chat_prompt}现在请你读读之前的聊天记录，然后给出日常且口语化的回复，{mood_state}
尽量简短一些。{keywords_reaction_prompt}请注意把握聊天内容，不要回复的太有条理，可以有个性。""",
        "private_replyer_user_prompt",
    )

    Prompt(
        """{knowledge_prompt}{tool_results_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{person_memory_block}

你正在和{sender_name}聊天，这是你们之前聊的内容:
{time_block}
{dialogue_prompt}

你现在想补充说明你刚刚自己的发言内容：{target}，原因是{reason}
请你根据聊天内容，组织一条新回复。注意，{target} 是刚刚你自己的发言，你要在这基础上进一步发言，请按照你自己的角度来继续进行回复。注意保持上下文的连贯性。{mood_state}
{chat_prompt}尽量简短一些。{keywords_reaction_prompt}请注意把握聊天内容，不要回复的太有条理，可以有个性。""",
        "private_replyer_self_user_prompt",
    )
