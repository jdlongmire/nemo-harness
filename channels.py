"""Named channel logging for 4M cross-module data flows.

Channels from the 4M cross-module interface diagram:
  Ch1: Memory -> Mind   (mid-reasoning recall via semantic_search)
  Ch2: Mind -> Memory   (persistence via action tags + running summary)
  Ch3: Morals -> Mind   (constraint enforcement, output validation)
  Ch4: Mission -> Mind  (intent-driven tool/guide selection)

Each channel is a named Python logger under 'nemo.channel.*',
producing structured log lines that are grep-able and observable.
"""

from __future__ import annotations

import logging

# Parent logger for all channels
_parent = logging.getLogger('nemo.channel')


def _get_channel(name: str) -> logging.Logger:
    return logging.getLogger(f'nemo.channel.{name}')


# Ch1: Memory -> Mind
ch1_memory_to_mind = _get_channel('ch1_memory_mind')

# Ch2: Mind -> Memory
ch2_mind_to_memory = _get_channel('ch2_mind_memory')

# Ch3: Morals -> Mind
ch3_morals_to_mind = _get_channel('ch3_morals_mind')

# Ch4: Mission -> Mind
ch4_mission_to_mind = _get_channel('ch4_mission_mind')


def log_memory_recall(cid: str, query: str, result_count: int):
    """Ch1: Memory recalled context for Mind's reasoning."""
    ch1_memory_to_mind.info('[%s] Ch1:Memory->Mind recall query=%r results=%d', cid, query[:80], result_count)


def log_memory_inject(cid: str, block_size: int):
    """Ch1: Memory context block injected into system prompt."""
    ch1_memory_to_mind.info('[%s] Ch1:Memory->Mind context_inject block_chars=%d', cid, block_size)


def log_memory_persist(cid: str, entry_type: str, name: str):
    """Ch2: Mind persisted data to Memory via action tag."""
    ch2_mind_to_memory.info('[%s] Ch2:Mind->Memory persist type=%s name=%r', cid, entry_type, name)


def log_summary_persist(cid: str, summary_len: int):
    """Ch2: Mind built running summary from evicted context."""
    ch2_mind_to_memory.info('[%s] Ch2:Mind->Memory running_summary chars=%d', cid, summary_len)


def log_morals_violation(cid: str, check: str, detail: str):
    """Ch3: Morals flagged a violation for Mind/output."""
    ch3_morals_to_mind.warning('[%s] Ch3:Morals->Mind violation check=%s detail=%s', cid, check, detail)


def log_morals_pass(cid: str, check: str):
    """Ch3: Morals check passed."""
    ch3_morals_to_mind.debug('[%s] Ch3:Morals->Mind pass check=%s', cid, check)


def log_intent_selection(cid: str, detected: str, effective: str, guides: list[str], tool_count: str):
    """Ch4: Mission's intent classifier selected guides/tools for Mind."""
    ch4_mission_to_mind.info(
        '[%s] Ch4:Mission->Mind intent detected=%s effective=%s guides=%s tools=%s',
        cid, detected, effective, guides, tool_count,
    )


def log_coherence_check(cid: str, intent: str, mode: str, coherent: bool, reason: str = ''):
    """Ch4: Mission coherence validation result."""
    if coherent:
        ch4_mission_to_mind.info('[%s] Ch4:Mission->Mind coherence_ok intent=%s mode=%s', cid, intent, mode)
    else:
        ch4_mission_to_mind.warning(
            '[%s] Ch4:Mission->Mind coherence_fail intent=%s mode=%s reason=%s',
            cid, intent, mode, reason,
        )
