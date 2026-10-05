"""A turn running when compaction closes a lineage root must still flush.

RIC-97.  The cross-process turn lease is keyed on the conversation's *lineage
root* (the walk in ``_session_turn_lease_key_on_conn``).  ``end_reason`` is
mutable state on an ancestor, so closing a session on the chain with
``end_reason='compression'`` MOVES the key of every session below it -- while
the live turn's row still sits under the pre-close key.  The write guard then
finds no row owned by its holder and refuses the flush, and the operator is
told *"another Hermes process took over this session"* although the only
process involved built that fence itself.

Measured impact: nine consecutive Paperclip CEO runs lost their whole turn to
this, ``errorCode=adapter_failed`` / ``exitCode 1``, 56% of the company-wide
``adapter_failed`` count that day.

The closing paths must re-key the lease row inside their own transaction, so the
key is invariant for the life of a turn.

## Which conversations have a chain at all

A continuation must inherit its parent's *source* to be walked
(``_compression_child_source``, ``agent/conversation_compression.py:3251``), and
``_is_explicit_fork_child_row`` additionally short-circuits on
``source == 'tool'``:

``CHAINING_SOURCES`` (``cli``, ``telegram``): the continuation is walked, so
closing an ancestor really does move the live key -- this is the failure class
for a turn writing on a *descendant* of the closed session, and
``probe_key_move.py`` reproduces it against the real DB.

``tool`` (one-shot / ``hermes chat -q``, every Paperclip run): the lineage walk
will not follow the continuation, but closing still MOVES the closing session's
own key onto its new root, and the run that published that continuation holds
its row on exactly that id.  That is the shape measured live in run
``f789c61e`` -- the process published ``20260929_175723_08ed50``, adopted it,
and was then fenced out of its own new id; pinned by
``test_a_run_adopting_its_own_continuation_still_flushes``.
"""

from __future__ import annotations

import os

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError

CHAINING_SOURCES = ["cli", "telegram"]


def _seed_conversation(db: SessionDB, *, source: str) -> None:
    """ROOT (compression-closed) -> ANC (open) -> LIVE (open), all one conversation.

    The incident's shape: the turn resumes a session whose grandparent is already
    a closed compression segment while its immediate parent is still open, so the
    admission walk stops at the resumed session."""
    db.create_session("ROOT", source=source)
    db.append_message("ROOT", "user", "the original conversation")
    db.end_session("ROOT", "compression")
    db.create_session("ANC", source=source, parent_session_id="ROOT")
    db.append_message("ANC", "user", "first compaction continuation")
    db.create_session("LIVE", source=source, parent_session_id="ANC")
    db.append_message("LIVE", "user", "resumed turn")


def _holder(tag: str = "turn") -> str:
    return f"pid={os.getpid()}:{tag}"


def _rotate(db: SessionDB, closing: str, child: str, holder: str, source: str) -> None:
    """The real rotation path: lock, publish (close + child + handoff, one txn), unlock.

    *source* is the parent's own source, exactly as ``_compression_child_source`` passes it:
    a compression child IS the same conversation."""
    assert db.try_acquire_compression_lock(closing, holder, ttl_seconds=300.0)
    db.publish_compression_child(
        parent_session_id=closing, child_session_id=child, source=source,
        messages=[{"role": "user", "content": "compacted continuation"}],
        compression_lock_holder=holder)
    db.release_compression_lock(closing, holder)


def _lease_rows(db: SessionDB) -> list[tuple[str, str]]:
    return sorted(tuple(r) for r in db._conn.execute(
        "SELECT conversation_id, holder FROM session_turn_leases").fetchall())


@pytest.mark.parametrize("source", CHAINING_SOURCES + ["tool"])
def test_turn_that_adopts_the_published_continuation_still_flushes(tmp_path, source):
    """Shape A (run ``83eb8b11``, 298.5 s): the compaction closes the session the turn
    is running on, the process adopts the continuation it published, and the flush
    lands on the new id.

    Parametrized over ``tool`` as well as the chaining sources, because for a one-shot /
    Paperclip run the lineage walk will not follow the continuation it published
    (``_is_explicit_fork_child_row`` short-circuits on ``source == 'tool'``), so the row has to
    be carried onto the new id rather than re-keyed by the walk -- the shape measured live in run
    ``f789c61e``, where the process adopted ``20260929_175723_08ed50`` and was then fenced out of
    its own new id.
    """
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source=source)
    holder = _holder("shape-a")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    _rotate(db, "LIVE", "CHILD", holder, source)

    # The turn follows the conversation onto the continuation, exactly as the
    # adoption path does, and writes the rows it owes -- under the same holder.
    assert db.append_message("CHILD", "assistant", "turn's own flush",
                             turn_lease_holder=holder) > 0
    assert _lease_rows(db) == [("CHILD" if source == "tool" else "LIVE", holder)]


@pytest.mark.parametrize("source", CHAINING_SOURCES)
def test_turn_running_on_a_closed_ancestor_still_flushes(tmp_path, source):
    """Shape B (the eight 30-56 s runs): the compaction closes an open ANCESTOR while
    the turn keeps writing on the session it resumed.

    This is the shape the re-key exists for: closing ANC lifts the key of every session
    below it all the way to ROOT, while the live turn's row still sits at LIVE."""
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source=source)
    holder = _holder("shape-b")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    _rotate(db, "ANC", "CHILD", holder, source)

    # The key really did move, and the row moved with it: the live turn still flushes.
    assert db._session_turn_lease_key("LIVE") == "ROOT"
    assert db.append_message("LIVE", "assistant", "turn's own flush",
                             turn_lease_holder=holder) > 0
    assert _lease_rows(db) == [("ROOT", holder)]


@pytest.mark.parametrize("source", CHAINING_SOURCES)
def test_end_session_compression_closes_the_other_door(tmp_path, source):
    """``end_session(<ancestor>, 'compression')`` moves the same key as a rotation.

    It is the primitive the whole failure class funnels through, so the re-key has
    to happen there too -- not only in ``publish_compression_child``."""
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source=source)
    holder = _holder("end-session")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    db.end_session("ANC", "compression")

    assert db._session_turn_lease_key("LIVE") == "ROOT"
    assert db.append_message("LIVE", "assistant", "turn's own flush",
                             turn_lease_holder=holder) > 0
    assert _lease_rows(db) == [("ROOT", holder)]


def test_a_run_adopting_its_own_continuation_still_flushes(tmp_path):
    """The production shape, replayed from the live failing run ``f789c61e``.

    ``source='tool'`` (every ``hermes chat -q`` / Paperclip run): the lineage walk will not follow
    the continuation, but closing the resumed session drops ITS OWN key onto the new root -- and the
    run that published that continuation holds its row on exactly that id.  Measured on the real
    log: resume ``20260929_165154_f18f7f`` -> own compaction publishes
    ``20260929_175723_08ed50`` -> process adopts it -> ``append_message`` on the NEW id raises
    ``SessionTurnLeaseLostError``, and the operator is told another process took over.  No other
    process existed.

    The row must follow the conversation onto its own new terminus, so the process that published
    the continuation can write to it.
    """
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source="tool")
    holder = _holder("adopts-own-child")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    _rotate(db, "LIVE", "CHILD", holder, "tool")

    # The walk does not follow a tool continuation, so its key is still its own id -- but the
    # rotating conversation's row followed it onto the id the turn now writes on.
    assert db._session_turn_lease_key("CHILD") == "CHILD"
    assert db.append_message("CHILD", "assistant", "turn's own flush",
                             turn_lease_holder=holder) > 0
    assert _lease_rows(db) == [("CHILD", holder)]


def test_a_non_chaining_source_never_moves_a_descendants_key(tmp_path):
    """``source='tool'``: a turn writing on a *descendant* of the closed session is not walked into
    the chain, so moving the lease row must not invent one either.

    Measured on the incident's own rows: every child of the compression-closed
    ``20260929_165154_f18f7f`` is ``source='tool'`` and each resolves to itself.  Inventing a chain
    here would let a one-shot run share (or take over) its resumed origin's lease.
    """
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source="tool")
    holder = _holder("non-chaining")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    _rotate(db, "ANC", "CHILD", holder, "tool")

    assert db._session_turn_lease_key("LIVE") == "LIVE"
    assert db.append_message("LIVE", "assistant", "turn's own flush",
                             turn_lease_holder=holder) > 0
    assert ("LIVE", holder) in _lease_rows(db)


@pytest.mark.parametrize("source", CHAINING_SOURCES)
def test_closing_the_root_hands_the_conversation_to_its_continuation_not_to_nobody(tmp_path, source):
    """One conversation keeps exactly ONE lease row, at the key its own writes resolve.

    The row must move onto the new terminus rather than be dropped: dropping it would
    leave a live turn unleased, and duplicating would let a second process in."""
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source=source)
    holder = _holder("one-row")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)
    assert _lease_rows(db) == [("LIVE", holder)]

    # Publish a continuation under ANC: the conversation's new terminus is ROOT (ANC is now
    # a closed segment), which is where the live turn's own writes resolve.
    _rotate(db, "ANC", "CHILD", holder, source)

    rows = _lease_rows(db)
    assert len(rows) == 1
    assert rows[0][0] == db._session_turn_lease_key("LIVE")
    assert rows[0][0] == db._session_turn_lease_key("CHILD")


def test_rekey_does_not_let_a_second_process_share_the_conversation(tmp_path):
    """The fence still fences: moving the row must not open the conversation up."""
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source="cli")
    holder = _holder("owner")
    assert db.try_acquire_session_turn_lease("LIVE", holder, ttl_seconds=300.0)

    _rotate(db, "ANC", "CHILD", holder, "cli")

    assert not db.try_acquire_session_turn_lease("LIVE", "pid=1:turn=foreign", ttl_seconds=300.0)
    with pytest.raises(SessionTurnLeaseLostError):
        db.append_message("LIVE", "assistant", "impostor", turn_lease_holder="pid=1:turn=foreign")
    assert db._conn.execute("SELECT COUNT(*) FROM session_turn_leases").fetchone()[0] == 1


def test_rekey_never_takes_over_another_conversations_row(tmp_path):
    """A target key already held by someone else is left alone.

    The re-key may only ever move a row onto a key nothing else owns; taking over
    would hand one process's conversation to another."""
    db = SessionDB(tmp_path / "state.db")
    _seed_conversation(db, source="cli")
    mine, foreign = _holder("mine"), "pid=1:turn=foreign"
    assert db.try_acquire_session_turn_lease("LIVE", mine, ttl_seconds=300.0)

    # A foreign holder appears on the key the re-key would otherwise move onto.
    assert db.try_acquire_session_turn_lease("ROOT", foreign, ttl_seconds=300.0)

    _rotate(db, "ANC", "CHILD", mine, "cli")

    holders = {h for _, h in _lease_rows(db)}
    assert holders == {mine, foreign}
    # The owner keeps the row it had (still under its own key, not the taken one) and the
    # foreign row is untouched.
    assert ("LIVE", mine) in _lease_rows(db)
    assert ("ROOT", foreign) in _lease_rows(db)


def test_a_delegate_fork_keeps_its_own_conversation(tmp_path):
    """Explicit provenance still forks the lease domain -- the fix must not merge them."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("PARENT", source="tool")
    db.end_session("PARENT", "compression")
    db.create_session("DELEGATE", source="subagent", parent_session_id="PARENT",
                      model_config={"_delegate_from": "PARENT"})

    assert db._session_turn_lease_key("PARENT") == "PARENT"
    assert db._session_turn_lease_key("DELEGATE") == "DELEGATE"


def test_a_foreign_origin_child_keeps_its_own_conversation(tmp_path):
    """A child that only points at a compression-closed session as its ORIGIN is not a
    continuation of it, and must keep its own lease domain.

    This is the incident's actual row shape: ``20260929_175723_08ed50`` (``source='tool'``,
    no fork markers) re-resumed under the compression-closed ``20260929_165154_f18f7f``.
    Such a run must not inherit -- nor be handed -- its origin's live lease."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("ORIGIN", source="tool")
    db.append_message("ORIGIN", "user", "the rotated conversation")
    db.end_session("ORIGIN", "compression")
    db.create_session("RESUMED", source="tool", parent_session_id="ORIGIN")

    assert db._session_turn_lease_key("ORIGIN") == "ORIGIN"
    assert db._session_turn_lease_key("RESUMED") == "RESUMED"

    # ... and a lease on the origin is not reachable from the resumed run.
    assert db.try_acquire_session_turn_lease("ORIGIN", "pid=1:turn=origin", ttl_seconds=300.0)
    assert db.try_acquire_session_turn_lease("RESUMED", "pid=2:turn=resumed", ttl_seconds=300.0)
    assert len(_lease_rows(db)) == 2
