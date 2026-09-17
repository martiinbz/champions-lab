# Dashboard simplification implementation plan

> **For Codex:** Follow test-driven development and verify the rendered Streamlit page before completion.

**Goal:** Add a configurable simulation command and reshape the dashboard into a clear probability table, 36-team expected ranking, and all-team expected-position evolution view.

**Architecture:** Keep snapshot loading and probability calculations pure and testable. Add small presentation helpers for ordered data, crests, and evolution data, then compose them in Streamlit. Keep all images local for offline use.

**Tech stack:** Python, pandas, Streamlit, Plotly, PowerShell, pytest.

---

### Task 1: Lock the dashboard data contract with failing tests

**Files:**
- Modify: `tests/test_dashboard.py`
- Test: `tests/test_dashboard.py`

1. Add tests for the exact probability-column order.
2. Add tests that expected ranking contains every team and is ordered 1 through 36 by expected rank.
3. Add tests that expected-position history includes every team in every saved snapshot.
4. Add tests for local crest lookup and deterministic fallback.
5. Run the targeted tests and confirm they fail for the missing behavior.

### Task 2: Implement pure dashboard helpers

**Files:**
- Modify: `app/streamlit_app.py`
- Create: `app/assets/crests/README.md`
- Create: `app/assets/crests/team-crests.json`
- Test: `tests/test_dashboard.py`

1. Implement the approved probability order.
2. Implement ranking and evolution data helpers.
3. Implement local crest resolution and fallback badges.
4. Run targeted tests until green.

### Task 3: Build the simplified Streamlit composition

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `tests/test_dashboard.py`

1. Add concise visual styling and summary cards.
2. Render the probability table with crests.
3. Render the complete expected ranking with crests.
4. Render all-team expected-position evolution with a reversed y-axis.
5. Move technical details into a collapsed expander.
6. Verify with Streamlit AppTest and a browser render.

### Task 4: Add the configurable simulation command

**Files:**
- Create: `Nueva-simulacion.ps1`
- Modify: `README.md`
- Modify: `tests/test_entrypoints.py`

1. Add a failing test for command existence and parameter forwarding.
2. Implement the PowerShell wrapper around the existing simulation CLI.
3. Document the 200,000-simulation example and the update-then-simulate workflow.
4. Run targeted tests until green.

### Task 5: Final verification and delivery

**Files:**
- Modify as needed from verification findings.

1. Run the full test suite.
2. Run a small real simulation via the new command.
3. Inspect the live dashboard at desktop and narrow viewport sizes.
4. Review the git diff, commit, and push to GitHub.
