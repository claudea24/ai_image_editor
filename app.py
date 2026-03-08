import streamlit as st
import os
import json
import re
import base64
import tempfile
import io
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# ---------------------------
# Load API key
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# LangGraph: State & Nodes
# ---------------------------

class EditState(TypedDict):
    image_b64: str
    mask_b64: Optional[str]       # updated each retry to focus only on what still needs changing
    edit_inside: bool             # True = edit subject bbox; False = edit background
    original_prompt: str
    prompt: str
    prompt_history: list
    attempts: int                 # total DALL-E API calls (safety cap, always increments)
    accepted_attempts: int        # accepted edits only (counts toward max_attempts)
    max_attempts: int
    edited_b64: Optional[str]
    verified: bool
    feedback: str
    # ── RL fields ──
    best_b64: Optional[str]       # best result seen so far (highest reward)
    best_score: float             # reward score of best_b64 (0–10)
    score_history: list           # [(accepted_attempt, score, feedback), ...]
    satisfaction_threshold: float # stop when best_score >= this (0–10 scale)
    # ── Sizing: track padding so result can be cropped back to original ──
    orig_w: int
    orig_h: int
    pad_x: int
    pad_y: int


def rephrase_prompt_node(state: EditState) -> EditState:
    """Rewrite the user's prompt into a well-engineered DALL-E 2 inpainting prompt."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                f"You are an expert at writing DALL-E 2 inpainting prompts.\n"
                f"The user wants to edit a photo with this instruction: \"{state['original_prompt']}\"\n\n"
                f"Rewrite it as a high-quality DALL-E 2 inpainting prompt that:\n"
                f"- Describes the FINAL desired appearance of the edited region (not the action)\n"
                f"- Is specific about colors, textures, lighting, and style\n"
                f"- Matches the photorealistic style of the original image\n"
                f"- Is under 400 characters\n\n"
                f"Reply with ONLY the rewritten prompt, nothing else."
            )
        }],
        max_tokens=200
    )
    rephrased = (response.choices[0].message.content or "").strip()
    return {
        **state,
        "prompt": rephrased,
        "prompt_history": [rephrased]
    }


def prepare_node(state: EditState) -> EditState:
    """Ask GPT-4o for the main subject bbox + whether the edit targets inside or outside it.

    Mask convention for DALL-E 2:
      - transparent (alpha=0) = edit this area
      - opaque (alpha=255)    = preserve this area

    edit_inside=True  → horse color change: make horse bbox transparent, background opaque
    edit_inside=False → background change:  make horse bbox opaque, background transparent
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f'For the edit request: "{state["original_prompt"]}", '
                        f'find the bounding box of the PRIMARY SUBJECT (e.g. horse, person, car — NOT background). '
                        f'Also decide: does the edit target the subject itself (edit_inside=true) '
                        f'or the area surrounding it like the background (edit_inside=false)?\n'
                        f'Reply ONLY with JSON using 0–1 fractions: '
                        f'{{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9, "edit_inside": true}}'
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['image_b64']}"}
                }
            ]
        }],
        max_tokens=120
    )

    content = response.choices[0].message.content or ""
    match = re.search(r'\{.*\}', content, re.DOTALL)

    img = Image.open(io.BytesIO(base64.b64decode(state["image_b64"])))
    w, h = img.size
    x1, y1, x2, y2 = 0, 0, w, h
    edit_inside = True  # fallback: edit the subject

    if match:
        try:
            data = json.loads(match.group())
            x1 = int(data.get("x1", 0) * w)
            y1 = int(data.get("y1", 0) * h)
            x2 = int(data.get("x2", 1) * w)
            y2 = int(data.get("y2", 1) * h)
            edit_inside = bool(data.get("edit_inside", True))
        except (json.JSONDecodeError, KeyError):
            pass

    if edit_inside:
        # Edit subject bbox, preserve background
        mask = Image.new("RGBA", (w, h), (255, 255, 255, 255))  # all opaque
        mask.paste(Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0)), (x1, y1))
    else:
        # Edit background (everything outside subject bbox), preserve subject
        mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # all transparent
        mask.paste(Image.new("RGBA", (x2 - x1, y2 - y1), (255, 255, 255, 255)), (x1, y1))

    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {**state, "mask_b64": mask_b64, "edit_inside": edit_inside}


def edit_node(state: EditState) -> EditState:
    """Call DALL-E 2 with the subject mask to edit only the target area.
    Always edits from best_b64 if available (RL: exploit the best state found so far).
    """
    # RL: edit from best known result, not always from the original
    source_b64 = state["best_b64"] if state["best_b64"] else state["image_b64"]
    image_bytes = base64.b64decode(source_b64)
    mask_bytes = base64.b64decode(state["mask_b64"])

    img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    mask_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        img_tmp.write(image_bytes); img_tmp.flush(); img_tmp.close()
        mask_tmp.write(mask_bytes); mask_tmp.flush(); mask_tmp.close()

        with open(img_tmp.name, "rb") as img_f, open(mask_tmp.name, "rb") as mask_f:
            response = client.images.edit(
                model="dall-e-2",
                image=img_f,
                mask=mask_f,
                prompt=state["prompt"],
                size="1024x1024",
                response_format="b64_json"
            )
        edited_b64 = response.data[0].b64_json if response.data else None
    finally:
        os.unlink(img_tmp.name)
        os.unlink(mask_tmp.name)

    return {**state, "edited_b64": edited_b64, "attempts": state["attempts"] + 1}


REALISM_GATE      = 4.0  # reject if realism_score < this (disconnected / incoherent)
PRESERVATION_GATE = 4.0  # reject if preservation_score < this (too much collateral change)

def score_node(state: EditState) -> EditState:
    """RL reward function: GPT-4o scores the edit on THREE criteria (0–10 each).

    Criteria:
      1. edit_score         — how well the instruction was applied to the target area
      2. preservation_score — did the non-mentioned areas stay unchanged?
      3. realism_score      — realistic / coherent? no disconnected edges / artifacts?

    Hard gates (both must pass or the result is rejected):
      - realism_score      >= REALISM_GATE      (4.0)
      - preservation_score >= PRESERVATION_GATE (4.0)

    When rejected:
      - accepted_attempts is NOT incremented (rejected calls are free retries)
      - best_b64 reverts to the previous best so the next edit starts from there
      - feedback explains WHY it was rejected so update_mask & refine_prompt can fix it

    Composite (when both gates pass):
      score = 0.5 * edit + 0.3 * preservation + 0.2 * realism
      accepted_attempts += 1
    """
    if not state["edited_b64"]:
        return {**state, "verified": False, "feedback": "No image returned from DALL-E.",
                "score_history": state["score_history"] + [(state["accepted_attempts"], 0, "No image")]}

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f'You are evaluating an AI image edit. The instruction was: "{state["original_prompt"]}"\n\n'
                        f'You are shown TWO images:\n'
                        f'  Image 1 (first): the ORIGINAL before editing\n'
                        f'  Image 2 (second): the EDITED result\n\n'
                        f'Score the edited result on THREE dimensions (each 0–10):\n\n'
                        f'  edit_score: How well did the edit follow the instruction in the target area?\n'
                        f'    10 = perfectly applied, 0 = target area unchanged or completely wrong\n\n'
                        f'  preservation_score: Did the NON-TARGET areas stay exactly the same as the original?\n'
                        f'    Compare every part NOT mentioned in the instruction against Image 1.\n'
                        f'    10 = everything outside the target is pixel-identical, '
                        f'0 = background / other elements were heavily altered\n\n'
                        f'  realism_score: Is the edited image realistic and visually coherent?\n'
                        f'    Check: disconnected edges, visible seams, unnatural blending, '
                        f'color mismatches at the boundary, warped or distorted regions.\n'
                        f'    10 = seamless and photorealistic, 0 = obviously fake or disconnected\n\n'
                        f'Reply ONLY with valid JSON (no markdown, no code block):\n'
                        f'{{"edit_score": 8.0, "preservation_score": 9.0, "realism_score": 8.5, '
                        f'"what_changed_correctly": "...", '
                        f'"what_is_still_wrong": "...", '
                        f'"what_changed_outside_target": "...", '
                        f'"realism_issues": "none" or "describe any disconnected/seam issues"}}'
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['image_b64']}"}   # original
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['edited_b64']}"}  # edited
                }
            ]
        }],
        max_tokens=400
    )

    content = response.choices[0].message.content or ""
    match = re.search(r'\{.*\}', content, re.DOTALL)
    edit_score = 0.0
    preservation_score = 0.0
    realism_score = 0.0
    feedback_parts = {}
    if match:
        try:
            result = json.loads(match.group())
            edit_score = float(result.get("edit_score", 0))
            preservation_score = float(result.get("preservation_score", 0))
            realism_score = float(result.get("realism_score", 0))
            feedback_parts = {
                "what_changed_correctly":     result.get("what_changed_correctly", ""),
                "what_is_still_wrong":        result.get("what_is_still_wrong", ""),
                "what_changed_outside_target":result.get("what_changed_outside_target", ""),
                "realism_issues":             result.get("realism_issues", ""),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    structured_feedback = (
        f"Still wrong: {feedback_parts.get('what_is_still_wrong', 'unknown')} | "
        f"Outside target changed: {feedback_parts.get('what_changed_outside_target', 'unknown')} | "
        f"Realism issues: {feedback_parts.get('realism_issues', 'none')}"
    )

    # ── Hard gates: reject if preservation or realism is below threshold ──
    rejection_reasons = []
    if preservation_score < PRESERVATION_GATE:
        rejection_reasons.append(
            f"preservation {preservation_score:.1f}/10 < {PRESERVATION_GATE} "
            f"(outside target changed: {feedback_parts.get('what_changed_outside_target', '?')})"
        )
    if realism_score < REALISM_GATE:
        rejection_reasons.append(
            f"realism {realism_score:.1f}/10 < {REALISM_GATE} "
            f"(issues: {feedback_parts.get('realism_issues', '?')})"
        )

    if rejection_reasons:
        full_feedback = (
            f"REJECTED — {'; '.join(rejection_reasons)} | "
            f"edit {edit_score:.1f} | {structured_feedback}"
        )
        score_history = state["score_history"] + [(state["accepted_attempts"], 0.0, full_feedback)]
        # accepted_attempts NOT incremented — rejected calls are free retries
        return {
            **state,
            "verified": False,
            "feedback": full_feedback,
            "best_b64": state["best_b64"],    # revert to previous best
            "best_score": state["best_score"],
            "score_history": score_history,
            # accepted_attempts unchanged
        }

    # ── Both gates passed: compute composite score ──
    score = round(0.5 * edit_score + 0.3 * preservation_score + 0.2 * realism_score, 2)
    full_feedback = (
        f"edit {edit_score:.1f} + preservation {preservation_score:.1f} + "
        f"realism {realism_score:.1f} → {score:.1f}/10 | {structured_feedback}"
    )

    accepted_attempts = state["accepted_attempts"] + 1
    score_history = state["score_history"] + [(accepted_attempts, score, full_feedback)]

    # RL hill-climbing: only keep result if composite score improved
    if score > state["best_score"]:
        best_b64 = state["edited_b64"]
        best_score = score
    else:
        best_b64 = state["best_b64"]
        best_score = state["best_score"]

    return {
        **state,
        "verified": best_score >= state["satisfaction_threshold"],
        "feedback": full_feedback,
        "best_b64": best_b64,
        "best_score": best_score,
        "accepted_attempts": accepted_attempts,
        "score_history": score_history,
    }


def refine_prompt_node(state: EditState) -> EditState:
    """Build a targeted follow-up prompt based on the SPECIFIC PARTS still wrong in the feedback.

    E.g. if feedback says 'the legs and head retain the original color',
    generate: 'The horse legs and head rendered in solid white fur, photorealistic'
    rather than a generic restatement of the original instruction.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                f"You are refining an image editing prompt for DALL-E 2 inpainting.\n"
                f"Original user request: \"{state['original_prompt']}\"\n"
                f"Last prompt used: \"{state['prompt']}\"\n"
                f"Reviewer feedback: \"{state['feedback']}\"\n\n"
                f"Step 1: From the feedback, identify the SPECIFIC PARTS or AREAS that "
                f"are still incorrect (e.g. 'the legs and head retain the original color').\n"
                f"Step 2: Write a DALL-E 2 inpainting prompt that targets ONLY those specific "
                f"parts — describe exactly what those parts should look like in the final image.\n\n"
                f"Example: if feedback says 'legs and head retain original coloration', "
                f"write: 'The horse legs and head in solid bright white fur, matching the "
                f"rest of the white body, photorealistic, natural lighting'\n\n"
                f"Reply with ONLY the new prompt text. Be specific about the parts and "
                f"their desired appearance. Do not include preamble or explanation."
            )
        }],
        max_tokens=200
    )
    new_prompt = (response.choices[0].message.content or "").strip()
    return {
        **state,
        "prompt": new_prompt,
        "prompt_history": state["prompt_history"] + [new_prompt]
    }


def update_mask_node(state: EditState) -> EditState:
    """Before each retry, ask GPT-4o to identify ONLY what still needs changing.

    Sends the current best image and asks for the tightest bounding box around
    the area that is NOT YET correct. Only that region becomes transparent (editable);
    everything else stays opaque so DALL-E cannot touch it.
    """
    source_b64 = state["best_b64"] if state["best_b64"] else state["image_b64"]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f'The edit instruction was: "{state["original_prompt"]}"\n'
                        f'Reviewer feedback from the last attempt: "{state["feedback"]}"\n\n'
                        f'The feedback names SPECIFIC PARTS that are still wrong '
                        f'(e.g. "legs and head retain original color"). '
                        f'Locate those exact parts in the image and draw the TIGHTEST bounding box '
                        f'that encloses only those not-yet-correct regions. '
                        f'Ignore everything that is already correct. '
                        f'Reply ONLY with JSON using 0–1 fractions of image width/height: '
                        f'{{"x1": 0.2, "y1": 0.1, "x2": 0.7, "y2": 0.6}}'
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{source_b64}"}
                }
            ]
        }],
        max_tokens=120
    )

    content = response.choices[0].message.content or ""
    match = re.search(r'\{.*\}', content, re.DOTALL)

    if not match:
        return state  # fallback: keep existing mask unchanged

    try:
        data = json.loads(match.group())
        img = Image.open(io.BytesIO(base64.b64decode(source_b64)))
        w, h = img.size
        x1 = int(max(0, data.get("x1", 0)) * w)
        y1 = int(max(0, data.get("y1", 0)) * h)
        x2 = int(min(1, data.get("x2", 1)) * w)
        y2 = int(min(1, data.get("y2", 1)) * h)

        # Only the still-incorrect region is transparent (editable); all else is opaque
        mask = Image.new("RGBA", (w, h), (255, 255, 255, 255))  # fully opaque = preserve
        mask.paste(Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0)), (x1, y1))  # transparent = edit

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        return {**state, "mask_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return state  # fallback: keep existing mask unchanged


def should_retry(state: EditState) -> str:
    if state["best_score"] >= state["satisfaction_threshold"]:
        return "done"
    # Stop when enough ACCEPTED edits have been made
    if state["accepted_attempts"] >= state["max_attempts"]:
        return "done"
    # Absolute safety cap: max 4x total DALL-E calls to prevent infinite rejection loops
    if state["attempts"] >= state["max_attempts"] * 4:
        return "done"
    return "refine"


@st.cache_resource
def build_workflow():
    graph = StateGraph(EditState)
    graph.add_node("rephrase_prompt", rephrase_prompt_node)
    graph.add_node("prepare", prepare_node)
    graph.add_node("edit", edit_node)
    graph.add_node("score", score_node)
    graph.add_node("update_mask", update_mask_node)   # re-focuses mask on still-wrong pixels
    graph.add_node("refine_prompt", refine_prompt_node)
    graph.set_entry_point("rephrase_prompt")
    graph.add_edge("rephrase_prompt", "prepare")
    graph.add_edge("prepare", "edit")
    graph.add_edge("edit", "score")
    # On retry: first tighten the mask, then refine the prompt, then edit again
    graph.add_conditional_edges("score", should_retry, {"refine": "update_mask", "done": END})
    graph.add_edge("update_mask", "refine_prompt")
    graph.add_edge("refine_prompt", "edit")
    return graph.compile()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Photo Editor", layout="wide", page_icon="🖼️")

st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }

  .card { border: 1px solid #e0e0e0; border-radius: 12px; padding: 1rem;
          background: #fafafa; text-align: center; }
  .card-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
                text-transform: uppercase; color: #888; margin-bottom: 0.5rem; }
  .placeholder { height: 300px; display: flex; align-items: center;
                 justify-content: center; background: #f0f0f0;
                 border-radius: 8px; border: 2px dashed #ccc; color: #aaa; font-size: 0.9rem; }
  .history-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
                   text-transform: uppercase; color: #888; margin: 1.5rem 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "history" not in st.session_state:
    st.session_state.history = []   # list of PIL Images
if "last_status" not in st.session_state:
    st.session_state.last_status = None   # dict: {verified, feedback, prompt_history, attempts}
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None
# Pending workflow: decouple button click from execution so old result is cleared first
if "_workflow_pending" not in st.session_state:
    st.session_state._workflow_pending = False
if "_workflow_prompt" not in st.session_state:
    st.session_state._workflow_prompt = ""
if "_workflow_max_attempts" not in st.session_state:
    st.session_state._workflow_max_attempts = 2
if "_workflow_satisfaction" not in st.session_state:
    st.session_state._workflow_satisfaction = 7.0

# ── Sidebar: upload + controls ──
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem'><span style='font-size:1.3rem'>✦</span> <b>AI Photo Editor</b></div>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_id = (uploaded_file.name, uploaded_file.size)
        if st.session_state.uploaded_file_id != file_id:
            # Only reset when a genuinely new file is uploaded
            st.session_state.uploaded_file_id = file_id
            st.session_state.current_image = Image.open(uploaded_file)
            st.session_state.result_image = None
            st.session_state.history = []
            st.session_state.last_status = None

    with st.form("edit_form"):
        st.markdown("<div style='margin-top:1.2rem; font-size:0.75rem; letter-spacing:1px; color:#6666aa; text-transform:uppercase;'>Edit Instruction</div>", unsafe_allow_html=True)
        prompt = st.text_area("Prompt", placeholder="e.g. change the background to white",
                              label_visibility="collapsed", height=80)

        st.markdown("<div style='margin-top:1rem; font-size:0.75rem; letter-spacing:1px; color:#6666aa; text-transform:uppercase;'>Satisfaction Threshold</div>", unsafe_allow_html=True)
        satisfaction_pct = st.slider("Satisfaction %", 10, 100, 70, step=5, label_visibility="collapsed",
                                      help="Stop retrying once the result scores at or above this % (e.g. 70 = stop at 7/10)")
        satisfaction_threshold = satisfaction_pct / 10.0  # convert to 0–10 scale

        st.markdown("<div style='margin-top:1rem; font-size:0.75rem; letter-spacing:1px; color:#6666aa; text-transform:uppercase;'>Max Retry Attempts</div>", unsafe_allow_html=True)
        max_attempts = st.slider("Max retries", 1, 10, 2, label_visibility="collapsed")

        apply_clicked = st.form_submit_button(
            "Apply Edit", use_container_width=True,
            disabled=st.session_state._workflow_pending
        )

    _generating = st.session_state._workflow_pending

    # "Save & Continue" — only show when there's a result to commit and not generating
    save_and_continue = False
    if st.session_state.get("result_image") is not None and not _generating:
        save_and_continue = st.button("✓ Save Edit & Continue", use_container_width=True,
                                      type="primary")

    st.markdown("---")

    # Undo / Download — hidden while generating
    if not _generating:
        col_undo, col_dl = st.columns(2)
        with col_undo:
            if st.button("↩ Undo", use_container_width=True) and st.session_state.history:
                st.session_state.current_image = st.session_state.history.pop()
                st.session_state.result_image = None
                st.session_state.last_status = None
                st.rerun()
        with col_dl:
            export_img = st.session_state.result_image or st.session_state.current_image
            if export_img:
                dl_buf = io.BytesIO()
                export_img.save(dl_buf, format="PNG")
                st.download_button("↓ Save", data=dl_buf.getvalue(),
                                   file_name="edited.png", mime="image/png",
                                   use_container_width=True)

# ── Hero ──
st.title("🖼️ AI Photo Editor")
st.caption("Describe your edit — the agent locates the subject, edits only that area, and verifies the result.")
st.divider()

# ── Main image area ──
col_orig, col_result = st.columns(2, gap="large")

with col_orig:
    st.markdown('<div class="card"><div class="card-label">Original</div></div>', unsafe_allow_html=True)
    if st.session_state.current_image:
        st.image(st.session_state.current_image, use_container_width=True)
    else:
        st.markdown('<div class="placeholder">Upload an image to get started</div>', unsafe_allow_html=True)

with col_result:
    _last = st.session_state.last_status
    _below_threshold = _last and not _last.get("verified")
    _thr = _last.get("satisfaction_threshold", st.session_state._workflow_satisfaction) if _last else None
    label_html = (
        '<div class="card-label" style="color:#c0392b;">Result — Below Threshold</div>'
        if _below_threshold else
        '<div class="card-label">Result</div>'
    )
    st.markdown(f'<div class="card">{label_html}</div>', unsafe_allow_html=True)
    if st.session_state.result_image:
        st.image(st.session_state.result_image, use_container_width=True)
        if _below_threshold:
            st.markdown(
                f'<div style="background:#fff0f0;border:1px solid #e74c3c;border-radius:8px;'
                f'padding:0.5rem 0.8rem;margin-top:0.4rem;color:#c0392b;font-size:0.85rem;">'
                f'⚠️ Best result shown — score {_last["best_score"]:.1f}/10 did not reach '
                f'the {_thr * 10:.0f}% threshold. Save & Continue to use it anyway, or retry.'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div class="placeholder">Your edited image will appear here</div>', unsafe_allow_html=True)

# ── Status ──
if st.session_state.last_status:
    s = st.session_state.last_status
    st.divider()
    total = s["attempts"]
    passed_gates = s.get("accepted_attempts", total)  # passed realism+preservation hard gates
    rejected_gates = total - passed_gates              # rejected by hard gates
    # "Accepted" means the final result met the satisfaction threshold and was shown
    final_accepted = 1 if s["verified"] else 0
    attempt_label = (
        f"{final_accepted} accepted / {total} total "
        f"({rejected_gates} rejected by quality gates)"
    )
    if s["verified"]:
        st.success(f"Score {s['best_score']:.1f}/10 — Verified — {attempt_label}")
    else:
        st.warning(f"Best score {s['best_score']:.1f}/10 — {attempt_label}")

    if s.get("score_history"):
        threshold = s.get("satisfaction_threshold", st.session_state._workflow_satisfaction)
        with st.expander("Reward score history (RL progress)"):
            for i, entry in enumerate(s["score_history"]):
                attempt, score, fb = entry
                is_rejected_gate = fb.startswith("REJECTED")
                below_threshold = not is_rejected_gate and score < threshold
                # Only the very last entry can be "accepted" (shown as result), and only if verified
                is_final_accepted = (
                    not is_rejected_gate
                    and i == len(s["score_history"]) - 1
                    and s["verified"]
                )
                bar = "█" * int(score) + "░" * (10 - int(score))
                if is_rejected_gate:
                    label = "**❌ Rejected by quality gates**"
                elif is_final_accepted:
                    label = f"**✅ Accepted (attempt #{attempt})**"
                else:
                    label = f"**⚠️ Below threshold (attempt #{attempt})**"
                st.markdown(f"{label} `{bar}` {score:.1f}/10 — {fb}")

    if len(s.get("prompt_history", [])) >= 1:
        with st.expander("View prompt history"):
            st.caption(f"Your prompt: {s['original_prompt']}")
            for i, p in enumerate(s["prompt_history"]):
                label = "Rephrased for DALL-E" if i == 0 else f"Refined attempt {i + 1}"
                st.markdown(f"**{label}:** {p}")

# ── Generating indicator ──
if st.session_state._workflow_pending:
    st.divider()
    st.markdown(
        '<div style="text-align:center;padding:1rem;color:#6666aa;font-size:0.95rem;">'
        '⏳ Generating your edit — this may take a moment…'
        '</div>',
        unsafe_allow_html=True
    )

# ── Save & Continue: commit result as new base ──
if save_and_continue and st.session_state.result_image is not None:
    st.session_state.history.append(st.session_state.current_image)
    st.session_state.current_image = st.session_state.result_image
    st.session_state.result_image = None
    st.session_state.last_status = None
    st.rerun()

# ── On button click: stash params, clear old result, rerun so UI refreshes before workflow ──
if apply_clicked and prompt and st.session_state.current_image:
    st.session_state._workflow_prompt = prompt
    st.session_state._workflow_max_attempts = max_attempts
    st.session_state._workflow_satisfaction = satisfaction_threshold
    st.session_state._workflow_pending = True
    st.session_state.result_image = None
    st.session_state.last_status = None
    st.rerun()

# ── Run workflow (on the fresh render after result was cleared) ──
if st.session_state._workflow_pending and st.session_state.current_image:
    st.session_state._workflow_pending = False
    prompt = st.session_state._workflow_prompt
    max_attempts = st.session_state._workflow_max_attempts
    satisfaction_threshold = st.session_state._workflow_satisfaction
    if prompt:
        # DALL-E 2 requires image and mask to be the same square size (1024x1024).
        # Resize preserving aspect ratio, then center-pad to 1024x1024.
        # Use a neutral gray pad so edits like "change background to white" aren't
        # trivially satisfied by the white border and inflate the score.
        # Track pad offsets so we can crop the result back to original dimensions.
        img = st.session_state.current_image.convert("RGBA")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        orig_w, orig_h = img.width, img.height
        pad_x = (1024 - orig_w) // 2
        pad_y = (1024 - orig_h) // 2
        padded = Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))
        padded.paste(img, (pad_x, pad_y))
        buf = io.BytesIO()
        padded.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        initial_state: EditState = {
            "image_b64": image_b64,
            "mask_b64": None,
            "edit_inside": True,       # set properly by prepare_node
            "original_prompt": prompt,
            "prompt": prompt,
            "prompt_history": [prompt],
            "attempts": 0,
            "accepted_attempts": 0,
            "max_attempts": max_attempts,
            "edited_b64": None,
            "verified": False,
            "feedback": "",
            "best_b64": None,
            "best_score": 0.0,
            "score_history": [],
            "satisfaction_threshold": satisfaction_threshold,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }

        with st.spinner("Locating subject, editing, and verifying..."):
            try:
                workflow = build_workflow()
                # Rejected attempts don't count toward max_attempts but still consume graph steps.
                # Safety cap is 4x max_attempts total DALL-E calls, each cycle = 6 nodes.
                recursion_limit = max_attempts * 4 * 6 + 20
                final_state = workflow.invoke(
                    initial_state,
                    config={"recursion_limit": recursion_limit}
                )
            except Exception as e:
                st.error(f"Workflow error: {e}")
                final_state = None

        if final_state and final_state.get("verified"):
            result_b64 = final_state.get("best_b64") or final_state.get("edited_b64")
            result_full = Image.open(io.BytesIO(base64.b64decode(result_b64)))
            # Crop away the padding to restore original image dimensions
            px, py = final_state["pad_x"], final_state["pad_y"]
            ow, oh = final_state["orig_w"], final_state["orig_h"]
            edited_image = result_full.crop((px, py, px + ow, py + oh))
            # Don't update current_image — user must explicitly click "Save & Continue"
            st.session_state.result_image = edited_image
            st.session_state.last_status = {
                "verified": True,
                "feedback": final_state["feedback"],
                "attempts": final_state["attempts"],
                "accepted_attempts": final_state["accepted_attempts"],
                "prompt_history": final_state["prompt_history"],
                "original_prompt": final_state["original_prompt"],
                "best_score": final_state["best_score"],
                "score_history": final_state["score_history"],
                "satisfaction_threshold": satisfaction_threshold,
            }
            st.rerun()
        elif final_state:
            best_score = final_state.get("best_score", 0.0)
            best_b64 = final_state.get("best_b64") or final_state.get("edited_b64")
            if best_b64:
                best_full = Image.open(io.BytesIO(base64.b64decode(best_b64)))
                px, py = final_state["pad_x"], final_state["pad_y"]
                ow, oh = final_state["orig_w"], final_state["orig_h"]
                st.session_state.result_image = best_full.crop((px, py, px + ow, py + oh))
            else:
                st.session_state.result_image = None
            st.session_state.last_status = {
                "verified": False,
                "feedback": final_state["feedback"],
                "attempts": final_state["attempts"],
                "accepted_attempts": final_state["accepted_attempts"],
                "prompt_history": final_state["prompt_history"],
                "original_prompt": final_state["original_prompt"],
                "best_score": best_score,
                "score_history": final_state["score_history"],
                "satisfaction_threshold": satisfaction_threshold,
            }
            st.rerun()
        else:
            st.error("No image was returned. Try a different prompt or image.")

# ── History strip (always at the very bottom) ──
if st.session_state.history:
    st.divider()
    st.markdown('<div class="history-label">Edit History — click Restore to go back</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(st.session_state.history), 8))
    for i, hist_img in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        with cols[i]:
            # Show small fixed-width thumbnail
            thumb = hist_img.copy()
            thumb.thumbnail((120, 120), Image.LANCZOS)
            st.image(thumb, width=100)
            if st.button("Restore", key=f"restore_{i}"):
                st.session_state.history = st.session_state.history[:idx]
                st.session_state.current_image = hist_img
                st.session_state.result_image = None
                st.session_state.last_status = None
                st.rerun()
