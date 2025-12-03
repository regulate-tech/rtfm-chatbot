# --- IMPORTS ---
import os
import random
import time

# Universal TOML support (Pro Tip: Handles Python < 3.11 automatically)
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import streamlit as st
from ollama import Client

# Try imports for stability
try:
    from ddgs import DDGS

    HAS_DDG = True
except ImportError:
    HAS_DDG = False

# --- CONSTANTS ---
# Defined globally so we can calculate 'Next Module' logic
CHAPTER_LIST = [
    "🏠 Home",
    "1. Raw Recruit",
    "2. Policy Binder",
    "3. Pager & Phone",
    "4. Bedside Manner",
    "5. The Burnout",
    "6. Insider Threat",
    "7. The Full Monty",
]
DEFAULT_SUMMARY = "You learnt about another feature of chatbots in this module, now moving on to the next."


# --- CONFIG LOADER ---
@st.cache_data
def load_config(filename="config.toml"):
    """Loads the TOML configuration file for text and settings."""
    if not os.path.exists(filename):
        return None
    # 'rb' mode is required for tomllib
    with open(filename, "rb") as f:
        return tomllib.load(f)


def validate_config(config_data):
    """
    Sanity checks the TOML file on startup.
    Returns a list of warnings (if any).
    """
    warnings = []

    # 1. defined required sections (keys in the TOML)
    required_sections = [
        "app_settings",
        "ollama_settings",
        "models",
        "chapter_1",
        "chapter_2",
        "chapter_3",
        "chapter_4",
        "chapter_5",
        "chapter_6",
    ]

    for section in required_sections:
        if section not in config_data:
            st.error(f"🚨 CRITICAL: Missing section '[{section}]' in config.toml")
            st.stop()  # Stop execution if a whole section is missing

        # 2. Check for 'summary' specifically (Non-critical warning)
        # We skip checking 'app_settings' or 'models' for summaries
        if section.startswith("chapter_") and "summary" not in config_data[section]:
            warnings.append(f"⚠️ Missing 'summary' in [{section}]. Using default text.")

    return warnings


# Load Config Immediately
config = load_config()

# Fallback if config is missing (Safety Check)
if not config:
    st.error("🚨 Configuration file 'config.toml' not found!")
    st.stop()

# RUN THE VALIDATOR
config_warnings = validate_config(config)

# OPTIONAL: Display warnings in the sidebar for the developer
if config_warnings:
    with st.sidebar:
        with st.expander("🛠️ Config Warnings", expanded=False):
            for w in config_warnings:
                st.caption(w)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title=config["app_settings"]["title"],
    layout="wide",
    page_icon=config["app_settings"]["icon"],
)


# --- GLOBAL SETUP ---
@st.cache_resource
def get_ollama_client():
    try:
        client = Client(host="http://127.0.0.1:11434")
        client.list()
        return client, True
    except Exception:
        return None, False


client, connection_status = get_ollama_client()

# --- HELPER FUNCTIONS ---


def read_uploaded_text(uploaded_file):
    """
    Reads a text file uploaded via Streamlit and returns the string content.
    Handles basic decoding errors.
    """
    if uploaded_file is None:
        return ""
    try:
        # standard utf-8 decoding
        return uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return "Error: File must be a text file (UTF-8)."


def render_instructions(title, content):
    with st.expander(f"📖 {title} (Click to Read Instructions)", expanded=True):
        st.markdown(content)


def init_recording_state():
    if "recording_log" not in st.session_state:
        st.session_state["recording_log"] = []
    if "is_recording" not in st.session_state:
        st.session_state["is_recording"] = False


def save_interaction(module, label, user_query, ai_response, model, stats):
    if st.session_state.get("is_recording"):
        params = {
            "ctx": st.session_state.get("ctx_slider", "Default"),
            "out": st.session_state.get("out_slider", "Default"),
        }
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "module": module,
            "label": label,
            "model": model,
            "query": user_query,
            "response": ai_response,
            "params": params,
            "stats": stats or {},
        }
        st.session_state["recording_log"].append(entry)


def generate_markdown_log():
    log = st.session_state.get("recording_log", [])
    if not log:
        return "# Empty Log"
    md = f"# {config['app_settings']['title']} - Session Log\n**Date:** {time.strftime('%Y-%m-%d')}\n---\n\n"
    for i, entry in enumerate(log, 1):
        md += f"## {i}. {entry['module']} ({entry['timestamp']})\n**Type:** {entry['label']} | **Model:** `{entry['model']}`\n"
        if s := entry.get("stats"):
            try:
                tps = s.get("eval_count", 0) / (s.get("eval_duration", 1) / 1e9)
                md += (
                    f"**Stats:** {tps:.1f} T/s | {s.get('total_duration',0)/1e9:.2f}s\n"
                )
            except Exception:
                pass
        md += f"\n### Query:\n> {entry['query']}\n\n### Response:\n{entry['response']}\n---\n"
    return md


def get_live_wait_times():
    """
    Generates fake data using locations defined in config.toml.
    """
    st.toast("Paging Operations Centre...", icon="📟")
    time.sleep(1)

    # 1. Get the Chapter 3 config
    c3_conf = config.get("chapter_3", {})

    # 2. Get the template and locations (with safe defaults)
    template = c3_conf.get(
        "pager_response_template", "{loc1}: {val1}m | {loc2}: {val2}m"
    )
    loc1 = c3_conf.get("location_1", "Location A")
    loc2 = c3_conf.get("location_2", "Location B")

    # 3. Generate random data (The "Live" element)
    # We use different ranges to make it feel realistic
    val1 = random.randint(5, 45)  # Quick location
    val2 = random.randint(60, 180)  # Busy location

    # 4. Inject everything into the template
    return (
        template.replace("{loc1}", loc1)
        .replace("{loc2}", loc2)
        .replace("{val1}", str(val1))
        .replace("{val2}", str(val2))
    )


def perform_search(query, use_mock=False):
    """
    Performs a real or mock search, with a configurable domain limit.
    """
    # 1. Load Chapter 3 Config
    c3_conf = config.get("chapter_3", {})

    # DEBUG: Visual confirmation
    if use_mock:
        st.toast(f"🕵️‍♀️ Triggering MOCK search for: {query}")
    else:
        st.toast(f"🌐 Triggering REAL search for: {query}")

    if use_mock:
        time.sleep(1.5)
        results = c3_conf.get("mock_search_results", [])
        if not results:
            return [
                {
                    "title": "Configuration Error",
                    "href": "#",
                    "body": "No mock results found in config.toml.",
                }
            ]
        return results

    # 2. Check for DuckDuckGo library
    if not HAS_DDG:
        return [
            {
                "title": "Error",
                "href": "#",
                "body": "duckduckgo-search library missing.",
            }
        ]

    # 3. Construct the Query
    # We get the domain limit from config (defaulting to 'nhs.uk' if missing, for safety)
    domain_limit = c3_conf.get("search_domain_limit", "nhs.uk")

    if domain_limit:
        # If a limit exists, we append the 'site:' operator
        search_query = f"{query} site:{domain_limit}"
    else:
        # If it's empty string "", we just search the query as-is
        search_query = query

    try:
        # We perform the search
        results = list(
            DDGS().text(search_query, region="uk-en", max_results=3, backend="html")
        )
        return results
    except Exception as e:
        return [{"title": "Connection Error", "href": "#", "body": f"Error: {e}"}]


def format_search_results(results):
    context_text = ""
    for r in results:
        context_text += f"SOURCE: [{r['title']}]({r['href']})\nCONTENT: {r['body']}\n\n"
    return context_text


def stream_chat(model, messages, override_options=None, stats_container=None):
    ctx_limit = st.session_state.get("ctx_slider", 4096)
    output_limit = st.session_state.get("out_slider", 750)
    options = {"num_ctx": ctx_limit, "num_predict": output_limit, "temperature": 0.7}
    if override_options:
        options.update(override_options)

    try:
        stream = client.chat(
            model=model, messages=messages, stream=True, options=options
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
            if stats_container is not None and "eval_count" in chunk:
                stats_container.update(chunk)
    except Exception as e:
        yield f"⚠️ **Model Error:** {str(e)}"


# --- NEW UI HELPERS ---


def render_response_metadata(stats, full_prompt):
    """
    Shows the Speed prominently with a tooltip explaining 'Speed' vs 'Max Tokens'.
    """
    if not stats:
        return

    # 1. Calculate Metrics
    try:
        eval_dur = stats.get("eval_duration", 0) / 1e9
        prompt_eval_dur = stats.get("prompt_eval_duration", 0) / 1e9
        tps = stats.get("eval_count", 0) / eval_dur if eval_dur > 0 else 0
    except Exception:
        tps = 0
        eval_dur = 0
        prompt_eval_dur = 0

    # 2. Determine Color (Visual Feedback)
    if tps > 30:
        color_str = ":green"  # Fast
    elif tps > 10:
        color_str = ":orange"  # Average
    else:
        color_str = ":red"  # Slow

    # 3. Tooltip Explanation
    # This distinguishes between "Thinking Speed" (Hardware) and "Response Limit" (Slider)
    speed_tooltip = """
    🧠 Brain Speed (Tokens/sec):
    This is how fast the AI is 'thinking'. 
    It depends on your hardware and the Model Size.
    
    🛑 Max Tokens (see Advanced Parameters):
    You can limit the reply size the model is ALLOWED to produce.
    This won't make it run faster but will shorten your wait.
    """

    # 4. Render Layout
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        # We use the native 'help' parameter to show the tooltip icon (?)
        st.markdown(f"**Speed:** {color_str}[**{tps:.1f} T/s**]", help=speed_tooltip)

    with c2:
        with st.popover("⏱️ Time", use_container_width=True):
            st.caption("Breakdown")
            st.write(f"**Total Time:** {stats.get('total_duration', 0)/1e9:.2f}s")
            st.write(f"**Processing:** {prompt_eval_dur:.2f}s")
            st.write(f"**Generating:** {eval_dur:.2f}s")

    with c3:
        with st.popover("📝 Prompt", use_container_width=True):
            st.caption("The raw text sent to the model:")
            st.code(full_prompt, language="text")


def render_module_transition(current_module_name, summary_text, placeholder):
    """
    Renders the Finish button into a specific UI placeholder (e.g., in the sidebar).
    """
    try:
        current_index = CHAPTER_LIST.index(current_module_name)
        next_index = current_index + 1
    except ValueError:
        st.error(f"🚨 PROGRAMMER ERROR: Module '{current_module_name}' not found.")
        return

    view_key = f"view_summary_{current_module_name}"

    def show_summary():
        st.session_state[view_key] = True

    # --- KEY CHANGE: Use the placeholder ---
    # We write into the specific container we passed in
    with placeholder.container():
        # Optional: Add a divider to separate it visually
        st.write("---")
        if st.button(
            "🏁 Finish Module",
            key=f"btn_finish_{current_module_name}",
            on_click=show_summary,
            use_container_width=True,
        ):
            pass

    # The Summary Text still appears in the main area (not the sidebar)
    if st.session_state.get(view_key, False):
        st.divider()
        st.info("🎓 **Module Summary**")
        st.markdown(summary_text)

        if next_index < len(CHAPTER_LIST):
            next_module_name = CHAPTER_LIST[next_index]

            def go_next():
                st.session_state["nav_selection"] = next_module_name
                st.session_state[view_key] = False

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.button(
                    f"➡️ Start {next_module_name}",
                    type="primary",
                    on_click=go_next,
                    use_container_width=True,
                )
        else:
            st.success("🎉 You have completed the entire Cookbook!")


def render_quick_prompts(prompts_list):
    """
    Renders a row of buttons. Returns the text of the button if clicked,
    otherwise returns None.
    """
    if not prompts_list:
        return None

    # Create columns for a horizontal layout
    cols = st.columns(len(prompts_list))
    selected_prompt = None

    for i, text in enumerate(prompts_list):
        # We use a unique key based on the text to avoid conflicts
        if cols[i].button(text, key=f"qp_{text[:10]}_{i}", use_container_width=True):
            selected_prompt = text

    return selected_prompt


# --- SIDEBAR ---
with st.sidebar:
    st.title(config["app_settings"]["title"])
    if connection_status:
        st.success("🟢 Ollama Connected")
    else:
        st.error("🔴 Ollama Disconnected")
        st.stop()

    try:
        models_info = client.list()
        model_names = [m["model"] for m in models_info["models"]]
    except Exception:
        model_names = []

    if "saved_model_choice" not in st.session_state:
        st.session_state["saved_model_choice"] = model_names[0] if model_names else None

    # --- NAVIGATION ---
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = CHAPTER_LIST[0]

    # Use 'nav_selection' key to allow programmatic changing of pages
    selected_module = st.radio("Select Chapter:", CHAPTER_LIST, key="nav_selection")

    module_finish_placeholder = st.empty()

    st.divider()

    st.subheader("📼 Session Recorder")
    init_recording_state()
    if not st.session_state["is_recording"]:
        if st.button("🔴 Start Recording", type="primary"):
            st.session_state["is_recording"] = True
            st.session_state["recording_log"] = []
            st.rerun()
    else:
        st.success(f"REC: {len(st.session_state['recording_log'])} items")
        if st.button("⏹️ Stop & Save"):
            st.session_state["is_recording"] = False
            st.rerun()
    if st.session_state["recording_log"]:
        st.download_button(
            "💾 Download Log",
            generate_markdown_log(),
            f"session_{time.strftime('%Y%m%d')}.md",
            "text/markdown",
        )

    st.divider()

    if "Home" not in selected_module and "Raw Recruit" not in selected_module:
        st.subheader("🍳 Kitchen Settings")
        try:
            idx = model_names.index(
                st.session_state.get("saved_model_choice", model_names[0])
            )
        except Exception:
            idx = 0

        def update_model():
            st.session_state["saved_model_choice"] = st.session_state["model_selector"]

        primary_model = st.selectbox(
            "Choose a Base Model",
            model_names,
            index=idx,
            key="model_selector",
            on_change=update_model,
        )

    with st.expander("⚙️ Advanced Parameters"):
        # 1. Get defaults from config (Safe .get() with fallback)
        # We look in 'ollama_settings' first
        ollama_conf = config.get("ollama_settings", {})

        default_ctx = ollama_conf.get("default_context_window", 4096)
        default_out = ollama_conf.get("default_max_tokens", 750)

        # 2. Use these variables in the sliders
        st.slider(
            "Context Window",
            2048,
            8192,
            value=default_ctx,  # <--- Uses TOML value
            step=512,
            key="ctx_slider",
        )

        st.slider(
            "Max Response",
            256,
            4096,
            value=default_out,  # <--- Uses TOML value
            step=256,
            key="out_slider",
        )

    if st.button("🗑️ Clear Page History"):
        st.session_state[f"history_{selected_module}"] = []
        st.rerun()


# --- MAIN CONTENT ---
if f"history_{selected_module}" not in st.session_state:
    st.session_state[f"history_{selected_module}"] = []

# HOME
if "Home" in selected_module:
    st.title(config["app_settings"]["intro_title"])
    st.markdown(config["app_settings"]["intro_text"])

# CHAPTER 1
# CHAPTER 1
elif "1. Raw Recruit" in selected_module:
    c_conf = config["chapter_1"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    # --- LOAD CONFIG & MODELS ---
    raw_models_config = config.get("models", {})
    if not raw_models_config:
        st.error("🚨 No [models] section found in config.toml")
        st.stop()

    MODEL_SIZES = {k: v for k, v in raw_models_config.items() if isinstance(v, dict)}

    # 1. MODEL DOWNLOADER UI (Keep existing logic)
    cols = st.columns(len(MODEL_SIZES))
    try:
        raw_models = [m["model"] for m in client.list()["models"]]
    except Exception:
        raw_models = []
    target_threads = config.get("ollama_settings", {}).get("num_thread", 2)

    for i, (size, details) in enumerate(MODEL_SIZES.items()):
        with cols[i]:
            st.markdown(f"**{size}** ({details['desc']})")
            custom_model_exists = any(details["tag"] in m for m in raw_models)
            if custom_model_exists:
                st.success(f"✅ Ready ({details['tag']})")
            else:
                base_exists = any(details["base"] in m for m in raw_models)
                btn_label = f"⚙️ Optimize {size}" if base_exists else f"⬇️ Get {size}"
                if st.button(btn_label, key=f"dl_{i}"):
                    # ... (Keep your existing download logic here) ...
                    pass

    st.divider()

    # 2. SELECT MODELS
    selected_sizes = st.multiselect(
        "Select Models:", list(MODEL_SIZES.keys()), default=list(MODEL_SIZES.keys())[:2]
    )

    # 3. FINISH BUTTON (Sidebar Persistence)
    render_module_transition(
        "1. Raw Recruit",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 4. QUICK PROMPTS
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 5. CHAT INPUT & CONSOLIDATION
    typed_input = st.chat_input(c_conf["input_placeholder"])
    user_input = clicked_prompt or typed_input

    if user_input:
        # UX Fix: If they clicked a button, show the message visually
        # (Since Ch1 doesn't have a history loop, we just show it once here)
        if clicked_prompt:
            with st.chat_message("user"):
                st.write(user_input)

        cols = st.columns(len(selected_sizes))
        for i, size in enumerate(selected_sizes):
            with cols[i]:
                st.markdown(f"**{size}**")
                tag = (
                    MODEL_SIZES[size]["tag"]
                    if any(MODEL_SIZES[size]["tag"] in m for m in raw_models)
                    else MODEL_SIZES[size]["base"]
                )
                # Stream the response directly
                res = st.write_stream(
                    stream_chat(tag, [{"role": "user", "content": user_input}])
                )

# CHAPTER 2
elif "2. Policy Binder" in selected_module:
    c_conf = config["chapter_2"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    policy_text = st.text_area("📋 Policy:", value=c_conf["default_policy"], height=150)

    # 1. FINISH BUTTON (Sidebar Persistence)
    render_module_transition(
        "2. Policy Binder",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 2. QUICK PROMPTS
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 3. CHAT INPUT & CONSOLIDATION
    typed_input = st.chat_input(c_conf["input_placeholder"])
    user_input = clicked_prompt or typed_input

    if user_input:
        # History Logic: Append manual inputs (clicks) to history
        if clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        # Render History
        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Append typed inputs to history
        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

        c1, c2 = st.columns(2)

        # LEFT: BASE (No Policy Context)
        with c1:
            st.info("🧠 Base Model")
            box = st.empty()
            full = ""
            stats = {}
            base_prompt = user_input

            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": base_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, base_prompt)
            save_interaction("Ch2", "Base", user_input, full, primary_model, stats)

        # RIGHT: RAG (With Policy Context)
        with c2:
            st.success("📘 RAG Model")
            rag_prompt = f"Context: {policy_text}\nQuestion: {user_input}"
            box = st.empty()
            full = ""
            stats = {}
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": rag_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, rag_prompt)
            save_interaction("Ch2", "RAG", user_input, full, primary_model, stats)

            # Save assistant response to history
            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )

# CHAPTER 3
elif "3. Pager & Phone" in selected_module:
    c_conf = config["chapter_3"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    c1, c2, c3 = st.columns(3)
    with c1:
        use_pager = st.toggle("Check Local Database", True)
    with c2:
        use_phone = st.toggle("Enable Web Search", True)
    with c3:
        mock_mode = st.toggle("Offline Mode (dummy search results)", False)

    # 1. SIDEBAR FINISH BUTTON (Moved up for persistence)
    render_module_transition(
        "3. Pager & Phone",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 2. QUICK PROMPTS (Clickable buttons)
    # This renders the buttons above the chat input
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 3. STANDARD CHAT INPUT
    typed_input = st.chat_input(c_conf["input_placeholder"])

    # 4. CONSOLIDATE INPUT
    user_input = clicked_prompt or typed_input

    if user_input:
        # If the user clicked a button, we manually mirror it as a chat message
        # so they can see what they "said" in the history.
        if clicked_prompt:
            # Add to history state so it survives reruns
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        # Standard History Rendering
        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # If it was typed (not clicked), it hasn't been added to history yet
        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)
        # (If it was clicked, we already added it above, so we just render the history loop)

        # --- LOGIC START ---
        ctx_str = ""
        tool = "None"

        # KEYWORD MATCHING
        # Note: We rely on the text from the button matching these keywords!
        pager_keywords = c_conf.get("keywords_pager", ["wait", "time", "page"])

        if use_pager and any(k in user_input.lower() for k in pager_keywords):
            tool = "Pager"
            ctx_str = get_live_wait_times()
        elif use_phone:
            tool = "Web"
            raw = perform_search(user_input, mock_mode)
            ctx_str = format_search_results(raw)

        c1, c2 = st.columns(2)

        # LEFT: BASE
        with c1:
            st.info("🧠 Memory")
            box = st.empty()
            full = ""
            stats = {}
            base_prompt = user_input
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": base_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, base_prompt)
            save_interaction("Ch3", "Base", user_input, full, primary_model, stats)

        # RIGHT: AGENT
        with c2:
            st.success(f"🛠️ Agent ({tool})")
            agent_prompt = f"Data: {ctx_str}\nQuestion: {user_input}"
            box = st.empty()
            full = ""
            stats = {}
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": agent_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, agent_prompt)
            save_interaction(
                "Ch3", f"Agent ({tool})", user_input, full, primary_model, stats
            )
            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )

# CHAPTER 4
elif "4. Bedside Manner" in selected_module:
    c_conf = config["chapter_4"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    # 1. PERSONA SELECTOR
    # We retrieve the keys (names) from the config
    p_names = list(c_conf["personas"].keys())
    p_sel = st.selectbox("Select Persona", p_names)

    # Get the System Prompt for the LLM
    sys_prompt = c_conf["personas"][p_sel]

    # 2. FINISH BUTTON (Moved up for persistence)
    render_module_transition(
        "4. Bedside Manner",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 3. DYNAMIC QUICK PROMPTS
    # Logic: Try to find prompts for the SPECIFIC persona.
    # If not found, fall back to "Default". If that fails, empty list.
    prompt_dict = c_conf.get("prompts", {})
    current_prompts = prompt_dict.get(p_sel, prompt_dict.get("Default", []))

    # Render the buttons
    clicked_prompt = render_quick_prompts(current_prompts)

    # 4. CHAT INPUT & CONSOLIDATION
    typed_input = st.chat_input(c_conf["input_placeholder"])
    user_input = clicked_prompt or typed_input

    if user_input:
        # History Logic: Append manual inputs (clicks) to history
        if clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        # Render History
        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Append typed inputs to history
        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

        c1, c2 = st.columns(2)

        # LEFT: DEFAULT (Control Group)
        with c1:
            st.info("🤖 Default Model")
            box = st.empty()
            full = ""
            stats = {}
            # Base model gets just the user input, no system persona
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": user_input}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, user_input)
            save_interaction("Ch4", "Default", user_input, full, primary_model, stats)

        # RIGHT: PERSONA (Experimental Group)
        with c2:
            st.success(f"🎭 {p_sel}")
            box = st.empty()
            full = ""
            stats = {}

            # Visualizer prompt (for the popover)
            visual_prompt = f"SYSTEM: {sys_prompt}\n\nUSER: {user_input}"

            # Persona model gets the System Prompt + User Input
            for ch in stream_chat(
                primary_model,
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_input},
                ],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, visual_prompt)
            save_interaction("Ch4", p_sel, user_input, full, primary_model, stats)

            # Save the Assistant's reply to history
            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )

# CHAPTER 5
elif "5. The Burnout" in selected_module:
    c_conf = config["chapter_5"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    # 1. CONTROLS
    noise_lvl = st.slider("Distraction Level", 50, 1000, 500)

    # Prepare the data
    secret = c_conf["secret_fact"]
    noise = c_conf["distraction_filler"] * noise_lvl

    # 2. FINISH BUTTON (Moved up for persistence)
    render_module_transition(
        "5. The Burnout",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 3. QUICK PROMPTS
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 4. CHAT INPUT & CONSOLIDATION
    typed_input = st.chat_input(c_conf["input_placeholder"])
    user_input = clicked_prompt or typed_input

    if user_input:
        # History Logic: Append manual inputs (clicks) to history
        if clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        # Render History
        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Append typed inputs to history
        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

        c1, c2 = st.columns(2)

        # LEFT: FOCUSED (The Control Group - Clean Context)
        with c1:
            st.info("🧠 Focused")
            # We give this model ONLY the secret and the question
            focused_prompt = f"{secret}\nQuestion: {user_input}"
            box = st.empty()
            full = ""
            stats = {}
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": focused_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, focused_prompt)
            save_interaction("Ch5", "Focused", user_input, full, primary_model, stats)

        # RIGHT: BURNED OUT (The Experimental Group - Noisy Context)
        with c2:
            st.error("🤯 Burned Out")
            # We construct the massive noise prompt
            noise_prompt = f"{secret} {noise}\nQuestion: {user_input}"
            box = st.empty()
            full = ""
            stats = {}

            # PRO TIP: We limit num_ctx (Context Window) here to FORCE the failure.
            # If the model has 4096 ctx, and we send 5000 tokens of noise,
            # the beginning (the secret) gets chopped off.
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": noise_prompt}],
                override_options={"num_ctx": 1024},  # Artificially small brain!
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, noise_prompt)
            save_interaction(
                "Ch5", "Burned Out", user_input, full, primary_model, stats
            )

            # Save assistant response to history
            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )


# CHAPTER 6
elif "6. Insider Threat" in selected_module:
    c_conf = config["chapter_6"]
    st.title(c_conf["title"])
    render_instructions("Instructions", c_conf["instructions"])

    # Show the "Poisoned" document so the user knows what they are attacking
    st.warning("⚠️ **System Context (Poisoned)**")
    st.code(c_conf["poison_text"], language="text")

    # 1. FINISH BUTTON (Moved up for persistence)
    render_module_transition(
        "6. Insider Threat",
        c_conf.get("summary", DEFAULT_SUMMARY),
        module_finish_placeholder,
    )

    # 2. RED TEAM PROMPTS
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 3. CHAT INPUT & CONSOLIDATION
    typed_input = st.chat_input(c_conf["input_placeholder"])
    user_input = clicked_prompt or typed_input

    if user_input:
        # History Logic: Append manual inputs (clicks) to history
        if clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        # Render History
        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Append typed inputs to history
        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

        c1, c2 = st.columns(2)

        # LEFT: BASE MODEL (The Control - Ignorant of the Poison)
        with c1:
            st.info("🛡️ Base (Safe)")
            box = st.empty()
            full = ""
            stats = {}
            # Base model has NO context, so it shouldn't know the password
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": user_input}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, user_input)
            save_interaction("Ch6", "Base", user_input, full, primary_model, stats)

        # RIGHT: POISONED MODEL (The Victim - Has the dangerous context)
        with c2:
            st.error("☠️ Poisoned (Compromised)")
            # We inject the "Poison Text" (which likely contains a fake password or bad instruction)
            # right before the user's question.
            poison_prompt = f"Context: {c_conf['poison_text']}\nQuestion: {user_input}"

            box = st.empty()
            full = ""
            stats = {}
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": poison_prompt}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, poison_prompt)
            save_interaction("Ch6", "Poisoned", user_input, full, primary_model, stats)

            # Save assistant response to history
            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )

# CHAPTER 7: THE FULL MONTY
elif "7. The Full Monty" in selected_module:
    c_conf = config.get("chapter_7", {})
    st.title(c_conf.get("title", "7. The Full Monty"))
    render_instructions("Instructions", c_conf.get("instructions", ""))

    # 1. CONFIGURATION LAB (The "Workbench")
    with st.expander("🛠️ Workbench: Configure Your Model", expanded=True):
        # We create 3 columns now: Persona, Files, and Tools
        col_p, col_f, col_t = st.columns([1, 1, 1])

        # A. Persona Configuration
        with col_p:
            st.subheader("1. Persona")
            custom_persona = st.text_area(
                "System Prompt",
                value=c_conf.get("default_persona", "You are a helpful AI."),
                height=200,
                help="Define who the AI is (e.g., 'You are a pirate').",
            )

        # B. Data Injection (The Files)
        with col_f:
            st.subheader("2. Knowledge")
            file_good = st.file_uploader(
                "📂 Good Data", type=["txt", "md"], key="f_good"
            )
            file_bad = st.file_uploader(
                "☠️ Poison Data", type=["txt", "md"], key="f_bad"
            )
            file_junk = st.file_uploader(
                "🗑️ Junk Data", type=["txt", "md"], key="f_junk"
            )

        # C. Tool Configuration (Pager & Phone)
        with col_t:
            st.subheader("3. Tools")
            st.caption("Enable capabilities:")
            use_pager = st.toggle(
                "Pager (Live DB)",
                value=True,
                help="Triggers on keywords: 'wait', 'time', 'page'.",
            )
            use_phone = st.toggle(
                "Phone (Web Search)",
                value=True,
                help="Performs a real DuckDuckGo search.",
            )
            mock_mode = st.toggle(
                "Mock Search (no internet)",
                value=False,
                help="Simulate search if no internet.",
            )

    # 2. PROCESS INPUTS
    # A. Read Files
    txt_good = read_uploaded_text(file_good)
    txt_bad = read_uploaded_text(file_bad)
    txt_junk = read_uploaded_text(file_junk)

    # 3. FINISH BUTTON
    render_module_transition(
        "7. The Full Monty",
        c_conf.get("summary", "Module Complete!"),
        module_finish_placeholder,
    )

    # 4. QUICK PROMPTS
    clicked_prompt = render_quick_prompts(c_conf.get("quick_prompts", []))

    # 5. CHAT INPUT
    typed_input = st.chat_input(c_conf.get("input_placeholder", "Test your system..."))
    user_input = clicked_prompt or typed_input

    if user_input:
        # History Handling
        if clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )

        for msg in st.session_state[f"history_{selected_module}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not clicked_prompt:
            st.session_state[f"history_{selected_module}"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.write(user_input)

        # --- TOOL LOGIC (The "Agent" Layer) ---
        tool_outputs = ""

        # Check Pager (Keywords)
        pager_keywords = config.get("chapter_3", {}).get(
            "keywords_pager", ["wait", "time", "page"]
        )

        if use_pager and any(k in user_input.lower() for k in pager_keywords):
            wait_data = get_live_wait_times()
            tool_outputs += f"\n--- TOOL OUTPUT: PAGER (DATABASE) ---\n{wait_data}\n"

        # Check Phone (Toggle)
        if use_phone:
            # We assume if the toggle is ON, we search for the user's input
            raw_search = perform_search(user_input, mock_mode)
            search_text = format_search_results(raw_search)
            tool_outputs += (
                f"\n--- TOOL OUTPUT: PHONE (WEB SEARCH) ---\n{search_text}\n"
            )

        # --- CONTEXT ASSEMBLY ---
        full_context_str = ""
        # Order matters! We usually put high-value tools at the top, junk at the bottom.
        if tool_outputs:
            full_context_str += tool_outputs
        if txt_good:
            full_context_str += f"\n--- TRUSTED KNOWLEDGE ---\n{txt_good}\n"
        if txt_bad:
            full_context_str += f"\n--- UNTRUSTED DATA ---\n{txt_bad}\n"
        if txt_junk:
            full_context_str += f"\n--- BACKGROUND NOISE ---\n{txt_junk}\n"

        # Metrics
        ctx_len = len(full_context_str)
        if ctx_len > 0:
            st.info(f"📚 Compiled Context: {ctx_len} chars (Files + Tools)")

        c1, c2 = st.columns(2)

        # LEFT: BASE MODEL
        with c1:
            st.info("🛡️ Base Model (Ignorant)")
            # It sees NONE of the context, files, or tools.
            box = st.empty()
            full = ""
            stats = {}
            for ch in stream_chat(
                primary_model,
                [{"role": "user", "content": user_input}],
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(stats, user_input)
            save_interaction("Ch7", "Base", user_input, full, primary_model, stats)

        # RIGHT: THE FULL MONTY
        with c2:
            st.success("🤖 The Full Monty (Omniscient)")

            final_prompt = (
                f"CONTEXT DATA:\n{full_context_str}\n\nUSER QUESTION:\n{user_input}"
            )

            box = st.empty()
            full = ""
            stats = {}

            messages = [
                {"role": "system", "content": custom_persona},
                {"role": "user", "content": final_prompt},
            ]

            for ch in stream_chat(
                primary_model,
                messages,
                stats_container=stats,
            ):
                full += ch
                box.markdown(full + "▌")
            box.markdown(full)

            render_response_metadata(
                stats, f"SYSTEM: {custom_persona}\n\n{final_prompt}"
            )
            save_interaction(
                "Ch7", "Full Monty", user_input, full, primary_model, stats
            )

            st.session_state[f"history_{selected_module}"].append(
                {"role": "assistant", "content": full}
            )
