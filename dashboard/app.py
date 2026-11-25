"""Décroche AI — bilingual telecom customer-support dashboard."""

from pathlib import Path
import subprocess
import sys
import tempfile

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.support import TelecomSupportAgent

st.set_page_config(
    page_title="Décroche AI | Customer Support",
    page_icon="☎",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --navy:#08172b; --blue:#1677ff; --cyan:#37b7c3; --ink:#15233c;
            --muted:#667085; --line:#e7edf5; --surface:#f4f7fb; }
    .stApp { background:
        radial-gradient(circle at 92% 8%, rgba(55,183,195,.11), transparent 20rem),
        linear-gradient(135deg, #f7f9fc 0%, #eef3f9 100%);
    }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem; }
    .block-container { max-width: 1180px; padding-top: 1rem; padding-bottom: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .topbar { display:flex; align-items:center; justify-content:space-between;
               background:var(--navy); color:white; border-radius:16px;
               padding:18px 22px; margin-bottom:18px;
               box-shadow:0 14px 35px rgba(8,23,43,.14); }
    .brand { display:flex; align-items:center; gap:12px; }
    .brandmark { width:44px; height:44px; border-radius:12px; display:grid;
                 place-items:center; background:linear-gradient(135deg,#1677ff,#37b7c3);
                 font-size:22px; }
    .brand h1 { font-size:21px; margin:0; line-height:1.1; }
    .brand p { margin:4px 0 0; color:#bac8db; font-size:12px; }
    .live-pill { background:#11335c; color:#b9e9ff; border:1px solid #28527b;
                 border-radius:999px; padding:8px 12px; font-size:12px; }
    .section-card { background:white; border:1px solid var(--line); border-radius:16px;
                    padding:18px; box-shadow:0 5px 18px rgba(15,35,67,.04); }
    .section-title { color:var(--ink); font-size:16px; font-weight:700; margin-bottom:12px; }
    .step-row { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:16px; }
    .step { background:white; border:1px solid var(--line); border-radius:12px;
            padding:10px 8px; text-align:center; color:var(--muted); font-size:11px; }
    .step strong { display:block; color:var(--ink); font-size:13px; margin-top:3px; }
    .step.active { border-color:#1677ff; background:#eef5ff; }
    .metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .metric { background:#f8fafc; border:1px solid var(--line); border-radius:11px;
              padding:10px 12px; }
    .metric label { display:block; color:var(--muted); font-size:11px; margin-bottom:3px; }
    .metric span { color:var(--ink); font-size:14px; font-weight:700; }
    .message { border-radius:12px; padding:11px 13px; margin:7px 0; font-size:13px; }
    .customer-msg { background:#eef5ff; border-left:3px solid #1677ff; }
    .agent-msg { background:#effaf8; border-left:3px solid #37b7c3; }
    [data-testid="stTextInput"] input { border-radius:10px; }
    div[data-testid="stAudioInput"] { border:1px dashed #b9c7dc; border-radius:14px;
                                      background:#fbfdff; padding:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

language = st.sidebar.radio("Conversation language", ["Français", "English"], index=0)
language_code = "fr" if language == "Français" else "en"

if "agent" not in st.session_state:
    st.session_state.agent = TelecomSupportAgent(language_code)
if st.session_state.agent.language != language_code:
    st.session_state.agent = TelecomSupportAgent(language_code)
for key in ("last_audio", "response_audio", "response_audio_format"):
    if key not in st.session_state:
        st.session_state[key] = None

title_fr = "Technicien support mobile"
title_en = "Mobile Support Agent"
subtitle_fr = "Assistance réseau, SIM, données et facturation"
subtitle_en = "Network, SIM, data and billing assistance"

st.markdown(
    f"""
    <div class="topbar">
      <div class="brand">
        <div class="brandmark">☎</div>
        <div><h1>Décroche AI</h1><p>{title_fr if language_code == 'fr' else title_en}</p></div>
      </div>
      <div class="live-pill">● {('CALL CONNECTÉ' if language_code == 'fr' else 'CALL CONNECTED')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="step-row">
      <div class="step active"><span>01</span><strong>{'Appel' if language_code == 'fr' else 'Call'}</strong></div>
      <div class="step active"><span>02</span><strong>{'Transcription' if language_code == 'fr' else 'Transcript'}</strong></div>
      <div class="step active"><span>03</span><strong>{'Diagnostic' if language_code == 'fr' else 'Diagnose'}</strong></div>
      <div class="step"><span>04</span><strong>{'Dossier' if language_code == 'fr' else 'Case'}</strong></div>
      <div class="step"><span>05</span><strong>{'Résolution' if language_code == 'fr' else 'Resolution'}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], gap="medium")

with left:
    st.markdown(
        f'<div class="section-card"><div class="section-title">{("Parler au support" if language_code == "fr" else "Speak to customer support")}</div>',
        unsafe_allow_html=True,
    )
    audio = st.audio_input(
        "Record your problem" if language_code == "en" else "Enregistrer le problème"
    )
    st.caption(
        "Microphone: allow browser permission, record 3–10 seconds, then select the recording."
        if language_code == "en"
        else "Microphone : autorisez l’accès dans le navigateur, enregistrez 3 à 10 secondes, puis validez."
    )
    if audio:
        st.audio(audio)
        if st.button("Transcrire et répondre" if language_code == "fr" else "Transcribe and answer", type="primary"):
            audio_name = getattr(audio, "name", "recording.wav")
            suffix = Path(str(audio_name)).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
                stream.write(audio.getvalue())
                audio_path = Path(stream.name)
            try:
                from backend.config import settings
                from backend.stt import SpeechToText

                with st.status(
                    "Transcription du message..." if language_code == "fr" else "Transcribing recording...",
                    expanded=False,
                ):
                    transcription = SpeechToText(
                        settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type,
                    ).transcribe(
                        audio_path, language=language_code
                    )
                st.session_state.last_audio = transcription.text
                reply = st.session_state.agent.reply(transcription.text)
                response_path = ROOT / "output" / "dashboard_support_response.wav"
                try:
                    if settings.elevenlabs_api_key:
                        from backend.tts import TextToSpeech
                        tts_path = response_path.with_suffix(".mp3")
                        TextToSpeech().synthesize(reply, language_code, tts_path)
                    else:
                        from backend.local_tts import synthesize_local
                        tts_path = synthesize_local(reply, language_code, response_path)
                    st.session_state.response_audio = tts_path.read_bytes()
                    st.session_state.response_audio_format = tts_path.suffix.lstrip(".")
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    st.warning(str(error))
                st.rerun()
            except RuntimeError as error:
                st.error(str(error))
            finally:
                audio_path.unlink(missing_ok=True)

    st.write("")
    placeholder = (
        "Ma carte SIM n'a plus de réseau"
        if language_code == "fr"
        else "My mobile SIM has no network"
    )
    typed = st.text_input(
        "Message test" if language_code == "fr" else "Test message",
        placeholder=placeholder,
    )
    if st.button("Envoyer" if language_code == "fr" else "Send", disabled=not typed):
        reply = st.session_state.agent.reply(typed)
        try:
            from backend.local_tts import synthesize_local
            response_path = ROOT / "output" / "dashboard_support_text_response.wav"
            tts_path = synthesize_local(reply, language_code, response_path)
            st.session_state.response_audio = tts_path.read_bytes()
            st.session_state.response_audio_format = "wav"
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            st.warning(str(error))
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    state = st.session_state.agent.state
    labels = (
        ("Problème" if language_code == "fr" else "Problem", state.issue or state.issue_category),
        ("Catégorie" if language_code == "fr" else "Category", state.issue_category),
        ("Mobile", state.mobile_number),
        ("Nom" if language_code == "fr" else "Name", state.customer_name),
        ("Ville / secteur" if language_code == "fr" else "City / area", state.location),
        ("Statut" if language_code == "fr" else "Status",
         ("Résolu" if language_code == "fr" else "Resolved") if state.resolved
         else ("En cours" if language_code == "fr" else "In progress")),
    )
    metric_html = "".join(
        f'<div class="metric"><label>{label}</label><span>{value or ("À préciser" if language_code == "fr" else "To be provided")}</span></div>'
        for label, value in labels
    )
    st.markdown(
        f'<div class="section-card"><div class="section-title">'
        f'{"Dossier d’assistance" if language_code == "fr" else "Support case"}'
        f'</div><div class="metric-grid">{metric_html}</div></div>',
        unsafe_allow_html=True,
    )
    if state.resolved:
        st.success(
            f"{'Référence' if language_code == 'fr' else 'Reference'}: {state.case_reference}"
        )

st.write("")
conversation_col, audio_col = st.columns([1.55, 1], gap="medium")
with conversation_col:
    st.markdown(
        f'<div class="section-card"><div class="section-title">'
        f'{"Conversation" if language_code == "fr" else "Conversation"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("last_audio"):
        st.caption("Dernière transcription" if language_code == "fr" else "Latest transcription")
        st.write(st.session_state.last_audio)
    if state.messages:
        for message in state.messages:
            css = "customer-msg" if message["role"] == "customer" else "agent-msg"
            role = (
                ("Client" if message["role"] == "customer" else "Agent")
                if language_code == "fr"
                else ("Customer" if message["role"] == "customer" else "Agent")
            )
            st.markdown(
                f'<div class="message {css}"><b>{role}</b><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption(
            "Aucun échange pour le moment."
            if language_code == "fr"
            else "No exchanges yet."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with audio_col:
    st.markdown(
        f'<div class="section-card"><div class="section-title">'
        f'{"Réponse vocale" if language_code == "fr" else "Voice response"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("response_audio"):
        st.audio(
            st.session_state.response_audio,
            format=f"audio/{st.session_state.response_audio_format}",
        )
    else:
        st.caption(
            "La réponse audio apparaîtra ici."
            if language_code == "fr"
            else "The agent voice response will appear here."
        )
    if st.button("Nouvel appel" if language_code == "fr" else "New call"):
        st.session_state.agent = TelecomSupportAgent(language_code)
        st.session_state.last_audio = None
        st.session_state.response_audio = None
        st.session_state.response_audio_format = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
