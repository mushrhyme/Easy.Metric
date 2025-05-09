# visitor_logger.py
from pathlib import Path
from datetime import datetime
import json
import streamlit as st
import socket
import logging

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

def get_remote_ip():
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip


def log_visitor():
    """
    방문자의 IP 주소와 User Agent 정보를 JSON 파일에 기록합니다.
    """
    # 관리자 계정 체크
    if st.session_state.get('id') == '9999999' or st.session_state.get('name') == 'admin' or st.session_state.get('name') == '조유민':
        return

    log_file = Path("visitor_logs.json")

    # Get client information
    ip_address = get_remote_ip()

    if ip_address not in ["127.0.0.1", "172.17.146.217", "172.17.173.18", "172.17.17.85", "172.17.17.99"]:
        visitor_info = {
            "timestamp": datetime.now().isoformat(),
            "ip_address": ip_address,
            "page": st.query_params.get("page", "main")
        }

        # Read existing logs or create new list
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        else:
            logs = []

        # Append new visitor info
        logs.append(visitor_info)

        # Write updated logs back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        # Print current visitor info for debugging
        print(f"Visitor logged: {visitor_info}")

def log_login():
    """
    로그인한 사용자의 정보를 JSON 파일에 기록합니다.
    """
    # 관리자 계정 체크
    if st.session_state.get('id') == '9999999' or st.session_state.get('name') == 'admin' or st.session_state.get('name') == '조유민':
        return

    log_file = Path("login_logs.json")

    # Get user information from session state
    user_id = st.session_state.get('id')
    user_name = st.session_state.get('name')
    ip_address = get_remote_ip()

    if user_id and user_name:
        login_info = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_name": user_name,
            "ip_address": ip_address,
            "page": st.query_params.get("page", "main")
        }

        # Read existing logs or create new list
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        else:
            logs = []

        # Append new login info
        logs.append(login_info)

        # Write updated logs back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        print(f"Login logged: {login_info}")
