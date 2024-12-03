# error_handler.py

import logging
import traceback
from datetime import datetime
import os
import json
import streamlit as st

class StreamlitErrorHandler:
    def __init__(self, log_dir='error_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger('streamlit_app')
        self.logger.setLevel(logging.ERROR)
        # 기존 핸들러 제거
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 파일 핸들러 설정
        log_file = os.path.join(log_dir, 'app_errors.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def handle_error(self, error, context=None):
        """에러 처리 및 로깅"""
        # 기본 컨텍스트 정보 설정
        error_context = {
            'user_id': st.session_state.get('id', ''),
            'user_name': st.session_state.get('name', ''),
            'current_tab': st.session_state.get('tabs_', ''),
            'session_state': {
                key: str(value) for key, value in st.session_state.items()
                if key not in ['df', 'base_df', 'stats_df', 'diff_df', 'test_df']  # 큰 데이터프레임 제외
            }
        }
        
        # 추가 컨텍스트 정보 병합
        if context:
            error_context.update(context)

        # 에러 정보 구성
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': error_context
        }
        
        # 로그 파일에 기록
        self.logger.error(f"Error occurred: {error_info['error_message']}")
        
        # 상세 에러 정보를 JSON 파일로 저장
        error_file = os.path.join(
            self.log_dir, 
            f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        # 사용자에게 에러 메시지 표시
        if st.session_state.get('id') == '':  # 로그인하지 않은 사용자
            st.error("오류가 발생했습니다. 로그인 후 다시 시도해주세요.")
        else:
            st.error(f"오류가 발생했습니다. 관리자에게 문의해주세요.\n에러 로그: {error_file}")

def main_with_error_handling(main_func):
    """메인 함수를 에러 핸들링으로 감싸는 데코레이터"""
    error_handler = StreamlitErrorHandler()
    
    def wrapper():
        try:
            main_func()
        except Exception as e:
            error_handler.handle_error(e)
    
    return wrapper
