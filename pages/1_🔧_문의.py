import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="시스템 관리",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)


def load_css(css_file):
    with open(css_file, 'r') as f:
        return f.read()


# CSS 적용
st.markdown(f'<style>{load_css("/Users/macpro2/Desktop/Minitab/utils/style.css")}</style>', unsafe_allow_html=True)


def save_question(question_data, file_path='questions.json'):
    """Q&A 데이터를 JSON 파일에 저장"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        else:
            questions = []

        questions.append(question_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"질문 저장 중 오류 발생: {str(e)}")
        return False


def load_recent_error():
    """현재 로그인한 사용자의 가장 최근 에러 로그 파일을 읽어옴"""
    try:
        # 현재 로그인한 사용자 ID 확인
        current_user_id = st.session_state.get('id')
        if not current_user_id:
            return None

        # error_logs 디렉토리의 모든 에러 로그 파일 검색
        error_files = list(Path('error_logs').glob('error_*.json'))
        if not error_files:
            return None

        # 사용자의 최근 에러 찾기
        user_error_files = []
        for file in error_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                    # 현재 사용자의 에러인 경우에만 추가
                    if error_data['context']['user_id'] == current_user_id:
                        user_error_files.append((file, error_data))
            except Exception as e:
                st.warning(f"파일 {file.name} 읽기 실패: {str(e)}")
                continue

        # 사용자의 에러가 없는 경우
        if not user_error_files:
            return None

        # 타임스탬프를 기준으로 가장 최근 에러 찾기
        latest_error = max(user_error_files,
                         key=lambda x: datetime.fromisoformat(x[1]['timestamp']))
        latest_file, error_data = latest_error

        return {
            'timestamp': datetime.fromisoformat(error_data['timestamp']),
            'error_type': error_data['error_type'],
            'error_message': error_data['error_message'],
            'traceback': error_data['traceback'],
            'context': error_data['context'],
            'file_name': latest_file.name
        }
    except Exception as e:
        st.warning(f"에러 로그 읽기 실패: {str(e)}")
        return None
def display_qa_form():
    """일반 사용자를 위한 Q&A 폼"""
    st.title("❓ 문의사항 등록")

    # 로그인 상태 확인
    if 'id' not in st.session_state or 'name' not in st.session_state:
        st.warning("로그인을 먼저 해 주세요")
        return  # 로그인하지 않은 경우 여기서 함수 종료

    # 로그인된 경우 사용자 정보 표시
    user_id = st.session_state.id
    st.info(f"사용자: {st.session_state.name} ({user_id})")

    # 최근 에러 로그 확인
    recent_error = load_recent_error()
    if recent_error and recent_error['context']['user_id'] == user_id:
        with st.expander("최근 발생한 에러 정보", expanded=True):
            st.info(f"""
📅 발생 시간: {recent_error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
🔍 에러 유형: {recent_error['error_type']}
❌ 에러 메시지: {recent_error['error_message']}
📍 발생 위치: {recent_error['context']['current_tab']}
""")
            if st.toggle("상세 트레이스백 보기"):
                st.code(recent_error['traceback'])

            # 에러 정보를 폼에 자동으로 채우기 위한 버튼
            if st.button("이 에러에 대해 문의하기"):
                st.session_state.fill_error_info = recent_error

    # 문의사항 입력 폼
    with st.form("question_form"):
        # 자동 채우기 기본값 설정
        default_title = ""
        default_category = "일반 문의"
        default_content = ""

        # 에러 정보 자동 채우기가 요청된 경우
        if hasattr(st.session_state, 'fill_error_info') and st.session_state.fill_error_info:
            error = st.session_state.fill_error_info
            default_title = f"에러 발생 문의: {error['error_type']}"
            default_category = "버그 신고"
            default_content = f"""
발생 시간: {error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
에러 유형: {error['error_type']}
에러 메시지: {error['error_message']}
발생 위치: {error['context']['current_tab']}

[자동 생성된 에러 정보]
로그 파일: {error['file_name']}

상세 트레이스백:
{error['traceback']}

추가 설명:
"""

        title = st.text_input("제목", value=default_title)
        category = st.selectbox(
            "카테고리",
            ["일반 문의", "버그 신고", "기능 제안", "기타"],
            index=["일반 문의", "버그 신고", "기능 제안", "기타"].index(default_category)
        )
        content = st.text_area("내용", value=default_content, height=300)

        submitted = st.form_submit_button("제출하기")

        if submitted:
            if not title.strip() or not content.strip():
                st.error("제목과 내용을 모두 입력해주세요.")
                return

            question_data = {
                "title": title,
                "category": category,
                "content": content,
                "user_id": st.session_state.id,
                "user_name": st.session_state.name,
                "timestamp": datetime.now().isoformat(),
                "status": "접수됨"
            }

            if save_question(question_data):
                st.success("문의사항이 성공적으로 등록되었습니다!")
                # 성공적으로 저장된 후에 session_state 초기화
                if hasattr(st.session_state, 'fill_error_info'):
                    del st.session_state.fill_error_info
@st.cache_data(ttl=3600)  # 1시간 캐시
def load_error_logs(log_dir='error_logs'):
    """모든 에러 로그 파일을 읽어서 DataFrame으로 변환"""
    error_logs = []
    for file in Path(log_dir).glob('error_*.json'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                log = json.load(f)
                error_info = {
                    'timestamp': datetime.fromisoformat(log['timestamp']),
                    'error_type': log['error_type'],
                    'error_message': log['error_message'],
                    'user_id': log['context']['user_id'],
                    'name': log['context']['user_name'],
                    'current_tab': log['context']['current_tab'],
                    'file_name': file.name,
                    'traceback': log['traceback']
                }
                error_logs.append(error_info)
        except Exception as e:
            st.warning(f"로그 파일 {file.name} 읽기 실패: {str(e)}")
    return pd.DataFrame(error_logs)

def display_statistics(filtered_df):
    """통계 정보를 표시하는 함수"""
    # 기본 통계
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 에러 수", len(filtered_df))
    with col2:
        st.metric("영향 받은 사용자 수", filtered_df['user_id'].nunique())
    with col3:
        st.metric("고유 에러 타입", filtered_df['error_type'].nunique())

    # 시간별 에러 발생 추이
    st.subheader("📈 시간별 에러 발생 추이")
    fig = px.line(
        filtered_df.set_index('timestamp').resample('D').size(),
        title="일별 에러 발생 횟수"
    )
    st.plotly_chart(fig)

    # 에러 타입별 분포
    st.subheader("📊 에러 타입 분포")
    error_type_counts = filtered_df['error_type'].value_counts()
    fig = px.pie(values=error_type_counts.values, names=error_type_counts.index)
    st.plotly_chart(fig)


def display_error_list(df):
    """상세 에러 목록을 표시하는 함수"""
    st.subheader("🔍 필터 옵션")

    # 날짜 필터
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

    # 날짜로 필터링
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    # 사용자 이름 검색
    user_search = st.text_input("사용자 이름 검색", "")
    if user_search:
        filtered_df = filtered_df[filtered_df['name'].str.contains(user_search, case=False, na=False)]

    # 에러 타입 필터
    error_types = ['전체'] + list(filtered_df['error_type'].unique())
    selected_type = st.selectbox("에러 타입 선택", error_types)
    if selected_type != '전체':
        filtered_df = filtered_df[filtered_df['error_type'] == selected_type]

    # 정렬 옵션
    sort_options = {
        '최신순': ('timestamp', False),
        '오래된순': ('timestamp', True),
        '사용자명순': ('name', True),
        '에러타입순': ('error_type', True)
    }
    selected_sort = st.selectbox("정렬 기준", list(sort_options.keys()))
    sort_column, ascending = sort_options[selected_sort]
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

    # 페이지네이션 설정
    items_per_page = 10  # 고정값으로 설정
    total_pages = len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0)

    if total_pages > 0:
        col1, col2 = st.columns([7, 3])
        with col1:
            st.write(f"총 {len(filtered_df)}개의 항목")
        with col2:
            page_number = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, key="error_page")

        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        st.write("검색 결과가 없습니다.")
        return
    # 구분선 추가
    st.markdown("---")

    # 에러 목록을 테이블로 먼저 보여주기
    st.dataframe(
        page_df[['timestamp', 'name', 'error_type', 'error_message']]
        .rename(columns={
            'timestamp': '발생시간',
            'name': '사용자명',
            'error_type': '에러유형',
            'error_message': '에러메시지'
        }),
        hide_index=True
    )

    # 선택된 에러의 상세 정보 표시
    selected_error = st.selectbox(
        "상세 정보를 볼 에러를 선택하세요",
        page_df.apply(
            lambda x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {x['name']} - {x['error_type']}",
            axis=1
        ).tolist()
    )

    if selected_error:
        selected_row = page_df[page_df.apply(
            lambda
                x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {x['name']} - {x['error_type']}" == selected_error,
            axis=1
        )].iloc[0]

        with st.expander("🔍 상세 정보", expanded=True):
            tab1, tab2 = st.tabs(["기본 정보", "전체 트레이스백"])

            with tab1:
                st.write(f"**사용자:** {selected_row['user_id']} ({selected_row['name']})")
                st.write(f"**현재 탭:** {selected_row['current_tab']}")
                st.write(f"**에러 메시지:** {selected_row['error_message']}")

            with tab2:
                st.code(selected_row['traceback'])


def load_visitor_logs(file_path='visitor_logs.json'):
    """방문자 로그를 DataFrame으로 변환"""
    try:
        with open(file_path, 'r') as f:
            logs = json.load(f)
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.warning(f"방문자 로그 읽기 실패: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'ip_address'])


def display_visitor_stats(df):
    """방문자 통계를 표시하는 함수"""
    if df.empty:
        st.info("방문자 로그가 없습니다.")
        return

    # 기본 통계
    col1, col2, col3 = st.columns(3)

    total_visits = len(df)
    unique_ips = df['ip_address'].nunique()

    # 일평균 방문자 수 계산 로직 수정
    date_diff = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400  # 일 단위로 변환
    if date_diff < 1:  # 하루 미만인 경우
        avg_daily_visits = total_visits  # 당일 방문자 수를 그대로 사용
    else:
        avg_daily_visits = round(total_visits / date_diff, 1)

    with col1:
        st.metric("총 방문 수", total_visits)
    with col2:
        st.metric("고유 방문자 수", unique_ips)
    with col3:
        if date_diff < 1:
            st.metric("오늘의 방문", total_visits)
        else:
            st.metric("일평균 방문", avg_daily_visits)

    # 시간대별 방문자 수 차트
    st.subheader("📊 시간대별 방문자 수")
    hourly_visits = df.groupby(df['timestamp'].dt.hour)['ip_address'].count()

    # 없는 시간대 0으로 채우기
    all_hours = pd.Series(0, index=range(24))
    hourly_visits = hourly_visits.add(all_hours, fill_value=0)

    fig_hourly = px.bar(
        x=hourly_visits.index,
        y=hourly_visits.values,
        labels={'x': '시간', 'y': '방문자 수'},
        title='시간대별 방문자 분포'
    )
    fig_hourly.update_xaxes(ticktext=[f"{i}시" for i in range(24)], tickvals=list(range(24)))
    st.plotly_chart(fig_hourly)

    # 일별 방문자 수 추이
    st.subheader("📈 일별 방문자 추이")
    daily_visits = df.groupby(df['timestamp'].dt.date).agg({
        'ip_address': ['count', 'nunique']
    }).reset_index()
    daily_visits.columns = ['date', 'total_visits', 'unique_visitors']

    fig_daily = px.line(
        daily_visits,
        x='date',
        y=['total_visits', 'unique_visitors'],
        labels={
            'value': '방문자 수',
            'date': '날짜',
            'variable': '구분'
        },
        title='일별 방문자 추이'
    )
    # 범례 이름 변경
    fig_daily.for_each_trace(lambda t: t.update(
        name={'total_visits': '총 방문수', 'unique_visitors': '고유 방문자'}[t.name],
        showlegend=True
    ))
    fig_daily.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_daily)

    # IP 주소별 방문 횟수
    st.subheader("🔍 자주 방문한 IP")
    top_ips = df['ip_address'].value_counts().reset_index()
    top_ips.columns = ['IP 주소', '방문 횟수']

    fig_ips = px.bar(
        top_ips,
        x='IP 주소',
        y='방문 횟수',
        title='방문자 IP별 방문 횟수'
    )
    st.plotly_chart(fig_ips)

    # 방문 기록 테이블
    st.subheader("📋 상세 방문 기록")
    st.dataframe(
        df.sort_values('timestamp', ascending=False)
        .rename(columns={
            'timestamp': '방문 시간',
            'ip_address': 'IP 주소'
        }),
        hide_index=True
    )


@st.cache_data(ttl=3600)  # 1시간 캐시
def load_questions(file_path='questions.json'):
    """문의사항 데이터를 DataFrame으로 변환"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            df = pd.DataFrame(questions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame(columns=['timestamp', 'title', 'category', 'content', 'user_id', 'user_name', 'status'])
    except Exception as e:
        st.warning(f"문의사항 데이터 읽기 실패: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'title', 'category', 'content', 'user_id', 'user_name', 'status'])


def update_question_status(questions, index, new_status, file_path='questions.json'):
    """문의사항 상태 업데이트"""
    try:
        questions_copy = questions.copy()
        questions_copy['timestamp'] = questions_copy['timestamp'].astype(str)
        questions_copy.at[index, 'status'] = new_status
        questions_dict = questions_copy.to_dict('records')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions_dict, f, ensure_ascii=False, indent=2)

        # 캐시 즉시 초기화
        load_questions.clear()
        return True
    except Exception as e:
        st.error(f"상태 업데이트 중 오류 발생: {str(e)}")
        return False


def display_questions_dashboard():
    """문의사항 관리 대시보드"""
    st.subheader("📬 문의사항 관리")

    # 데이터 로드
    df = load_questions()
    if df.empty:
        st.info("등록된 문의사항이 없습니다.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 필터 옵션 섹션
    st.subheader("🔍 필터 옵션")

    # 날짜 필터
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

    # 날짜로 필터링
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    # 추가 필터 옵션
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = ['전체'] + list(filtered_df['category'].unique())
        selected_category = st.selectbox("카테고리", categories)

    with col2:
        statuses = ['전체'] + list(filtered_df['status'].unique())
        selected_status = st.selectbox("처리상태", statuses)

    with col3:
        user_search = st.text_input("사용자명 검색")

    # 필터링 적용
    if selected_category != '전체':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_status != '전체':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    if user_search:
        filtered_df = filtered_df[filtered_df['user_name'].str.contains(user_search, case=False, na=False)]

    # 정렬 옵션
    sort_options = {
        '최신순': ('timestamp', False),
        '오래된순': ('timestamp', True),
        '사용자명순': ('user_name', True),
        '카테고리순': ('category', True)
    }
    selected_sort = st.selectbox("정렬 기준", list(sort_options.keys()))
    sort_column, ascending = sort_options[selected_sort]
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

    # 페이지네이션 설정
    items_per_page = 10  # 고정값으로 설정
    total_pages = len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0)

    if total_pages > 0:
        col1, col2 = st.columns([7, 3])
        with col1:
            st.write(f"총 {len(filtered_df)}개의 항목")
        with col2:
            page_number = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, key="question_page")

        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        st.write("검색 결과가 없습니다.")
        return

    # 통계 요약
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 문의", len(df))
    with col2:
        st.metric("처리 대기", len(df[df['status'] == '접수됨']))
    with col3:
        st.metric("처리 중", len(df[df['status'] == '처리중']))
    with col4:
        st.metric("처리 완료", len(df[df['status'] == '완료']))

    # 문의사항 목록을 테이블로 먼저 보여주기
    st.markdown("---")
    st.dataframe(
        page_df[['timestamp', 'user_name', 'category', 'title', 'status']]
        .rename(columns={
            'timestamp': '등록시간',
            'user_name': '작성자',
            'category': '카테고리',
            'title': '제목',
            'status': '상태'
        }),
        hide_index=True
    )

    # 선택된 문의사항의 상세 정보 표시
    st.markdown("---")
    selected_question = st.selectbox(
        "상세 정보를 볼 문의사항을 선택하세요",
        page_df.apply(
            lambda x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M')} - {x['user_name']} - {x['title']}",
            axis=1
        ).tolist()
    )

    if selected_question:
        selected_row = page_df[page_df.apply(
            lambda
                x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M')} - {x['user_name']} - {x['title']}" == selected_question,
            axis=1
        )].iloc[0]

        with st.expander("📝 문의사항 상세 정보", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.write(f"**작성자:** {selected_row['user_name']} ({selected_row['user_id']})")
                st.write(f"**카테고리:** {selected_row['category']}")

            with col2:
                st.write(f"**등록시간:** {selected_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**현재상태:** {selected_row['status']}")

            with col3:
                new_status = st.selectbox(
                    "상태 변경",
                    ['접수됨', '처리중', '완료'],
                    index=['접수됨', '처리중', '완료'].index(selected_row['status']),
                    key=f"status_{selected_row.name}"
                )
                if new_status != selected_row['status']:
                    if st.button("상태 업데이트", key=f"update_{selected_row.name}"):
                        if update_question_status(df, selected_row.name, new_status):
                            st.success("상태가 업데이트되었습니다.")
                            st.rerun()

            st.markdown("---")
            st.markdown("**제목:**")
            st.markdown(selected_row['title'])
            st.markdown("**문의내용:**")
            st.markdown(selected_row['content'])
def display_error_dashboard():
    st.title("🔍 시스템 모니터링 대시보드")

    # 관리자 확인
    if st.session_state.id not in ['2024087']:  # 관리자 ID 목록
        st.error("접근 권한이 없습니다.")
        return
    # 캐시 초기화 버튼
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("캐시 초기화", type="primary"):
            load_error_logs.clear()
            load_questions.clear()
            st.success("캐시가 초기화되었습니다!")
            st.rerun()
    with col2:
        st.info("데이터가 제대로 반영되지 않으면 캐시 초기화 버튼을 눌러주세요.")

    # 데이터 로드
    error_df = load_error_logs()
    visitor_df = load_visitor_logs()

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 에러 통계", "📝 상세 에러 목록", "👥 방문자 분석", "💬 문의사항 관리"])

    with tab1:
        display_statistics(error_df)

    with tab2:
        display_error_list(error_df)

    with tab3:
        display_visitor_stats(visitor_df)

    with tab4:
        display_questions_dashboard()

def main():
    # 관리자 확인
    is_admin = st.session_state.get('id') in ['2024087']

    if is_admin:
        display_error_dashboard()
    else:
        display_qa_form()


if __name__ == "__main__":
    main()