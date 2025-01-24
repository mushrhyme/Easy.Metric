import os
import json
import re


def create_users_json():
    # DB 폴더 경로
    db_path = os.path.join(os.getcwd(), "DB")

    # 사용자 정보를 저장할 리스트
    users = []

    # DB 폴더 내의 모든 폴더 검사
    for folder in os.listdir(db_path):
        # 정규표현식을 사용하여 폴더명에서 ID와 이름 추출
        # 2024087(조유민) 형식의 폴더명에서 추출
        match = re.match(r'(\d+)\(([^)]+)\)', folder)

        if match:
            user_id = match.group(1)  # ID (숫자 부분)
            user_name = match.group(2)  # 이름 (괄호 안의 부분)

            # 사용자 정보를 딕셔너리로 만들어 리스트에 추가
            users.append({
                "id": user_id,
                "name": user_name
            })

    # JSON 형식으로 변환
    users_data = {"users": users}

    # JSON 파일 생성
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_data, f, ensure_ascii=False, indent=4)

    print(f"users.json 파일이 생성되었습니다. 총 {len(users)}명의 사용자가 등록되었습니다.")


# 함수 실행
if __name__ == "__main__":
    create_users_json()