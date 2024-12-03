import os
import glob
import json

def read_json_safely(file_path):
    try:
        # 방법 1: 기본 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print("첫 번째 시도 실패, 다른 방법 시도...")

        try:
            # 방법 2: BOM 제거
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print("두 번째 시도 실패, 다른 방법 시도...")

            try:
                # 방법 3: 문자열로 읽은 후 공백 제거
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                data = json.loads(content)
                return data
            except json.JSONDecodeError as e:
                print("세 번째 시도 실패")
                print(f"오류 위치: {e.pos}")
                print(f"오류 라인: {e.lineno}")
                print(f"오류 컬럼: {e.colno}")
                print(f"오류 메시지: {str(e)}")

                # 파일 내용 출력
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print("\n파일 처음 500자:")
                print(content[:500])
                return None


filelist = list(filter(lambda x: '20241129_1035' in x, glob.glob("./error_logs/*")))
for filename in filelist:
    print("="*100)
    print(filename)
    print("=" * 100)
    result = read_json_safely(filename)
    print(result["traceback"])




